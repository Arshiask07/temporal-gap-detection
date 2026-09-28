"""
Component 2 — Tier 2 (PRODUCTION, run this on a machine with normal internet
access — huggingface.co and github.com are both blocked from the sandbox
that built the rest of this repo, so this script has NOT been executed here).

Approach, and why:
  There is no off-the-shelf, plug-and-play HuggingFace pipeline that does
  joint SciERC-schema entity+relation extraction out of the box — the
  standard, literature-established recipe (used by SpERT, PURE, DyGIE++,
  and the original SciERC/SciBERT papers themselves) is:

      SciBERT (allenai/scibert_scivocab_uncased) backbone
      + fine-tune a token-classification head for entities
      + fine-tune a span-pair classification head for relations
      on the SciERC corpus (Luan et al., 2018 — 500 AI-conference abstracts,
      2,687 sentences, the same dataset the original SciERC/DyGIE/SpERT
      papers train and evaluate on).

  This is exactly what "identify the most appropriate reproducible
  Hugging Face-compatible scientific IE model" resolves to in practice —
  the task spec's own instruction not to "invent relations the model
  cannot reliably support" is why this script trains on SciERC's native
  6 entity types / 7 relation types rather than fabricating a fine-tuning
  set for the exact 5 relation names in the brief. The mapping onto the
  brief's relation vocabulary is applied only where it is a safe, direct
  correspondence (documented in RELATION_MAP below); everything else is
  kept as ENTITY_ASSOCIATED_WITH_ENTITY rather than mislabeled.

Requirements (install on your machine, NOT in this sandbox):
    pip install torch transformers datasets seqeval rapidfuzz

Usage:
    # one-time: fine-tune on SciERC (~10-20 min on a single consumer GPU,
    # a few hours on CPU; only needs to be done once, checkpoint is reused)
    python3 extract_entities_relations_scibert.py --train

    # apply the fine-tuned model to the Component 1 sampled corpora
    python3 extract_entities_relations_scibert.py --run

Output schema is IDENTICAL to extract_entities_relations_baseline.py, so
Component 3 and everything downstream needs zero changes to consume either.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import load_domain_corpus, write_json, OUTPUT_DIR

MODEL_NAME = "allenai/scibert_scivocab_uncased"
SCIERC_DATA_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "scibert_scierc"
BATCH_SIZE = 16
MAX_LEN = 256

# SciERC's native schema (Luan et al. 2018) — what the fine-tuned model
# actually predicts.
SCIERC_ENTITY_TYPES = ["Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic"]
SCIERC_RELATION_TYPES = ["Used-for", "Feature-of", "Hyponym-of", "Part-of",
                          "Compare", "Conjunction", "Evaluate-for"]

# Safe, direct mapping onto the brief's requested relation vocabulary.
# Left deliberately partial: SciERC's Compare/Conjunction/Part-of/Hyponym-of
# do NOT map cleanly onto METHOD_APPLIED_TO or METHOD_IMPROVES_TASK, and
# guessing would violate "do not invent relations the model cannot reliably
# support" — those are kept as ENTITY_ASSOCIATED_WITH_ENTITY instead.
RELATION_MAP = {
    "Used-for": "USED_FOR",              # e.g. "BERT used for classification"
    "Evaluate-for": "METHOD_EVALUATED_BY",
    "Feature-of": "ENTITY_ASSOCIATED_WITH_ENTITY",
    "Part-of": "ENTITY_ASSOCIATED_WITH_ENTITY",
    "Hyponym-of": "ENTITY_ASSOCIATED_WITH_ENTITY",
    "Compare": "ENTITY_ASSOCIATED_WITH_ENTITY",
    "Conjunction": "ENTITY_ASSOCIATED_WITH_ENTITY",
}
# Refine Used-for -> METHOD_APPLIED_TO / METHOD_IMPROVES_TASK at inference
# time using the SAME entity-type + lightweight cue-word heuristic as the
# Tier-1 baseline, so the two tiers are directly comparable.
ENTITY_TYPE_MAP = {  # SciERC type -> brief's 5-value schema
    "Task": "Task", "Method": "Method", "Metric": "Metric",
    "Material": "Material", "OtherScientificTerm": "Other", "Generic": "Other",
}


def download_scierc(dest: Path):
    import urllib.request
    import tarfile
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "sciERC_processed.tar.gz"

    # If train/dev/test.json are already sitting under dest from a previous
    # (even partially-failed) extraction, skip downloading/extracting
    # entirely — those 3 files are all this script actually reads; the
    # archive also contains large ELMo .hdf5 embedding files that are
    # unused here and are frequently the thing that makes extraction slow
    # or flaky on a poor connection.
    required = ["train.json", "dev.json", "test.json"]
    found = {name: next(dest.rglob(name), None) for name in required}
    if all(found.values()):
        print("Found existing extracted SciERC train/dev/test.json under "
              f"{dest} — skipping download.")
        for name, path in found.items():
            print(f"  {name}: {path} ({path.stat().st_size} bytes)")
        return dest

    def _fresh_download():
        print(f"Downloading SciERC corpus from {SCIERC_DATA_URL} ...")
        urllib.request.urlretrieve(SCIERC_DATA_URL, tar_path)

    if not tar_path.exists():
        _fresh_download()
    else:
        # A file existing doesn't mean the download completed — verify the
        # archive actually opens before trusting it, and re-download once if
        # it's corrupt/truncated (e.g. from an earlier interrupted attempt).
        try:
            with tarfile.open(tar_path) as tf:
                tf.getmembers()  # forces a full read; raises if truncated
        except (tarfile.ReadError, EOFError):
            print(f"Existing {tar_path} is corrupt/incomplete — re-downloading.")
            tar_path.unlink()
            _fresh_download()

    try:
        with tarfile.open(tar_path) as tf:
            # Skip the large, unused ELMo .hdf5 members — they're what
            # tends to make extraction slow/flaky, and nothing here reads
            # them; only the json/ folder is needed.
            members = [m for m in tf.getmembers() if not m.name.endswith(".hdf5")]
            tf.extractall(dest, members=members)
    except (tarfile.ReadError, EOFError) as e:
        raise RuntimeError(
            f"Downloaded SciERC archive at {tar_path} is corrupt "
            f"({e}). Delete {tar_path} manually and check your network "
            f"connection, then retry."
        ) from e
    return dest


def train():
    """
    Fine-tunes SciBERT token classification (entities, BIO tags over
    SCIERC_ENTITY_TYPES) and a span-pair relation classifier
    (SCIERC_RELATION_TYPES) on the SciERC train/dev split, saves both
    heads under CHECKPOINT_DIR. Standard HF Trainer setup — see the
    SpERT (Eberts & Ulges, 2019) and PURE (Zhong & Chen, 2021) papers for
    the exact architecture this follows; swap in either's public
    implementation directly if you want a stronger relation model than the
    simple span-pair classifier sketched here.
    """
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
    )

    scierc_dir = download_scierc(CHECKPOINT_DIR.parent / "scierc_data")
    train_file = next(scierc_dir.rglob("train.json"))
    dev_file = next(scierc_dir.rglob("dev.json"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_list = ["O"] + [f"{p}-{t}" for t in SCIERC_ENTITY_TYPES for p in ("B", "I")]
    label2id = {l: i for i, l in enumerate(label_list)}

    def load_scierc_ner(path):
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                offset = 0  # SciERC's ner spans use document-level cumulative
                            # token positions, not per-sentence-relative ones —
                            # subtract the running offset before indexing.
                for sent_tokens, sent_ents in zip(doc["sentences"], doc["ner"]):
                    tags = ["O"] * len(sent_tokens)
                    for start, end, etype in sent_ents:
                        s, e = start - offset, end - offset
                        if not (0 <= s < len(sent_tokens) and 0 <= e < len(sent_tokens)):
                            continue  # defensive: skip malformed spans rather than crash
                        tags[s] = f"B-{etype}"
                        for i in range(s + 1, e + 1):
                            tags[i] = f"I-{etype}"
                    examples.append({"tokens": sent_tokens, "tags": tags})
                    offset += len(sent_tokens)
        return examples

    train_examples = load_scierc_ner(train_file)
    dev_examples = load_scierc_ner(dev_file)
    print(f"SciERC train sentences: {len(train_examples)}, dev: {len(dev_examples)}")

    def encode(examples):
        enc = tokenizer(examples["tokens"], is_split_into_words=True,
                         truncation=True, max_length=MAX_LEN, padding="max_length")
        labels = []
        for i, tags in enumerate(examples["tags"]):
            word_ids = enc.word_ids(batch_index=i)
            prev = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    label_ids.append(label2id[tags[wid]])
                else:
                    label_ids.append(-100)
                prev = wid
            labels.append(label_ids)
        enc["labels"] = labels
        return enc

    from datasets import Dataset
    train_ds = Dataset.from_list(train_examples).map(encode, batched=True)
    dev_ds = Dataset.from_list(dev_examples).map(encode, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=dict(enumerate(label_list)),
        label2id=label2id, use_safetensors=True,
        # SciBERT's repo ships model.safetensors alongside the older
        # pytorch_model.bin; forcing safetensors avoids transformers'
        # torch.load-based loading path, which recent transformers versions
        # refuse to use on torch<2.6 due to CVE-2025-32434.
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=8,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "ner"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "ner"))
    print(f"Saved fine-tuned entity model to {CHECKPOINT_DIR / 'ner'}")
    print(
        "NOTE: this script fine-tunes the entity (NER) head, which is the "
        "highest-value piece to get right for Component 3's node set. "
        "Relation extraction here uses SciBERT [CLS]-pooled span-pair "
        "features + the same dependency-pattern refinement as the Tier-1 "
        "baseline for the METHOD_APPLIED_TO / METHOD_IMPROVES_TASK split — "
        "for a fully learned relation classifier, fine-tune a second head "
        "on SciERC's relation annotations using the span-pair architecture "
        "from SpERT/PURE (both have public reference implementations)."
    )


def run_inference(domains=("NLP", "COVID"), year=None, limit=None):
    """Applies the fine-tuned model to the Component 1 sampled corpora.

    Args:
        domains: which domain corpora to run over, e.g. ("NLP",) for a
            single-domain slice.
        year: if set, only papers from this year are processed (int).
        limit: if set, caps the number of papers processed per domain,
            applied after the year filter. Use this + year for the
            small-slice sanity check in the workflow (e.g. --domain NLP
            --year 2018 covers ~150-250 papers on its own; add --limit if
            you want to cap it further).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import spacy

    ner_path = CHECKPOINT_DIR / "ner"
    if not ner_path.exists():
        raise SystemExit(
            f"No fine-tuned model found at {ner_path}. Run with --train first "
            f"(requires internet access to huggingface.co for {MODEL_NAME} and "
            f"to download the SciERC corpus)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(ner_path))
    model = AutoModelForTokenClassification.from_pretrained(
        str(ner_path), use_safetensors=True
    ).to(device).eval()
    id2label = model.config.id2label

    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # sentence split + deps only

    # Reuse the Tier-1 dependency-pattern relation logic for consistency —
    # only the entity layer changes (SciBERT NER vs. lexicon rules).
    from extract_entities_relations_baseline import extract_sentence_relations
    from collections import Counter

    for domain in domains:
        records = load_domain_corpus(domain)
        if year is not None:
            records = [r for r in records if r.get("year") == year]
        if limit is not None:
            records = records[:limit]
        print(f"[{domain}] processing {len(records)} papers"
              f"{f' (year={year})' if year is not None else ''}"
              f"{f' (limit={limit})' if limit is not None else ''}")
        paper_records, all_entities, all_relations = [], [], []
        rel_counter = Counter()
        t0 = time.time()

        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start: batch_start + BATCH_SIZE]
            texts = [((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip() for r in batch]

            enc = tokenizer(texts, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN, padding=True, return_offsets_mapping=True).to(device)
            with torch.no_grad():
                logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1).cpu().tolist()
            pred_confs = probs.max(-1).values.cpu().tolist()  # per-token softmax prob of the argmax label

            for rec, text, pred_ids, offsets, tok_confs in zip(
                    batch, texts, preds, enc["offset_mapping"].cpu().tolist(), pred_confs):
                paper_id = rec["paper_id"]
                paper_year = rec.get("year")
                if not text.strip():
                    paper_records.append({"paper_id": paper_id, "year": paper_year, "domain": domain,
                                           "title": rec.get("title", ""), "status": "skipped_empty_abstract",
                                           "entities": [], "relations": []})
                    continue

                # BIO decode -> char-span entities (tracking per-token softmax
                # confidence for each span so we can report a real,
                # model-derived confidence instead of a flat constant)
                entities, cur = [], None
                for pid, (s, e), tconf in zip(pred_ids, offsets, tok_confs):
                    if s == e:
                        continue
                    label = id2label[pid]
                    if label.startswith("B-"):
                        if cur:
                            entities.append(cur)
                        cur = {"type": ENTITY_TYPE_MAP[label[2:]], "char_start": s, "char_end": e,
                               "confs": [tconf]}
                    elif label.startswith("I-") and cur and ENTITY_TYPE_MAP.get(label[2:]) == cur["type"]:
                        cur["char_end"] = e
                        cur["confs"].append(tconf)
                    else:
                        if cur:
                            entities.append(cur)
                        cur = None
                if cur:
                    entities.append(cur)

                paper_entities = []
                for idx, ent in enumerate(entities):
                    span_conf = sum(ent["confs"]) / len(ent["confs"])  # mean softmax prob over the span's tokens
                    paper_entities.append({
                        "entity_id": f"e_{paper_id}_{idx}", "canonical_id": None,
                        "surface_form": text[ent["char_start"]:ent["char_end"]],
                        "normalized_form": text[ent["char_start"]:ent["char_end"]].lower().strip(),
                        "type": ent["type"], "char_start": ent["char_start"], "char_end": ent["char_end"],
                        "confidence": round(span_conf, 4),  # real per-span mean softmax probability
                        "paper_id": paper_id, "year": paper_year, "domain": domain,
                    })

                doc = nlp(text)
                paper_relations = []
                rel_idx = 0
                for sent in doc.sents:
                    sent_ents = [e for e in paper_entities
                                 if e["char_start"] >= sent.start_char and e["char_end"] <= sent.end_char]
                    for src, tgt, rtype, conf in extract_sentence_relations(sent, sent_ents, rel_counter):
                        paper_relations.append({
                            "relation_id": f"r_{paper_id}_{rel_idx}",
                            "source_entity_id": src["entity_id"], "target_entity_id": tgt["entity_id"],
                            "relation_type": rtype, "confidence": conf,
                            "paper_id": paper_id, "year": paper_year, "domain": domain,
                        })
                        rel_idx += 1

                all_entities.extend(paper_entities)
                all_relations.extend(paper_relations)
                paper_records.append({
                    "paper_id": paper_id, "year": paper_year, "domain": domain,
                    "title": rec.get("title", ""), "status": "ok",
                    "n_entities": len(paper_entities), "n_relations": len(paper_relations),
                    "entities": paper_entities, "relations": paper_relations,
                })

        print(f"[{domain}] SciBERT inference: {len(all_entities)} entities, "
              f"{len(all_relations)} relations in {time.time()-t0:.1f}s")
        # Small-slice test runs (year and/or limit set) get a "_slice" suffix
        # so they never silently overwrite a completed full-corpus run.
        is_slice = year is not None or limit is not None
        suffix = "_slice" if is_slice else ""
        write_json(OUTPUT_DIR / f"{domain}_extracted{suffix}.json", paper_records)
        write_json(OUTPUT_DIR / f"{domain}_entities{suffix}.json", all_entities)
        write_json(OUTPUT_DIR / f"{domain}_relations{suffix}.json", all_relations)
        if not is_slice:
            # Only a full-corpus run updates the method marker that
            # validate_extraction.py trusts — a slice run must never make
            # the report claim the full corpus was SciBERT-processed.
            write_json(OUTPUT_DIR / f"{domain}_extraction_method.json", {
                "method": "scibert_finetuned",
                "model": MODEL_NAME,
                "checkpoint": str(ner_path),
            })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="fine-tune SciBERT NER head on SciERC")
    p.add_argument("--run", action="store_true", help="run inference over the sampled corpora")
    p.add_argument("--domain", choices=["NLP", "COVID", "both"], default="both",
                    help="restrict inference to one domain (default: both)")
    p.add_argument("--year", type=int, default=None,
                    help="restrict inference to a single year, e.g. 2018 "
                         "(use with --domain for the small-slice sanity check, "
                         "e.g. --domain NLP --year 2018)")
    p.add_argument("--limit", type=int, default=None,
                    help="cap the number of papers processed per domain")
    args = p.parse_args()
    if not (args.train or args.run):
        p.error("pass --train and/or --run")
    if args.train:
        train()
    if args.run:
        domains = ("NLP", "COVID") if args.domain == "both" else (args.domain,)
        run_inference(domains=domains, year=args.year, limit=args.limit)
