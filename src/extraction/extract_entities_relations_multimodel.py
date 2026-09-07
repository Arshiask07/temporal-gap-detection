"""
Component 2 — Tier 3: multi-model comparative extraction.

Fine-tunes and runs THREE alternative transformer backbones on the exact
same SciERC entity+relation task that extract_entities_relations_scibert.py
/ extract_relations_scibert.py already run for SciBERT alone, so the three
can be compared head-to-head for the comparative-study section of the
report:

    scibert      allenai/scibert_scivocab_uncased
                 Scientific-domain, uncased. (Already your Tier-2 model —
                 included here again so all three share one run/report path.)
    roberta      roberta-base
                 General-domain (web text + books), cased. No scientific
                 pretraining at all — this is the "generic strong baseline"
                 arm of the comparison.
    pubmedbert   microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
                 Biomedical-domain, uncased, pretrained from scratch on
                 PubMed abstracts + PMC full text. Expected to do best on
                 the COVID domain, worse than SciBERT on NLP-domain text.
                 (Upstream renamed this to BiomedNLP-BiomedBERT-*; the old
                 PubMedBERT name still resolves on transformers>=4.22.)

WHY ONE FILE: rather than duplicating the ~800 lines of training/inference
logic already written and tested in extract_entities_relations_scibert.py
and extract_relations_scibert.py three times over, this script imports
their model-agnostic pieces (SciERC loading, the relation span-pair model
class, the entity-type/relation-type schema mapping, the SciERC download
helper) and parameterizes the one thing that actually changes per model:
the HuggingFace checkpoint name and where its checkpoints/outputs go. Every
step below takes a `model_key` argument instead of hard-coding SciBERT.

OUTPUTS ARE KEPT FULLY SEPARATE PER MODEL, under three subfolders of the
existing Component 2 output/ directory, so nothing overwrites SciBERT's
existing output/{NLP,COVID}_*.json files and the three model runs never
collide:

    component2_entity_relation_extraction/
      output/
        scibert/{NLP,COVID}_extracted.json, _entities.json, _relations.json, ...
        roberta/{NLP,COVID}_extracted.json, _entities.json, _relations.json, ...
        pubmedbert/{NLP,COVID}_extracted.json, _entities.json, _relations.json, ...
      scripts/checkpoints/
        scibert_scierc/{ner, relation}
        roberta_scierc/{ner, relation}
        pubmedbert_scierc/{ner, relation}
        scierc_data/          <- downloaded ONCE, shared by all 3 models

Run this on a machine with real internet access (huggingface.co reachable) —
same caveat as extract_entities_relations_scibert.py.

Requirements:
    pip install torch transformers datasets seqeval rapidfuzz scikit-learn spacy requests
    python3 -m spacy download en_core_web_sm

Usage — one model at a time:
    python3 extract_entities_relations_multimodel.py --model roberta --step full
    python3 extract_entities_relations_multimodel.py --model pubmedbert --step full

Usage — all three models, one command, sequentially:
    python3 extract_entities_relations_multimodel.py --model all --step full

Usage — individual steps (e.g. to re-run just relation training for one model):
    python3 extract_entities_relations_multimodel.py --model roberta --step train-ner
    python3 extract_entities_relations_multimodel.py --model roberta --step run-ner --domain NLP --year 2018   # slice
    python3 extract_entities_relations_multimodel.py --model roberta --step run-ner                            # full corpus
    python3 extract_entities_relations_multimodel.py --model roberta --step train-relation
    python3 extract_entities_relations_multimodel.py --model roberta --step run-relation
    python3 extract_entities_relations_multimodel.py --model roberta --step f1

Usage — comparison report across whichever models have been run so far:
    python3 extract_entities_relations_multimodel.py --step compare

Output/relation schema is IDENTICAL across all three models (and identical
to the existing SciBERT-only scripts), so Component 3 and validate_extraction
-style reporting need zero changes to consume any of them.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import load_domain_corpus, write_json, read_json, OUTPUT_DIR, REPORT_DIR

# Reused, model-agnostic pieces from the existing SciBERT-only scripts —
# see module docstring for why these are imported rather than copy-pasted.
from extract_entities_relations_scibert import (
    SCIERC_ENTITY_TYPES, SCIERC_RELATION_TYPES, RELATION_MAP, ENTITY_TYPE_MAP,
    download_scierc,
)
from extract_entities_relations_baseline import extract_sentence_relations
from eval_scierc_f1 import load_scierc_ner
from extract_relations_scibert import (
    NO_RELATION, REL_LABELS, load_scierc_relations, build_model_class,
    MAX_PAIRS_PER_SENT_TRAIN_NEG, MAX_ENTITIES_PER_SENTENCE, PAIR_BATCH_SIZE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = SCRIPT_DIR / "checkpoints"
SCIERC_DATA_DIR = CHECKPOINTS_ROOT / "scierc_data"   # shared download, once

MODEL_REGISTRY = {
    "scibert": {
        "hf_name": "allenai/scibert_scivocab_uncased",
        "display_name": "SciBERT",
    },
    "roberta": {
        "hf_name": "roberta-base",
        "display_name": "RoBERTa",
    },
    "pubmedbert": {
        "hf_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "display_name": "PubMedBERT",
    },
}

# Kept identical across all three models on purpose — this is a comparative
# study of the BACKBONE, not of hyperparameters, so those are held fixed.
# BATCH_SIZE is deliberately small (fits ~4-6GB VRAM); GRAD_ACCUM_STEPS
# restores an effective batch size of BATCH_SIZE * GRAD_ACCUM_STEPS for
# training stability without needing more memory. Override with --batch-size
# / --grad-accum-steps if you still hit CUDA OOM, or run on CPU (slower but
# unlimited "VRAM").
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_LEN = 256
NER_EPOCHS = 8
NER_LR = 3e-5
REL_EPOCHS = 8
REL_LR = 2e-5


def checkpoint_dir(model_key: str) -> Path:
    return CHECKPOINTS_ROOT / f"{model_key}_scierc"


def rel_checkpoint_dir(model_key: str) -> Path:
    return checkpoint_dir(model_key) / "relation"


def model_output_dir(model_key: str) -> Path:
    d = OUTPUT_DIR / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Entity (NER) fine-tuning — same recipe as extract_entities_relations_
# scibert.py's train(), parameterized by model_key.
# --------------------------------------------------------------------------- #
def train_ner(model_key: str, batch_size: int = BATCH_SIZE, grad_accum: int = GRAD_ACCUM_STEPS):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
    )
    from datasets import Dataset

    hf_name = MODEL_REGISTRY[model_key]["hf_name"]
    ckpt_dir = checkpoint_dir(model_key)
    print(f"\n=== [{model_key}] fine-tuning {hf_name} (NER) on SciERC ===")

    scierc_dir = download_scierc(SCIERC_DATA_DIR)
    train_file = next(scierc_dir.rglob("train.json"))
    dev_file = next(scierc_dir.rglob("dev.json"))

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    label_list = ["O"] + [f"{p}-{t}" for t in SCIERC_ENTITY_TYPES for p in ("B", "I")]
    label2id = {l: i for i, l in enumerate(label_list)}

    train_examples = load_scierc_ner(train_file)
    dev_examples = load_scierc_ner(dev_file)
    print(f"[{model_key}] SciERC train sentences: {len(train_examples)}, dev: {len(dev_examples)}")

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

    train_ds = Dataset.from_list(train_examples).map(encode, batched=True)
    dev_ds = Dataset.from_list(dev_examples).map(encode, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        hf_name, num_labels=len(label_list), id2label=dict(enumerate(label_list)),
        label2id=label2id, use_safetensors=True,
    )

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=NER_EPOCHS,
        learning_rate=NER_LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(ckpt_dir / "ner"))
    tokenizer.save_pretrained(str(ckpt_dir / "ner"))
    print(f"[{model_key}] saved fine-tuned entity model to {ckpt_dir / 'ner'}")


# --------------------------------------------------------------------------- #
# Entity + Tier-1-heuristic-relation inference — same recipe as
# extract_entities_relations_scibert.py's run_inference(), parameterized.
# The heuristic relations produced here are a placeholder: run-relation
# (below) replaces them with the trained span-pair classifier.
# --------------------------------------------------------------------------- #
def run_ner_inference(model_key: str, domains=("NLP", "COVID"), year=None, limit=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import spacy

    ckpt_dir = checkpoint_dir(model_key)
    ner_path = ckpt_dir / "ner"
    if not ner_path.exists():
        raise SystemExit(
            f"[{model_key}] No fine-tuned model found at {ner_path}. "
            f"Run --step train-ner first."
        )
    out_dir = model_output_dir(model_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(ner_path))
    model = AutoModelForTokenClassification.from_pretrained(
        str(ner_path), use_safetensors=True
    ).to(device).eval()
    id2label = model.config.id2label

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    for domain in domains:
        records = load_domain_corpus(domain)
        if year is not None:
            records = [r for r in records if r.get("year") == year]
        if limit is not None:
            records = records[:limit]
        print(f"[{model_key}][{domain}] processing {len(records)} papers"
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
            pred_confs = probs.max(-1).values.cpu().tolist()

            for rec, text, pred_ids, offsets, tok_confs in zip(
                    batch, texts, preds, enc["offset_mapping"].cpu().tolist(), pred_confs):
                paper_id = rec["paper_id"]
                paper_year = rec.get("year")
                if not text.strip():
                    paper_records.append({"paper_id": paper_id, "year": paper_year, "domain": domain,
                                           "title": rec.get("title", ""), "status": "skipped_empty_abstract",
                                           "entities": [], "relations": []})
                    continue

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
                    span_conf = sum(ent["confs"]) / len(ent["confs"])
                    paper_entities.append({
                        "entity_id": f"e_{paper_id}_{idx}", "canonical_id": None,
                        "surface_form": text[ent["char_start"]:ent["char_end"]],
                        "normalized_form": text[ent["char_start"]:ent["char_end"]].lower().strip(),
                        "type": ent["type"], "char_start": ent["char_start"], "char_end": ent["char_end"],
                        "confidence": round(span_conf, 4),
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

        print(f"[{model_key}][{domain}] NER inference: {len(all_entities)} entities, "
              f"{len(all_relations)} heuristic relations in {time.time()-t0:.1f}s")
        is_slice = year is not None or limit is not None
        suffix = "_slice" if is_slice else ""
        write_json(out_dir / f"{domain}_extracted{suffix}.json", paper_records)
        write_json(out_dir / f"{domain}_entities{suffix}.json", all_entities)
        write_json(out_dir / f"{domain}_relations{suffix}.json", all_relations)
        if not is_slice:
            write_json(out_dir / f"{domain}_extraction_method.json", {
                "method": "scibert_finetuned" if model_key == "scibert" else f"{model_key}_finetuned",
                "model": MODEL_REGISTRY[model_key]["hf_name"],
                "checkpoint": str(ner_path),
            })


# --------------------------------------------------------------------------- #
# Relation classifier fine-tuning — same recipe as extract_relations_
# scibert.py's train(), parameterized by model_key.
# --------------------------------------------------------------------------- #
def train_relation(model_key: str, batch_size: int = BATCH_SIZE, grad_accum: int = GRAD_ACCUM_STEPS):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score as sk_f1

    hf_name = MODEL_REGISTRY[model_key]["hf_name"]
    rel_ckpt_dir = rel_checkpoint_dir(model_key)
    print(f"\n=== [{model_key}] fine-tuning {hf_name} (relation classifier) on SciERC ===")

    scierc_dir = download_scierc(SCIERC_DATA_DIR)
    train_file = next(scierc_dir.rglob("train.json"))
    dev_file = next(scierc_dir.rglob("dev.json"))

    train_examples = load_scierc_relations(train_file)
    dev_examples = load_scierc_relations(dev_file)
    print(f"[{model_key}] SciERC relation train sentences (with >=1 pair): "
          f"{len(train_examples)}, dev: {len(dev_examples)}")

    label2id = {l: i for i, l in enumerate(REL_LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    class RelDataset(Dataset):
        def __init__(self, examples):
            self.rows = []
            for ex in examples:
                for i, j, label in ex["pairs"]:
                    self.rows.append((ex["tokens"], ex["entities"][i], ex["entities"][j], label))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    def collate(batch):
        tokens_batch = [b[0] for b in batch]
        enc = tokenizer(tokens_batch, is_split_into_words=True, truncation=True,
                         max_length=MAX_LEN, padding=True, return_tensors="pt")
        word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch))]
        e1_start = torch.tensor([b[1][0] for b in batch])
        e1_end = torch.tensor([b[1][1] for b in batch])
        e2_start = torch.tensor([b[2][0] for b in batch])
        e2_end = torch.tensor([b[2][1] for b in batch])
        labels = torch.tensor([label2id[b[3]] for b in batch])
        return enc, e1_start, e1_end, e2_start, e2_end, word_ids_batch, labels

    train_ds, dev_ds = RelDataset(train_examples), RelDataset(dev_examples)
    print(f"[{model_key}] Total pairs — train: {len(train_ds)}, dev: {len(dev_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] Using device: {device}  (batch_size={batch_size}, "
          f"grad_accum={grad_accum}, effective_batch={batch_size*grad_accum})")
    ModelClass = build_model_class()
    model = ModelClass(hf_name, len(REL_LABELS)).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    optim = torch.optim.AdamW(model.parameters(), lr=REL_LR)
    total_steps = (len(train_loader) // grad_accum + 1) * REL_EPOCHS
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.1 * total_steps),
                                             num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for epoch in range(1, REL_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optim.zero_grad()
        for step, (enc, e1s, e1e, e2s, e2e, wids, labels) in enumerate(train_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(enc["input_ids"], enc["attention_mask"],
                                e1s.to(device), e1e.to(device), e2s.to(device), e2e.to(device), wids)
                loss = loss_fn(logits, labels.to(device)) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad()
            total_loss += loss.item() * grad_accum

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for enc, e1s, e1e, e2s, e2e, wids, labels in dev_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(enc["input_ids"], enc["attention_mask"],
                                e1s.to(device), e1e.to(device), e2s.to(device), e2e.to(device), wids)
                y_pred.extend(logits.argmax(-1).cpu().tolist())
                y_true.extend(labels.tolist())
        rel_label_ids = [label2id[l] for l in SCIERC_RELATION_TYPES]
        dev_f1 = sk_f1(y_true, y_pred, labels=rel_label_ids, average="micro", zero_division=0)
        print(f"[{model_key}] Epoch {epoch}/{REL_EPOCHS}  train_loss={total_loss/len(train_loader):.4f}  "
              f"dev_relation_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{model_key}]   ↳ new best (dev_relation_f1={best_f1:.4f}) — saved in memory")

    rel_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), rel_ckpt_dir / "model.pt")
    tokenizer.save_pretrained(str(rel_ckpt_dir))
    (rel_ckpt_dir / "labels.json").write_text(json.dumps(REL_LABELS, indent=2))
    print(f"[{model_key}] Saved best relation model (dev_relation_f1={best_f1:.4f}) to {rel_ckpt_dir}")


# --------------------------------------------------------------------------- #
# Trained relation classifier inference — same recipe as extract_relations_
# scibert.py's run_inference(), parameterized by model_key.
# --------------------------------------------------------------------------- #
def run_relation_inference(model_key: str, domains=("NLP", "COVID")):
    import torch
    from transformers import AutoTokenizer
    import spacy
    from itertools import permutations

    hf_name = MODEL_REGISTRY[model_key]["hf_name"]
    rel_ckpt_dir = rel_checkpoint_dir(model_key)
    out_dir = model_output_dir(model_key)

    if not (rel_ckpt_dir / "model.pt").exists():
        raise SystemExit(f"[{model_key}] No trained relation model found at {rel_ckpt_dir}. "
                          f"Run --step train-relation first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(rel_ckpt_dir))
    labels = json.loads((rel_ckpt_dir / "labels.json").read_text())
    ModelClass = build_model_class()
    model = ModelClass(hf_name, len(labels)).to(device)
    model.load_state_dict(torch.load(rel_ckpt_dir / "model.pt", map_location=device))
    model.eval()

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    for domain in domains:
        entities_path = out_dir / f"{domain}_entities.json"
        if not entities_path.exists():
            raise SystemExit(f"[{model_key}] No entities found at {entities_path}. "
                              f"Run --step run-ner (full corpus, no --year/--limit) first.")
        entities_by_paper = {}
        for e in read_json(entities_path):
            entities_by_paper.setdefault(e["paper_id"], []).append(e)

        records = {r["paper_id"]: r for r in load_domain_corpus(domain)}
        all_relations = []
        t0 = time.time()
        n_papers = 0

        for paper_id, ents in entities_by_paper.items():
            rec = records.get(paper_id)
            if not rec or len(ents) < 2:
                continue
            text = ((rec.get("title") or "") + ". " + (rec.get("abstract") or "")).strip()
            if not text:
                continue
            doc = nlp(text)
            n_papers += 1
            rel_idx = 0
            if n_papers % 50 == 0 or n_papers == 1:
                elapsed = time.time() - t0
                print(f"[{model_key}][{domain}] {n_papers}/{len(entities_by_paper)} papers "
                      f"({len(all_relations)} relations so far, {elapsed:.0f}s elapsed)", flush=True)

            for sent in doc.sents:
                sent_ents = [e for e in ents if e["char_start"] >= sent.start_char and e["char_end"] <= sent.end_char]
                if len(sent_ents) < 2:
                    continue
                if len(sent_ents) > MAX_ENTITIES_PER_SENTENCE:
                    sent_ents = sorted(sent_ents, key=lambda e: -e["confidence"])[:MAX_ENTITIES_PER_SENTENCE]
                sent_tokens = [tok.text for tok in sent]

                def char_to_word(char_pos):
                    for wi, tok in enumerate(sent):
                        if tok.idx <= char_pos < tok.idx + len(tok.text):
                            return wi
                    return 0

                word_spans = [(char_to_word(e["char_start"]), char_to_word(max(e["char_end"] - 1, e["char_start"])))
                              for e in sent_ents]

                all_pairs = list(permutations(range(len(sent_ents)), 2))
                if not all_pairs:
                    continue

                for batch_start in range(0, len(all_pairs), PAIR_BATCH_SIZE):
                    pairs = all_pairs[batch_start: batch_start + PAIR_BATCH_SIZE]
                    enc = tokenizer([sent_tokens] * len(pairs), is_split_into_words=True, truncation=True,
                                     max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
                    word_ids_batch = [enc.word_ids(batch_index=k) for k in range(len(pairs))]
                    e1s = torch.tensor([word_spans[i][0] for i, j in pairs]).to(device)
                    e1e = torch.tensor([word_spans[i][1] for i, j in pairs]).to(device)
                    e2s = torch.tensor([word_spans[j][0] for i, j in pairs]).to(device)
                    e2e = torch.tensor([word_spans[j][1] for i, j in pairs]).to(device)

                    with torch.no_grad():
                        logits = model(enc["input_ids"], enc["attention_mask"], e1s, e1e, e2s, e2e, word_ids_batch)
                        probs = torch.softmax(logits, dim=-1)
                        pred_ids = probs.argmax(-1).cpu().tolist()
                        confs = probs.max(-1).values.cpu().tolist()

                    for (i, j), pid, conf in zip(pairs, pred_ids, confs):
                        scierc_label = labels[pid]
                        if scierc_label == NO_RELATION:
                            continue
                        src, tgt = sent_ents[i], sent_ents[j]
                        mapped = RELATION_MAP.get(scierc_label, "ENTITY_ASSOCIATED_WITH_ENTITY")
                        if scierc_label == "Used-for" and src["type"] == "Method" and tgt["type"] in ("Task", "Material"):
                            mapped = "METHOD_APPLIED_TO"
                        all_relations.append({
                            "relation_id": f"r_{paper_id}_{rel_idx}",
                            "source_entity_id": src["entity_id"], "target_entity_id": tgt["entity_id"],
                            "relation_type": mapped, "scierc_relation_type": scierc_label,
                            "confidence": round(conf, 4),
                            "paper_id": paper_id, "year": rec.get("year"), "domain": domain,
                        })
                        rel_idx += 1

        print(f"[{model_key}][{domain}] trained relation model: {len(all_relations)} relations "
              f"across {n_papers} papers in {time.time()-t0:.1f}s")
        write_json(out_dir / f"{domain}_relations.json", all_relations)

        extracted = read_json(out_dir / f"{domain}_extracted.json")
        rels_by_paper = {}
        for r in all_relations:
            rels_by_paper.setdefault(r["paper_id"], []).append(r)
        for p in extracted:
            p["relations"] = rels_by_paper.get(p["paper_id"], [])
            p["n_relations"] = len(p["relations"])
        write_json(out_dir / f"{domain}_extracted.json", extracted)

        write_json(out_dir / f"{domain}_relation_method.json", {
            "method": f"{model_key}_relation_finetuned",
            "model": hf_name,
            "checkpoint": str(rel_ckpt_dir),
        })


# --------------------------------------------------------------------------- #
# SciERC held-out TEST-set entity F1 — same recipe as eval_scierc_f1.py,
# parameterized by model_key. Written into that model's checkpoint dir.
# --------------------------------------------------------------------------- #
def evaluate_f1(model_key: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    ckpt_dir = checkpoint_dir(model_key)
    ner_path = ckpt_dir / "ner"
    if not ner_path.exists():
        raise SystemExit(f"[{model_key}] No fine-tuned model found at {ner_path}. Run --step train-ner first.")

    scierc_dir = download_scierc(SCIERC_DATA_DIR)
    test_file = next(scierc_dir.rglob("test.json"))
    test_examples = load_scierc_ner(test_file)
    print(f"[{model_key}] SciERC test sentences: {len(test_examples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(ner_path))
    model = AutoModelForTokenClassification.from_pretrained(
        str(ner_path), use_safetensors=True
    ).to(device).eval()
    id2label = model.config.id2label

    y_true, y_pred = [], []
    BATCH = 16
    for start in range(0, len(test_examples), BATCH):
        batch = test_examples[start:start + BATCH]
        tokens_batch = [ex["tokens"] for ex in batch]
        enc = tokenizer(tokens_batch, is_split_into_words=True, truncation=True,
                         max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = logits.argmax(-1).cpu().tolist()

        for i, ex in enumerate(batch):
            word_ids = enc.word_ids(batch_index=i)
            seen = set()
            pred_tags = []
            for wid, pid in zip(word_ids, pred_ids[i]):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                pred_tags.append(id2label[pid])
            while len(pred_tags) < len(ex["tags"]):
                pred_tags.append("O")
            pred_tags = pred_tags[:len(ex["tags"])]
            y_true.append(ex["tags"])
            y_pred.append(pred_tags)

    print(f"\n=== [{model_key}] SciERC held-out TEST set — entity NER (seqeval, span-level) ===\n")
    print(classification_report(y_true, y_pred, digits=4))
    overall = {
        "model_key": model_key, "model": MODEL_REGISTRY[model_key]["hf_name"],
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "n_test_sentences": len(test_examples),
        "entity_types": SCIERC_ENTITY_TYPES,
    }
    out_path = ckpt_dir / "scierc_test_f1.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"[{model_key}] Overall: P={overall['precision']:.4f} R={overall['recall']:.4f} F1={overall['f1']:.4f}")
    print(f"[{model_key}] Saved to {out_path}")


# --------------------------------------------------------------------------- #
# Comparison report — reads whichever models have output so far and builds
# one side-by-side markdown table. Safe to run at any point; models with no
# output yet are just skipped with a note.
# --------------------------------------------------------------------------- #
def compute_stats(extracted, entities, relations):
    entity_type_counts = Counter(e["type"] for e in entities)
    relation_type_counts = Counter(r["relation_type"] for r in relations)
    total_r = len(relations) or 1
    typed = sum(v for k, v in relation_type_counts.items() if k != "ENTITY_ASSOCIATED_WITH_ENTITY")
    return {
        "papers": len(extracted),
        "total_entities": len(entities),
        "total_relations": len(relations),
        "entity_type_counts": dict(entity_type_counts),
        "relation_type_counts": dict(relation_type_counts),
        "typed_relation_share": typed / total_r,
        "other_entity_share": entity_type_counts.get("Other", 0) / max(len(entities), 1),
    }


def build_comparison_report():
    print("\n=== Building cross-model comparison report ===")
    sections = ["# COMPONENT 2 — CROSS-MODEL COMPARISON (SciBERT vs RoBERTa vs PubMedBERT)\n"]
    any_model_found = False

    for model_key, info in MODEL_REGISTRY.items():
        out_dir = OUTPUT_DIR / model_key
        ckpt_dir = checkpoint_dir(model_key)
        f1_path = ckpt_dir / "scierc_test_f1.json"

        sections.append(f"\n## {info['display_name']}  (`{info['hf_name']}`)\n")

        if f1_path.exists():
            f1_stats = json.loads(f1_path.read_text())
            sections.append(f"- SciERC held-out TEST entity F1: **{f1_stats['f1']:.4f}** "
                             f"(P={f1_stats['precision']:.4f}, R={f1_stats['recall']:.4f}, "
                             f"n={f1_stats['n_test_sentences']} sentences)")
        else:
            sections.append("- SciERC held-out TEST entity F1: *not yet evaluated — run --step f1*")

        found_any_domain = False
        for domain in ("NLP", "COVID"):
            extracted_path = out_dir / f"{domain}_extracted.json"
            entities_path = out_dir / f"{domain}_entities.json"
            relations_path = out_dir / f"{domain}_relations.json"
            if not (extracted_path.exists() and entities_path.exists() and relations_path.exists()):
                continue
            found_any_domain = True
            any_model_found = True
            extracted = read_json(extracted_path)
            entities = read_json(entities_path)
            relations = read_json(relations_path)
            st = compute_stats(extracted, entities, relations)
            sections.append(f"\n**{domain} domain:**")
            sections.append(f"- Papers: {st['papers']}  |  Entities: {st['total_entities']}  |  "
                             f"Relations: {st['total_relations']}")
            sections.append(f"- Typed-relation share (non-generic): **{st['typed_relation_share']:.1%}**")
            sections.append(f"- 'Other' entity-type share: **{st['other_entity_share']:.1%}**")
            top_entity = sorted(st["entity_type_counts"].items(), key=lambda x: -x[1])
            top_relation = sorted(st["relation_type_counts"].items(), key=lambda x: -x[1])
            sections.append(f"- Entity types: {', '.join(f'{k}={v}' for k, v in top_entity)}")
            sections.append(f"- Relation types: {', '.join(f'{k}={v}' for k, v in top_relation)}")

        if not found_any_domain:
            sections.append("\n*No full-corpus extraction output yet for this model — "
                             "run --step run-ner then --step run-relation.*")

    if not any_model_found:
        sections.append("\n*No model has produced full-corpus output yet.*")

    report_path = REPORT_DIR / "COMPONENT2_MODEL_COMPARISON.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Comparison report written to {report_path} ({report_path.stat().st_size} bytes)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
STEP_ORDER = ["train-ner", "run-ner", "train-relation", "run-relation", "f1"]


def run_full(model_key: str, batch_size: int = BATCH_SIZE, grad_accum: int = GRAD_ACCUM_STEPS):
    train_ner(model_key, batch_size=batch_size, grad_accum=grad_accum)
    run_ner_inference(model_key, domains=("NLP", "COVID"))
    train_relation(model_key, batch_size=batch_size, grad_accum=grad_accum)
    run_relation_inference(model_key, domains=("NLP", "COVID"))
    evaluate_f1(model_key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all",
                    help="which backbone to run (default: all three, sequentially)")
    p.add_argument("--step", choices=STEP_ORDER + ["full", "compare"], default="full",
                    help="which pipeline step to run (default: full = train-ner -> run-ner "
                         "-> train-relation -> run-relation -> f1)")
    p.add_argument("--domain", choices=["NLP", "COVID", "both"], default="both",
                    help="restrict run-ner to one domain (default: both)")
    p.add_argument("--year", type=int, default=None,
                    help="restrict run-ner to a single year, e.g. 2018 — use with --domain "
                         "for a small-slice sanity check before the full run")
    p.add_argument("--limit", type=int, default=None,
                    help="cap papers processed per domain during run-ner")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"per-device training batch size for train-ner/train-relation "
                         f"(default: {BATCH_SIZE}; lower this first if you hit CUDA OOM)")
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS,
                    help=f"gradient accumulation steps — effective batch size = "
                         f"batch-size * grad-accum-steps (default: {GRAD_ACCUM_STEPS})")
    args = p.parse_args()

    if args.step == "compare":
        build_comparison_report()
        return

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    domains = ("NLP", "COVID") if args.domain == "both" else (args.domain,)

    for model_key in models:
        if args.step == "full":
            run_full(model_key, batch_size=args.batch_size, grad_accum=args.grad_accum_steps)
        elif args.step == "train-ner":
            train_ner(model_key, batch_size=args.batch_size, grad_accum=args.grad_accum_steps)
        elif args.step == "run-ner":
            run_ner_inference(model_key, domains=domains, year=args.year, limit=args.limit)
        elif args.step == "train-relation":
            train_relation(model_key, batch_size=args.batch_size, grad_accum=args.grad_accum_steps)
        elif args.step == "run-relation":
            run_relation_inference(model_key, domains=domains)
        elif args.step == "f1":
            evaluate_f1(model_key)

    # After a full multi-model run, refresh the comparison report automatically.
    if args.step == "full" and args.model == "all":
        build_comparison_report()


if __name__ == "__main__":
    main()