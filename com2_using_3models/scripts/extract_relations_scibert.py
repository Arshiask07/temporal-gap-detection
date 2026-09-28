"""
Component 2 — Tier 2 relation extraction: fine-tune a real relation
classifier on SciERC's own relation annotations, replacing the Tier-1
dependency-pattern heuristic that both extract_entities_relations_baseline.py
and extract_entities_relations_scibert.py have been using so far.

WHY THIS SCRIPT EXISTS: extract_entities_relations_scibert.py only
fine-tuned SciBERT for entity recognition (NER). Relations were still being
produced by the same rule-based dependency-pattern layer from the Tier-1
baseline — real entities, but heuristic relations. That gap is exactly why
93%+ of relations in your report are the generic ENTITY_ASSOCIATED_WITH_ENTITY
fallback. This script closes it with a real, trained relation classifier.

Architecture (standard span-pair relation classification, the same family
used by SpERT/PURE): encode the sentence with SciBERT, mean-pool the token
embeddings inside each candidate entity's span to get two entity vectors,
concatenate [CLS] + entity1 + entity2 + (entity1 * entity2), feed through a
linear classifier over SciERC's 7 relation types plus a "no_relation" class
(needed because most entity PAIRS in a sentence are unrelated — without a
no_relation class the model would be forced to assign a real relation type
to every pair, badly hurting precision).

Usage:
    python3 extract_relations_scibert.py --train
    python3 extract_relations_scibert.py --run

Requires the entity outputs to already exist (run
extract_entities_relations_scibert.py --run first) — this script re-uses
those entity spans as the candidate pairs to classify, rather than
re-running NER itself.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from itertools import permutations

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import load_domain_corpus, write_json, read_json, OUTPUT_DIR
from extract_entities_relations_scibert import (
    MODEL_NAME, CHECKPOINT_DIR, MAX_LEN, download_scierc,
    SCIERC_RELATION_TYPES, RELATION_MAP,
)

REL_CHECKPOINT_DIR = CHECKPOINT_DIR / "relation"
NO_RELATION = "no_relation"
REL_LABELS = [NO_RELATION] + SCIERC_RELATION_TYPES
MAX_PAIRS_PER_SENT_TRAIN_NEG = 6  # cap negative pairs per sentence during training
MAX_ENTITIES_PER_SENTENCE = 12    # inference-time cap to avoid n*(n-1) blowup
PAIR_BATCH_SIZE = 32              # sub-batch pairs within a sentence during inference

# SciERC's raw processed JSON stores relation labels as "USED-FOR" (all caps,
# hyphenated) — different casing from SCIERC_RELATION_TYPES ("Used-for",
# shared with extract_entities_relations_scibert.py's RELATION_MAP). Normalize
# on read rather than changing the shared constant, since RELATION_MAP keys
# elsewhere depend on that exact casing.
_RAW_TO_CANONICAL = {t.upper().replace("_", "-"): t for t in SCIERC_RELATION_TYPES}


def normalize_relation_label(raw: str) -> str | None:
    return _RAW_TO_CANONICAL.get(raw.strip().upper().replace("_", "-"))


# --------------------------------------------------------------------------- #
# SciERC relation-label loading (document-level cumulative token offsets,
# same correction eval_scierc_f1.py already applies for NER spans)
# --------------------------------------------------------------------------- #
def load_scierc_relations(path):
    """Returns a list of {tokens, entities:[(s,e,type)], pairs:[(i,j,rel_label)]}
    one per sentence, with negative (no_relation) pairs included for every
    other entity-pair combination in that sentence."""
    examples = []
    total_unknown = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            offset = 0
            for sent_tokens, sent_ents, sent_rels in zip(doc["sentences"], doc["ner"], doc["relations"]):
                ents = []
                for s, e, etype in sent_ents:
                    s, e = s - offset, e - offset
                    if 0 <= s < len(sent_tokens) and 0 <= e < len(sent_tokens):
                        ents.append((s, e))
                span_to_idx = {span: i for i, span in enumerate(ents)}

                gold_pairs = {}  # (i,j) -> relation label
                unknown_labels = 0
                for s1, e1, s2, e2, rel in sent_rels:
                    s1, e1, s2, e2 = s1 - offset, e1 - offset, s2 - offset, e2 - offset
                    i, j = span_to_idx.get((s1, e1)), span_to_idx.get((s2, e2))
                    if i is not None and j is not None:
                        norm_rel = normalize_relation_label(rel)
                        if norm_rel is None:
                            unknown_labels += 1
                            continue
                        gold_pairs[(i, j)] = norm_rel

                if len(ents) >= 2:
                    all_pairs = list(permutations(range(len(ents)), 2))
                    pos = [(i, j, gold_pairs[(i, j)]) for (i, j) in all_pairs if (i, j) in gold_pairs]
                    neg_candidates = [(i, j) for (i, j) in all_pairs if (i, j) not in gold_pairs]
                    neg = neg_candidates[:max(len(pos) * 2, MAX_PAIRS_PER_SENT_TRAIN_NEG)]
                    pairs = pos + [(i, j, NO_RELATION) for (i, j) in neg]
                    if pairs:
                        examples.append({"tokens": sent_tokens, "entities": ents, "pairs": pairs})
                total_unknown += unknown_labels

                offset += len(sent_tokens)
    if total_unknown:
        print(f"  (skipped {total_unknown} gold relations with unrecognized label text — "
              f"check normalize_relation_label if this number looks large)")
    return examples


# --------------------------------------------------------------------------- #
# Model: SciBERT backbone + span-pair classification head
# --------------------------------------------------------------------------- #
def build_model_class():
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class SpanPairRelationModel(nn.Module):
        def __init__(self, model_name, num_labels):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
            hidden = self.bert.config.hidden_size
            self.classifier = nn.Linear(hidden * 4, num_labels)

        def span_pool(self, hidden_states, span_start, span_end, word_ids_batch):
            # mean-pool subword embeddings whose word_id falls within [start,end]
            pooled = []
            for b in range(hidden_states.size(0)):
                wids = word_ids_batch[b]
                mask = [(wid is not None and span_start[b] <= wid <= span_end[b]) for wid in wids]
                idx = [i for i, m in enumerate(mask) if m]
                if not idx:
                    pooled.append(hidden_states[b, 0])  # fallback to [CLS]
                else:
                    pooled.append(hidden_states[b, idx].mean(dim=0))
            return torch.stack(pooled)

        def forward(self, input_ids, attention_mask, e1_start, e1_end, e2_start, e2_end, word_ids_batch):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            cls = out[:, 0]
            v1 = self.span_pool(out, e1_start, e1_end, word_ids_batch)
            v2 = self.span_pool(out, e2_start, e2_end, word_ids_batch)
            feat = torch.cat([cls, v1, v2, v1 * v2], dim=-1)
            return self.classifier(feat)

    return SpanPairRelationModel


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score as sk_f1

    scierc_dir = download_scierc(CHECKPOINT_DIR.parent / "scierc_data")
    train_file = next(scierc_dir.rglob("train.json"))
    dev_file = next(scierc_dir.rglob("dev.json"))

    train_examples = load_scierc_relations(train_file)
    dev_examples = load_scierc_relations(dev_file)
    print(f"SciERC relation train sentences (with >=1 pair): {len(train_examples)}, dev: {len(dev_examples)}")

    label2id = {l: i for i, l in enumerate(REL_LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    print(f"Total pairs — train: {len(train_ds)}, dev: {len(dev_ds)}")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=16, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ModelClass = build_model_class()
    model = ModelClass(MODEL_NAME, len(REL_LABELS)).to(device)

    epochs = 8
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.1 * total_steps),
                                             num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for enc, e1s, e1e, e2s, e2e, wids, labels in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(enc["input_ids"], enc["attention_mask"],
                            e1s.to(device), e1e.to(device), e2s.to(device), e2e.to(device), wids)
            loss = loss_fn(logits, labels.to(device))
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for enc, e1s, e1e, e2s, e2e, wids, labels in dev_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(enc["input_ids"], enc["attention_mask"],
                                e1s.to(device), e1e.to(device), e2s.to(device), e2e.to(device), wids)
                y_pred.extend(logits.argmax(-1).cpu().tolist())
                y_true.extend(labels.tolist())
        # micro-F1 over relation classes only (exclude no_relation, matching
        # standard SciERC relation-extraction evaluation practice)
        rel_label_ids = [label2id[l] for l in SCIERC_RELATION_TYPES]
        dev_f1 = sk_f1(y_true, y_pred, labels=rel_label_ids, average="micro", zero_division=0)
        print(f"Epoch {epoch}/{epochs}  train_loss={total_loss/len(train_loader):.4f}  dev_relation_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ↳ new best (dev_relation_f1={best_f1:.4f}) — saved in memory")

    REL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), REL_CHECKPOINT_DIR / "model.pt")
    tokenizer.save_pretrained(str(REL_CHECKPOINT_DIR))
    (REL_CHECKPOINT_DIR / "labels.json").write_text(json.dumps(REL_LABELS, indent=2))
    print(f"\nSaved best relation model (dev_relation_f1={best_f1:.4f}) to {REL_CHECKPOINT_DIR}")


# --------------------------------------------------------------------------- #
# Inference — re-uses the entity spans already in {domain}_entities.json,
# groups them per-sentence via spaCy sentence splitting, classifies every
# ordered pair within a sentence, keeps only non-"no_relation" predictions.
# --------------------------------------------------------------------------- #
def run_inference(domains=("NLP", "COVID")):
    import torch
    from transformers import AutoTokenizer
    import spacy

    if not (REL_CHECKPOINT_DIR / "model.pt").exists():
        raise SystemExit(f"No trained relation model found at {REL_CHECKPOINT_DIR}. Run --train first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(REL_CHECKPOINT_DIR))
    labels = json.loads((REL_CHECKPOINT_DIR / "labels.json").read_text())
    ModelClass = build_model_class()
    model = ModelClass(MODEL_NAME, len(labels)).to(device)
    model.load_state_dict(torch.load(REL_CHECKPOINT_DIR / "model.pt", map_location=device))
    model.eval()

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    for domain in domains:
        entities_by_paper = {}
        for e in read_json(OUTPUT_DIR / f"{domain}_entities.json"):
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
                print(f"[{domain}] {n_papers}/{len(entities_by_paper)} papers "
                      f"({len(all_relations)} relations so far, {elapsed:.0f}s elapsed)", flush=True)

            for sent in doc.sents:
                sent_ents = [e for e in ents if e["char_start"] >= sent.start_char and e["char_end"] <= sent.end_char]
                if len(sent_ents) < 2:
                    continue
                # Cap pairs considered per sentence — a sentence with many
                # entities produces n*(n-1) ordered pairs, which can spike
                # into the hundreds and stall a single batch. Keep the
                # highest-confidence entities if there are too many.
                if len(sent_ents) > MAX_ENTITIES_PER_SENTENCE:
                    sent_ents = sorted(sent_ents, key=lambda e: -e["confidence"])[:MAX_ENTITIES_PER_SENTENCE]
                sent_word_starts = [tok.idx for tok in sent]
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

                # Process in sub-batches so one entity-dense sentence can't
                # spike memory/time with a single giant forward pass.
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

        print(f"[{domain}] trained relation model: {len(all_relations)} relations "
              f"across {n_papers} papers in {time.time()-t0:.1f}s")
        write_json(OUTPUT_DIR / f"{domain}_relations.json", all_relations)

        # update paper-level extracted.json with the new relations
        extracted = read_json(OUTPUT_DIR / f"{domain}_extracted.json")
        rels_by_paper = {}
        for r in all_relations:
            rels_by_paper.setdefault(r["paper_id"], []).append(r)
        for p in extracted:
            p["relations"] = rels_by_paper.get(p["paper_id"], [])
            p["n_relations"] = len(p["relations"])
        write_json(OUTPUT_DIR / f"{domain}_extracted.json", extracted)

        write_json(OUTPUT_DIR / f"{domain}_relation_method.json", {
            "method": "scibert_relation_finetuned",
            "model": MODEL_NAME,
            "checkpoint": str(REL_CHECKPOINT_DIR),
        })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--domain", choices=["NLP", "COVID", "both"], default="both")
    args = p.parse_args()
    if not (args.train or args.run):
        p.error("pass --train and/or --run")
    if args.train:
        train()
    if args.run:
        domains = ("NLP", "COVID") if args.domain == "both" else (args.domain,)
        run_inference(domains=domains)
