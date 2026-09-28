"""
Component 2 — SciERC held-out TEST-set F1 evaluation for the fine-tuned
SciBERT entity model.

Why this script exists:
  extract_entities_relations_scibert.py fine-tunes and (optionally) evaluates
  against SciERC's dev split during training, but nothing in this repo
  computed a final number against SciERC's held-out *test* split — and the
  paper's evaluation section explicitly promises "F1 scores on the SciERC
  benchmark for entity and relation extraction." This script closes that
  gap for the entity side (token-level P/R/F1 via seqeval, standard for
  BIO-tagged NER, same metric SciERC/SpERT/PURE report).

  Relation-side F1 is NOT computed here, because extract_entities_relations_
  scibert.py does not fine-tune a learned relation classifier (it reuses the
  Tier-1 dependency-pattern heuristic for relations — see that script's
  docstring). Reporting a "relation F1" against SciERC's test set would
  overstate what the heuristic relation layer actually is. If you want a
  real relation F1 number, fine-tune a span-pair relation head on SciERC's
  relation annotations (SpERT/PURE architecture) first.

Requirements: same as extract_entities_relations_scibert.py
    pip install torch transformers seqeval

Usage (after --train has produced a checkpoint):
    python3 eval_scierc_f1.py
"""
from __future__ import annotations
import json
from pathlib import Path

from extract_entities_relations_scibert import (
    CHECKPOINT_DIR, MAX_LEN, download_scierc, SCIERC_ENTITY_TYPES,
)


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
                        continue
                    tags[s] = f"B-{etype}"
                    for i in range(s + 1, e + 1):
                        tags[i] = f"I-{etype}"
                examples.append({"tokens": sent_tokens, "tags": tags})
                offset += len(sent_tokens)
    return examples


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    ner_path = CHECKPOINT_DIR / "ner"
    if not ner_path.exists():
        raise SystemExit(
            f"No fine-tuned model found at {ner_path}. Run "
            f"`python3 extract_entities_relations_scibert.py --train` first."
        )

    scierc_dir = download_scierc(CHECKPOINT_DIR.parent / "scierc_data")
    test_file = next(scierc_dir.rglob("test.json"))
    test_examples = load_scierc_ner(test_file)
    print(f"SciERC test sentences: {len(test_examples)}")

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
            # word_ids can under-cover the last word(s) if truncated; pad with "O"
            while len(pred_tags) < len(ex["tags"]):
                pred_tags.append("O")
            pred_tags = pred_tags[:len(ex["tags"])]
            y_true.append(ex["tags"])
            y_pred.append(pred_tags)

    print("\n=== SciERC held-out TEST set — entity NER (seqeval, span-level) ===\n")
    print(classification_report(y_true, y_pred, digits=4))
    overall = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "n_test_sentences": len(test_examples),
        "entity_types": SCIERC_ENTITY_TYPES,
    }
    out_path = CHECKPOINT_DIR / "scierc_test_f1.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\nOverall: P={overall['precision']:.4f} R={overall['recall']:.4f} F1={overall['f1']:.4f}")
    print(f"Saved to {out_path} — use this F1 in your Results section.")


if __name__ == "__main__":
    main()
