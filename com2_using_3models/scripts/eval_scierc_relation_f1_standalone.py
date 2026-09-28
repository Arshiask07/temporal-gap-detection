"""
Component 2 — SciERC held-out TEST-set relation F1 for the fine-tuned
SciBERT relation classifier.

Mirrors eval_scierc_f1.py's structure (entity-side) for the relation side:
  - loads checkpoints/scibert_scierc/relation/model.pt + labels.json
  - loads scierc_data/processed_data/json/test.json
  - extracts every gold relation-annotated entity pair from all 551 test
    sentences
  - runs the span-pair classifier, collects predicted labels
  - computes precision/recall/F1 across the 8 labels (no_relation + 7
    SciERC types) the same way sklearn's classification_report would
  - saves to checkpoints/scibert_scierc/scierc_relation_test_f1.json

Range check: if F1 comes back < 0.30 or > 0.85, something is wrong with
the eval script (label misalignment, wrong split, or no_relation included
in micro-F1), not the model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_relations_scibert import (
    NO_RELATION,
    REL_LABELS,
    load_scierc_relations,
    build_model_class,
    MAX_LEN,
)

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "scibert_scierc"
REL_DIR = CHECKPOINT_DIR / "relation"
SCIERC_DATA_DIR = CHECKPOINT_DIR.parent / "scierc_data"


def main():
    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import precision_score, recall_score, f1_score

    # 1. Load model + labels
    model_pt = REL_DIR / "model.pt"
    labels_path = REL_DIR / "labels.json"
    if not model_pt.exists():
        raise SystemExit(f"No model.pt at {model_pt}")
    if not labels_path.exists():
        raise SystemExit(f"No labels.json at {labels_path}")

    labels = json.loads(labels_path.read_text())
    if labels != REL_LABELS:
        print(f"WARNING: labels.json differs from REL_LABELS; using REL_LABELS")
        labels = REL_LABELS

    label2id = {l: i for i, l in enumerate(labels)}
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ModelClass = build_model_class()
    model = ModelClass(model_name, len(labels))

    state = torch.load(model_pt, map_location="cpu", mmap=True)
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # 2. Load SciERC test split
    test_path = SCIERC_DATA_DIR / "processed_data" / "json" / "test.json"
    if not test_path.exists():
        raise SystemExit(f"No test.json at {test_path}")
    test_examples = load_scierc_relations(test_path)
    print(f"SciERC test sentences (with >=1 pair): {len(test_examples)}")

    # 3. Build candidate pair list — every gold-annotated entity pair
    rows = []  # (tokens, e1_span, e2_span, gold_label)
    n_pos = 0
    for ex in test_examples:
        for i, j, label in ex["pairs"]:
            rows.append((ex["tokens"], ex["entities"][i], ex["entities"][j], label))
            if label != NO_RELATION:
                n_pos += 1
    n_pairs = len(rows)
    print(f"Test pairs (pos+neg): {n_pairs}  (positives: {n_pos})")

    if n_pairs == 0:
        raise SystemExit("No pairs found in test split — aborting.")

    # 4. Run inference in batches
    BATCH = 32
    y_true, y_pred = [], []
    with torch.no_grad():
        for start in range(0, n_pairs, BATCH):
            batch = rows[start : start + BATCH]
            tokens_batch = [r[0] for r in batch]
            enc = tokenizer(
                tokens_batch,
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(device)
            word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch))]
            e1s = torch.tensor([r[1][0] for r in batch]).to(device)
            e1e = torch.tensor([r[1][1] for r in batch]).to(device)
            e2s = torch.tensor([r[2][0] for r in batch]).to(device)
            e2e = torch.tensor([r[2][1] for r in batch]).to(device)
            gold = torch.tensor([label2id[r[3]] for r in batch]).to(device)

            logits = model(
                enc["input_ids"], enc["attention_mask"], e1s, e1e, e2s, e2e, word_ids_batch
            )
            pred_ids = logits.argmax(-1).cpu().tolist()
            y_pred.extend(pred_ids)
            y_true.extend(gold.cpu().tolist())

            done = min(start + BATCH, n_pairs)
            print(f"  ... {done}/{n_pairs} pairs scored", flush=True)

    # 5. Score — micro-F1 over the 7 typed relation classes, no_relation excluded
    typed_label_ids = [label2id[l] for l in labels if l != NO_RELATION]
    p = precision_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)
    r = recall_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=typed_label_ids, average="micro", zero_division=0)

    report = {
        "model_key": "scibert",
        "model": model_name,
        "task": "relation_classification",
        "split": "test",
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "n_test_sentences": len(test_examples),
        "n_test_pairs": n_pairs,
        "n_pos_pairs": n_pos,
        "label_set": [l for l in labels if l != NO_RELATION],
        "metric_note": (
            "micro-F1 over the 7 typed relation classes; "
            "no_relation excluded (standard SciERC relation extraction "
            "evaluation convention)."
        ),
    }
    print(f"\nSciERC TEST relation F1: P={report['precision']:.4f}  "
          f"R={report['recall']:.4f}  F1={report['f1']:.4f}")

    # 6. Range check
    if report["f1"] < 0.30 or report["f1"] > 0.85:
        print(f"\nWARNING: F1={report['f1']:.4f} is outside the expected "
              f"range [0.30, 0.85]. Something may be wrong with the eval "
              f"script, not the model.")

    # 7. Persist
    out_path = REL_DIR / "scierc_relation_test_f1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
