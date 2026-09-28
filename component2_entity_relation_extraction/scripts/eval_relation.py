"""
Component 2 — SciERC held-out TEST-set F1 evaluation for the fine-tuned
relation classifier. The entity-side equivalent of eval_scierc_f1.py.

Why this exists: extract_relations_scibert.py reports dev-set F1 during
training (to pick the best epoch), but nothing computes a final number
against SciERC's held-out *test* split — the same gap eval_scierc_f1.py
closed for entities. This closes it for relations.

Standard SciERC convention: no_relation is excluded from the positive-class
P/R/F1 average (it's the "negative" class — including it would inflate the
score with an easy majority class), matching how seqeval already excludes
"O" on the entity side.

Requirements: same as extract_relations_scibert.py
    pip install torch transformers scikit-learn

Usage (after extract_relations_scibert.py --train has produced a checkpoint):
    python3 eval_relation_f1.py
    python3 eval_relation_f1.py --batch-size 1   # if you hit a memory crash
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_entities_relations_scibert import CHECKPOINT_DIR, MAX_LEN, download_scierc
from extract_relations_scibert import (
    REL_CHECKPOINT_DIR, NO_RELATION, load_scierc_relations, build_model_class,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    if not (REL_CHECKPOINT_DIR / "model.pt").exists():
        raise SystemExit(
            f"No trained relation model found at {REL_CHECKPOINT_DIR}. Run "
            f"`python3 extract_relations_scibert.py --train` first."
        )

    scierc_dir = download_scierc(CHECKPOINT_DIR.parent / "scierc_data")
    test_file = next(scierc_dir.rglob("test.json"))
    test_examples = load_scierc_relations(test_file)
    n_pairs = sum(len(ex["pairs"]) for ex in test_examples)
    print(f"SciERC test sentences: {len(test_examples)}  (pairs to classify: {n_pairs})")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(REL_CHECKPOINT_DIR))
    labels = json.loads((REL_CHECKPOINT_DIR / "labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    ModelClass = build_model_class()
    model = ModelClass("allenai/scibert_scivocab_uncased", len(labels))
    # map_location=device avoids loading onto GPU then moving, which is one
    # of the more common causes of a memory spike during load on a small
    # machine (Step 2 of the debugging plan — explicit + minimal).
    state = torch.load(REL_CHECKPOINT_DIR / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()  # REQUIRED before inference — without this, dropout stays
                  # active and (more importantly for memory) autograd keeps
                  # tracking every op unless paired with torch.no_grad() below.

    # Flatten every (tokens, entity_span_1, entity_span_2, gold_label) row
    # across all test sentences, same shape the training loop's collate fn
    # consumes.
    rows = []
    for ex in test_examples:
        for i, j, label in ex["pairs"]:
            rows.append((ex["tokens"], ex["entities"][i], ex["entities"][j], label))

    y_true, y_pred = [], []
    BATCH = args.batch_size
    with torch.no_grad():  # REQUIRED — see docstring; this is what actually
                            # prevents the memory blow-up described in the
                            # debugging plan, not batch size on its own.
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tokens_batch = [b[0] for b in batch]
            enc = tokenizer(tokens_batch, is_split_into_words=True, truncation=True,
                             max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch))]
            e1s = torch.tensor([b[1][0] for b in batch]).to(device)
            e1e = torch.tensor([b[1][1] for b in batch]).to(device)
            e2s = torch.tensor([b[2][0] for b in batch]).to(device)
            e2e = torch.tensor([b[2][1] for b in batch]).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"], e1s, e1e, e2s, e2e, word_ids_batch)
            pred_ids = logits.argmax(-1).cpu().tolist()

            y_pred.extend(pred_ids)
            y_true.extend([label2id[b[3]] for b in batch])

            if (start // BATCH) % 20 == 0:
                print(f"  {start}/{len(rows)} pairs classified", flush=True)

    # Exclude no_relation from the positive-class average — standard
    # SciERC/SpERT/PURE convention, same reasoning as seqeval excluding "O".
    rel_label_ids = [label2id[l] for l in labels if l != NO_RELATION]
    rel_label_names = [l for l in labels if l != NO_RELATION]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=rel_label_ids, average="micro", zero_division=0
    )

    print("\n=== SciERC held-out TEST set — relation classification (no_relation excluded) ===\n")
    print(classification_report(
        y_true, y_pred, labels=rel_label_ids, target_names=rel_label_names,
        digits=4, zero_division=0,
    ))

    overall = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "n_test_pairs": len(rows),
        "n_test_sentences": len(test_examples),
        "relation_types": rel_label_names,
    }
    out_path = REL_CHECKPOINT_DIR / "scierc_relation_test_f1.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\nOverall (no_relation excluded): P={overall['precision']:.4f} "
          f"R={overall['recall']:.4f} F1={overall['f1']:.4f}")
    print(f"Saved to {out_path} — use this F1 in your Results section.")

    if not (0.30 <= overall["f1"] <= 0.85):
        print(
            "\n⚠️  F1 is outside the 0.30–0.85 range published SpERT/PURE results on "
            "this benchmark typically land in. Before trusting this number, double-check "
            "label alignment — specifically that `labels.json` was loaded from the SAME "
            "training run as model.pt (a mismatched label file would silently misalign "
            "predictions to the wrong class indices, which can look like a real F1 "
            "number while actually being meaningless)."
        )


if __name__ == "__main__":
    main()