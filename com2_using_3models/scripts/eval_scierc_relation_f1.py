"""
Component 2 — SciERC held-out TEST-set relation F1 evaluation.

Companion to eval_scierc_f1.py (which covers the entity side). Loads
each trained relation classifier from checkpoints/{model}_scierc/relation/
and computes precision/recall/F1 against SciERC's held-out test split
(test.json, 551 sentences), following the same micro-F1-over-relation-
classes-only convention that extract_relations_scibert.py uses for dev
during training (sklearn f1_score with labels=rel_label_ids, average=
'micro', zero_division=0). This is the standard SciERC relation-extraction
evaluation: only the 7 typed relation classes are scored, 'no_relation'
is excluded from both numerator and denominator.

Why this exists
---------------
eval_scierc_f1.py's docstring (line 14-20) explicitly states "Relation-
side F1 is NOT computed here" because the SciBERT-tier extraction path
originally used a heuristic relation layer. extract_relations_scibert.py
later added a real fine-tuned classifier, but no script ever ran a final
held-out *test* number for it. This file closes that gap, across all
three model checkpoints (scibert / roberta / pubmedbert), so the
COMPONENT2_MODEL_COMPARISON.md can carry relation F1 alongside entity F1
as the paper's evaluation section promises.

Standard reference range (for sanity-checking the output)
---------------------------------------------------------
SciERC relation F1 in published literature (SpERT, Eberts & Ulges 2019:
~0.65; PURE, Zhong & Chen 2021: ~0.70) is reported with *predicted*
entity spans. We use *gold* entity spans here (the standard convention
for SciERC relation classification evaluation), so numbers in the
0.50-0.70 range are normal. Anything below ~0.30 or above ~0.85 should
be treated as a bug (label misalignment, wrong split, or 'no_relation'
being accidentally included in micro-F1).

Requirements
------------
Same as extract_relations_scibert.py:
    pip install torch transformers scikit-learn

Usage
-----
    # one model
    python3 eval_scierc_relation_f1.py --model scibert
    python3 eval_scierc_relation_f1.py --model roberta
    python3 eval_scierc_relation_f1.py --model pubmedbert

    # all three
    python3 eval_scierc_relation_f1.py --model all

    # custom paths
    python3 eval_scierc_relation_f1.py --model scibert \
        --checkpoint-root /path/to/checkpoints \
        --output-dir /path/to/output

Outputs
-------
One JSON per model at <checkpoint>/{model}_scierc/scierc_relation_test_f1.json:
    {
      "model_key": "scibert",
      "model": "allenai/scibert_scivocab_uncased",
      "task": "relation_classification",
      "split": "test",
      "precision": 0.xxxx,
      "recall":    0.xxxx,
      "f1":        0.xxxx,
      "n_test_sentences": 551,
      "n_test_pairs":     N,
      "n_pos_pairs":      N (gold positive relation pairs)
      "label_set": ["Used-for", "Feature-of", "Hyponym-of",
                    "Part-of", "Compare", "Conjunction", "Evaluate-for"]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import permutations

# Make sibling scripts importable (extract_relations_scibert.py provides
# NO_RELATION, REL_LABELS, load_scierc_relations, build_model_class)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_relations_scibert import (
    NO_RELATION, REL_LABELS, load_scierc_relations, build_model_class,
    MAX_LEN,
)


DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] \
    / "component2_entity_relation_extraction" / "scripts" / "checkpoints"

# NOTE: For a standalone eval script matching eval_scierc_f1.py's role
# (entity-side, single-model SciBERT), use a separate file at
#   component2_entity_relation_extraction/scripts/eval_scierc_relation_f1.py
# that loads <CHECKPOINT_DIR>/relation/model.pt + labels.json,
# enumerates the 551 SciERC test sentences, runs the span-pair classifier,
# computes micro-F1 over the 7 typed classes (no_relation excluded), and
# saves to <CHECKPOINT_DIR>/scierc_relation_test_f1.json.
# ────────────────────────────────────────────────────────────────────────── #
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "checkpoints"   # per-model files live next to checkpoint

MODEL_REGISTRY = {
    "scibert":    "allenai/scibert_scivocab_uncased",
    "roberta":    "roberta-base",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


# ────────────────────────────────────────────────────────────────────────── #
# Checkpoint / model loading
# ────────────────────────────────────────────────────────────────────────── #
def _load_model_for_eval(model_key: str, checkpoint_root: Path):
    """Load the fine-tuned relation head for a given model_key.

    Mirrors the checkpoint layout used by extract_relations_scibert.py
    and extract_entities_relations_multimodel.py:
        <checkpoint_root>/<model_key>_scierc/relation/{model.pt, tokenizer.*, labels.json}
    """
    import torch
    from transformers import AutoTokenizer

    rel_dir = checkpoint_root / f"{model_key}_scierc" / "relation"
    model_pt = rel_dir / "model.pt"
    labels_path = rel_dir / "labels.json"
    if not model_pt.exists():
        raise SystemExit(
            f"No trained relation model at {model_pt}. "
            f"Run extract_entities_relations_multimodel.py --model {model_key} "
            f"--step train-relation first."
        )
    if not labels_path.exists():
        raise SystemExit(f"Missing labels.json at {labels_path}.")

    labels = json.loads(labels_path.read_text())
    if labels != REL_LABELS:
        # We rely on the same label ordering as training; warn but continue
        print(f"  WARNING: {labels_path} labels do not match REL_LABELS from "
              f"extract_relations_scibert.py. Got {labels}, expected {REL_LABELS}.")
        labels = REL_LABELS

    model_name = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ModelClass = build_model_class()
    model = ModelClass(model_name, len(labels))
    state = torch.load(model_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, labels, device


# ────────────────────────────────────────────────────────────────────────── #
# Evaluation
# ────────────────────────────────────────────────────────────────────────── #
def evaluate_one_model(model_key: str, checkpoint_root: Path, output_dir: Path,
                        scierc_test_path: Path) -> dict:
    """Compute SciERC test-set relation F1 for one model.

    Convention: micro-averaged F1 over the 7 typed relation classes,
    'no_relation' excluded (matches extract_relations_scibert.py training
    dev metric, and the standard SciERC relation-extraction eval).
    """
    import torch
    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f"\n=== [{model_key}] SciERC relation TEST eval ===")

    # 1. Load the model
    model, tokenizer, labels, device = _load_model_for_eval(model_key, checkpoint_root)
    label2id = {l: i for i, l in enumerate(labels)}

    # 2. Load gold SciERC test split (same loader training uses)
    test_examples = load_scierc_relations(scierc_test_path)
    print(f"[{model_key}] SciERC test sentences (with >=1 pair): {len(test_examples)}")

    # 3. Build candidate pair list — exactly the same set training saw
    # (positive pairs + a sample of negative pairs), so the metric is
    # directly comparable to the dev number printed during training.
    rows = []  # (tokens, e1_span, e2_span, gold_label)
    n_pos = 0
    for ex in test_examples:
        for i, j, label in ex["pairs"]:
            rows.append((ex["tokens"], ex["entities"][i], ex["entities"][j], label))
            if label != NO_RELATION:
                n_pos += 1
    n_pairs = len(rows)
    print(f"[{model_key}] test pairs (pos+neg): {n_pairs}  (positives: {n_pos})")

    if n_pairs == 0:
        raise SystemExit(f"[{model_key}] No pairs found in test split — aborting.")

    # 4. Run inference in batches
    BATCH = 32
    y_true, y_pred = [], []
    with torch.no_grad():
        for start in range(0, n_pairs, BATCH):
            batch = rows[start:start + BATCH]
            tokens_batch = [r[0] for r in batch]
            enc = tokenizer(tokens_batch, is_split_into_words=True,
                            truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch))]
            e1s = torch.tensor([r[1][0] for r in batch]).to(device)
            e1e = torch.tensor([r[1][1] for r in batch]).to(device)
            e2s = torch.tensor([r[2][0] for r in batch]).to(device)
            e2e = torch.tensor([r[2][1] for r in batch]).to(device)
            gold = torch.tensor([label2id[r[3]] for r in batch]).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"],
                           e1s, e1e, e2s, e2e, word_ids_batch)
            pred_ids = logits.argmax(-1).cpu().tolist()
            y_pred.extend(pred_ids)
            y_true.extend(gold.cpu().tolist())

            if (start // BATCH) % 5 == 0:
                done = min(start + BATCH, n_pairs)
                print(f"  ... {done}/{n_pairs} pairs scored", flush=True)

    # 5. Score — micro over typed relation classes only
    typed_label_ids = [label2id[l] for l in labels if l != NO_RELATION]
    p = precision_score(y_true, y_pred, labels=typed_label_ids,
                        average="micro", zero_division="0")
    r = recall_score(y_true, y_pred, labels=typed_label_ids,
                     average="micro", zero_division="0")
    f1 = f1_score(y_true, y_pred, labels=typed_label_ids,
                  average="micro", zero_division="0")

    report = {
        "model_key": model_key,
        "model": MODEL_REGISTRY[model_key],
        "task": "relation_classification",
        "split": "test",
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(float(f1), 4),
        "n_test_sentences": len(test_examples),
        "n_test_pairs": n_pairs,
        "n_pos_pairs": n_pos,
        "label_set": [l for l in labels if l != NO_RELATION],
        "metric_note": ("micro-F1 over the 7 SciERC relation classes; "
                        "'no_relation' excluded (standard SciERC relation "
                        "extraction evaluation convention, matches the "
                        "training-time dev metric in extract_relations_scibert.py).")
    }
    print(f"[{model_key}] SciERC TEST relation F1: "
          f"P={report['precision']:.4f}  R={report['recall']:.4f}  F1={report['f1']:.4f}")

    # 6. Persist next to the checkpoint so it lives with its model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_scierc" / "scierc_relation_test_f1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[{model_key}] Saved → {out_path}")
    return report


# ────────────────────────────────────────────────────────────────────────── #
# CLI
# ────────────────────────────────────────────────────────────────────────── #
def _resolve_scierc_test_path(checkpoint_root: Path) -> Path:
    """Find test.json under <checkpoint_root>/scierc_data/processed_data/json/.

    extract_relations_scibert.py::train() downloads the corpus to
    <CHECKPOINT_DIR.parent>/scierc_data, i.e. <checkpoint_root>/scierc_data,
    and we follow the same convention.
    """
    scierc_dir = checkpoint_root / "scierc_data"
    # Walk down — exact layout produced by extract_entities_relations_scibert.py
    candidate = scierc_dir / "processed_data" / "json" / "test.json"
    if candidate.exists():
        return candidate
    found = list(scierc_dir.rglob("test.json"))
    if not found:
        raise SystemExit(
            f"No SciERC test.json under {scierc_dir}. Run "
            f"extract_entities_relations_multimodel.py --model <any> "
            f"--step train-ner first (it downloads + extracts the corpus)."
        )
    return found[0]


def main():
    ap = argparse.ArgumentParser(
        description="SciERC held-out TEST-set relation F1 for any of "
                    "the three fine-tuned relation classifiers (scibert, "
                    "roberta, pubmedbert).")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"],
                    default="all",
                    help="Which model checkpoint to evaluate.")
    ap.add_argument("--checkpoint-root", type=Path,
                    default=DEFAULT_CHECKPOINT_ROOT,
                    help="Directory containing <model>_scierc/ subdirs. "
                         f"Default: {DEFAULT_CHECKPOINT_ROOT}")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Where to write scierc_relation_test_f1.json "
                         "(default: same as checkpoint-root, alongside "
                         "the model dir).")
    ap.add_argument("--scierc-test", type=Path, default=None,
                    help="Override path to SciERC test.json (auto-detected "
                         "under <checkpoint-root>/scierc_data otherwise).")
    args = ap.parse_args()

    scierc_test = args.scierc_test or _resolve_scierc_test_path(args.checkpoint_root)
    print(f"Using SciERC test split: {scierc_test}")

    targets = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    results = {}
    for mk in targets:
        try:
            results[mk] = evaluate_one_model(
                mk, args.checkpoint_root, args.output_dir, scierc_test)
        except SystemExit as e:
            print(f"  ⏭  [{mk}] skipped: {e}")
            results[mk] = {"model_key": mk, "status": "skipped",
                           "reason": str(e)}

    # Also drop a combined summary file for convenience
    summary_path = args.output_dir / "scierc_relation_test_f1_all.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()