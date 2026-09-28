"""
Component 2 — Tier 2: run the full SciBERT plan locally (laptop/desktop
with real internet access), step by step.

Run this FROM the scripts/ folder (same folder as
extract_entities_relations_scibert.py, eval_scierc_f1.py, common_io.py, etc.)
since it imports them directly:

    cd cappro/component2_entity_relation_extraction/scripts
    python3 run_scibert_pipeline.py

By default it runs every step in order, stopping if the small-slice quality
checks look bad. You can also run a single step:

    python3 run_scibert_pipeline.py --step 1      # connectivity check only
    python3 run_scibert_pipeline.py --step train
    python3 run_scibert_pipeline.py --step slice
    python3 run_scibert_pipeline.py --step check
    python3 run_scibert_pipeline.py --step f1
    python3 run_scibert_pipeline.py --step full
    python3 run_scibert_pipeline.py --step report

Requirements (install once):
    pip install torch transformers datasets seqeval rapidfuzz spacy requests
    python3 -m spacy download en_core_web_sm
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# --------------------------------------------------------------------------- #
# Step 1 — confirm real internet access
# --------------------------------------------------------------------------- #
def step1_connectivity():
    print("\n=== Step 1: checking real internet access to huggingface.co ===")
    import requests
    try:
        r = requests.get("https://huggingface.co", timeout=10)
    except Exception as e:
        print(f"FAILED: {e}")
        print("You're likely on a restricted network. Switch machines, use a "
              "VPN, or run this on Colab instead.")
        sys.exit(1)
    print(f"Status: {r.status_code}")
    if r.status_code != 200:
        print("Unexpected status — treat this as a failure and fix connectivity "
              "before continuing.")
        sys.exit(1)
    print("OK — real internet access confirmed.")


# --------------------------------------------------------------------------- #
# Step 2 — sanity check the script targets the right model/dataset
# --------------------------------------------------------------------------- #
def step2_sanity_check():
    print("\n=== Step 2: confirming script targets SciBERT + SciERC ===")
    from extract_entities_relations_scibert import MODEL_NAME, SCIERC_ENTITY_TYPES
    print(f"MODEL_NAME = {MODEL_NAME}")
    print(f"SCIERC_ENTITY_TYPES = {SCIERC_ENTITY_TYPES}")
    assert MODEL_NAME == "allenai/scibert_scivocab_uncased"
    print("OK.")


# --------------------------------------------------------------------------- #
# Fine-tune step
# --------------------------------------------------------------------------- #
def step_train():
    print("\n=== Fine-tuning SciBERT on SciERC (one-time; ~10-20 min on GPU, "
          "longer on CPU) ===")
    subprocess.run(
        [sys.executable, "extract_entities_relations_scibert.py", "--train"],
        cwd=SCRIPT_DIR, check=True,
    )


# --------------------------------------------------------------------------- #
# Step 3 — small slice
# --------------------------------------------------------------------------- #
def step3_slice():
    print("\n=== Step 3: running on a small slice first (NLP, 2018 only) ===")
    subprocess.run(
        [sys.executable, "extract_entities_relations_scibert.py", "--run",
         "--domain", "NLP", "--year", "2018"],
        cwd=SCRIPT_DIR, check=True,
    )


# --------------------------------------------------------------------------- #
# Step 4 — quality thresholds on the slice
# --------------------------------------------------------------------------- #
def step4_check_quality() -> bool:
    print("\n=== Step 4: checking quality thresholds on the slice ===")
    from common_io import OUTPUT_DIR

    entities = json.loads((OUTPUT_DIR / "NLP_entities_slice.json").read_text(encoding="utf-8"))
    relations = json.loads((OUTPUT_DIR / "NLP_relations_slice.json").read_text(encoding="utf-8"))

    type_counts = Counter(e["type"] for e in entities)
    other_share = type_counts.get("Other", 0) / max(len(entities), 1)
    print(f"Entity type distribution: {dict(type_counts)}")
    print(f"Other share: {other_share:.1%}  (baseline was 55% — want well below this)")

    rel_counts = Counter(r["relation_type"] for r in relations)
    typed = sum(v for k, v in rel_counts.items() if k != "ENTITY_ASSOCIATED_WITH_ENTITY")
    typed_share = typed / max(len(relations), 1)
    print(f"Relation type distribution: {dict(rel_counts)}")
    print(f"Typed-relation share: {typed_share:.1%}  (baseline was ~5% — want much higher)")

    confs = [e["confidence"] for e in entities]
    conf_counter = Counter(round(c, 2) for c in confs)
    top = conf_counter.most_common(5)
    flat_signature = len(confs) > 0 and top[0][1] / len(confs) > 0.5
    print(f"Most common confidence values: {top}")
    if confs:
        print(f"Confidence range: min={min(confs):.3f} max={max(confs):.3f} "
              f"mean={sum(confs)/len(confs):.3f}")

    ok = (other_share < 0.55) and (typed_share > 0.05) and not flat_signature
    if ok:
        print("\n✅ Slice looks healthy — safe to continue.")
    else:
        print("\n⚠️  Slice looks off (see thresholds above) — investigate before "
              "scaling to the full corpus.")
    return ok


# --------------------------------------------------------------------------- #
# Step 5 — SciERC held-out test F1
# --------------------------------------------------------------------------- #
def step5_f1():
    print("\n=== Step 5: SciERC held-out TEST-set F1 ===")
    subprocess.run(
        [sys.executable, "eval_scierc_f1.py"], cwd=SCRIPT_DIR, check=True,
    )


# --------------------------------------------------------------------------- #
# Step 6 — full corpus
# --------------------------------------------------------------------------- #
def step6_full_run():
    print("\n=== Step 6: running on the full corpus (NLP 2018-2024 + COVID "
          "2019-2024) — this takes meaningfully longer than the slice ===")
    subprocess.run(
        [sys.executable, "extract_entities_relations_scibert.py", "--run",
         "--domain", "both"],
        cwd=SCRIPT_DIR, check=True,
    )


# --------------------------------------------------------------------------- #
# Step 7 — regenerate report, archive the baseline
# --------------------------------------------------------------------------- #
def step7_report():
    print("\n=== Step 7: archiving baseline report and regenerating "
          "COMPONENT2_REPORT.md ===")
    from common_io import REPORT_DIR

    old = REPORT_DIR / "COMPONENT2_REPORT.md"
    baseline_copy = REPORT_DIR / "COMPONENT2_REPORT_BASELINE_FALLBACK.md"
    if old.exists() and not baseline_copy.exists():
        shutil.copy(old, baseline_copy)
        print(f"Archived baseline report to {baseline_copy}")
    subprocess.run(
        [sys.executable, "validate_extraction.py"], cwd=SCRIPT_DIR, check=True,
    )


STEPS = {
    "1": step1_connectivity,
    "connectivity": step1_connectivity,
    "2": step2_sanity_check,
    "sanity": step2_sanity_check,
    "train": step_train,
    "slice": step3_slice,
    "check": step4_check_quality,
    "f1": step5_f1,
    "full": step6_full_run,
    "report": step7_report,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=list(STEPS.keys()), default=None,
                    help="run a single step; omit to run the full pipeline in order")
    args = p.parse_args()

    if args.step:
        STEPS[args.step]()
        return

    # Full pipeline, in order, stopping early if the slice quality check fails.
    step1_connectivity()
    step2_sanity_check()
    step_train()
    step3_slice()
    ok = step4_check_quality()
    if not ok:
        print("\nStopping before Step 5/6 — fix the fine-tuning first, then "
              "re-run with --step slice once you've made changes.")
        sys.exit(1)
    step5_f1()
    print("\nSlice + F1 look reasonable. Re-run with --step full when you're "
          "ready to commit GPU time to the full corpus (this script doesn't "
          "auto-continue to the full run so you can review the F1 number "
          "first).")


if __name__ == "__main__":
    main()
