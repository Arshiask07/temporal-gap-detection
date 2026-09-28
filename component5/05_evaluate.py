#!/usr/bin/env python3
"""
Component 5 — Step 5: Evaluate ranked gaps via retrospective validation.

Reuses Component 3's retrospective methodology (post-cutoff co-mention check)
to score the top-K gaps produced by 04_fuse_and_rank against real holdout data
from Component 2's extracted.json paper texts.

For each domain (NLP, COVID) and each cutoff year (NLP: 2021, 2022, 2023;
COVID: 2020, 2021, 2022), runs:

    1. Alpha sweep: for alpha in [0, 0.25, 0.5, 0.75, 1.0], load the
       corresponding gaps file from 04, take the top-75, check how many have
       both entities' surface tokens co-occurring in any post-cutoff paper.
       Record hit_rate, hits, misses, candidate_gaps_scored, elapsed.

    2. Ablations:
       - structural_only  = alpha=1.0 gaps (same as alpha sweep's 1.0 entry)
       - semantic_only    = alpha=0.0 gaps (same as alpha sweep's 0.0 entry)
       - random_baseline  = randomly shuffle the top-75 alpha=0.5 pairs,
         compute hit rate over 3 seeds (42, 123, 456), report mean

Output:
    component5/output/ablation_{domain}.json
        {cutoff_year: {setting_name: {hit_rate, hits, misses,
                                       candidate_gaps_scored, elapsed_seconds,
                                       top_hits: [...]}}}
    component5/output/ablation_{domain}.md
        Markdown table: rows = settings, columns = cutoff years,
        cells = hit_rate (hits/total).  Best per domain in bold.
    component5/output/top_gaps_{domain}_latest.json
        Top-10 gaps for the latest year (NLP: 2024, COVID: 2024) at alpha=0.5,
        with surface_forms and types added.

Input:
    component5/output/gaps/{domain}_{year}_alpha{alpha}.json   (from 04)
    component2_entity_relation_extraction/output/{domain}_entities.json  (from 02)
    component2_entity_relation_extraction/output/{domain}_extracted.json  (from 02)
    component2_entity_relation_extraction/output/{domain}_relations.json  (from 02)

Runtime: CPU only, no torch/transformers/faiss needed (post-ranking eval).
Seeded: numpy.random.seed(42), random.seed(42).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────

# Paths are derived from this script's own location so the pipeline runs
# unchanged on any machine (Windows/macOS/Linux) with no hand-editing.
# Layout assumed:  <project_root>/component5/<this script>
#                  <project_root>/component2_entity_relation_extraction/output
PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"

COMP5_ROOT = PROJECT_ROOT / "component5"
COMP5_OUT = COMP5_ROOT / "output"
GAPS_DIR = COMP5_OUT / "gaps"
EVAL_DIR = COMP5_OUT  # ablation files go directly under component5/output/

# Cutoff years: must leave at least 1 year of post-cutoff data.
CUTOFF_YEARS = {
    "NLP": [2021, 2022, 2023],
    "COVID": [2020, 2021, 2022],
}

ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
ABLATION_SETTINGS = [
    ("structural_only", 1.0),
    ("semantic_only", 0.0),
    ("random_baseline", None),  # special: uses alpha=0.5 gaps, shuffled
]

TOP_K_EVAL = 75      # score this many gaps per setting per cutoff
RANDOM_SEEDS = [42, 123, 456]

LATEST_YEAR = {"NLP": 2024, "COVID": 2024}

DEGENERATE_WARN_PAIRS = 10  # warn if fewer than this many gap pairs found


# ── helpers (copied from component3, kept standalone) ─────────────────────

def _load_component2_data(domain: str):
    """Load entities, extracted (papers), and relations JSONs for *domain*.

    Returns (entities_list, papers_list, relations_list).
    Raises FileNotFoundError if any input is missing.
    """
    short = domain.split()[0].lower().capitalize()
    if short.lower() == "nlp":
        short = "NLP"
    elif short.lower().startswith("covid"):
        short = "COVID"

    ent_path = COMP2_OUT / f"{short}_entities.json"
    ext_path = COMP2_OUT / f"{short}_extracted.json"
    rel_path = COMP2_OUT / f"{short}_relations.json"

    for p in [ent_path, ext_path, rel_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Component 2 input missing: {p}\n"
                f"Run Component 2 extraction first, or check the path."
            )

    entities = json.loads(ent_path.read_text(encoding="utf-8"))
    extracted = json.loads(ext_path.read_text(encoding="utf-8"))
    relations = json.loads(rel_path.read_text(encoding="utf-8"))
    return entities, extracted, relations


def _surface_forms_from_entities(entities: list[dict]) -> dict[str, set[str]]:
    """canonical_id → set of lowercase surface tokens (≥4 chars)."""
    m: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        canon = e.get("canonical_id", "")
        surf = str(e.get("surface_form", "")).lower()
        import re
        for tok in re.findall(r"[a-z]+", surf):
            if len(tok) >= 4:
                m[canon].add(tok)
    return dict(m)


def _build_paper_index(papers: list[dict], cutoff: int) -> dict[str, set[str]]:
    """paper_id → set of lowercase tokens (≥4 chars) for post-cutoff papers."""
    import re
    idx: dict[str, set[str]] = {}
    for rec in papers:
        yr = int(rec.get("year", 0)) or 0
        if yr <= cutoff:
            continue
        pid = rec.get("paper_id", "")
        if not pid:
            continue
        txt = (str(rec.get("title", "")) + " " +
               str(rec.get("abstract", ""))).lower()
        idx[pid] = {t for t in re.findall(r"[a-z]+", txt) if len(t) >= 4}
    return idx


def _entity_surface_map(entities: list[dict]) -> dict[str, set[str]]:
    """canonical_id → set of lowercase surface tokens from all mentions."""
    import re
    m: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        canon = e.get("canonical_id", "")
        surf = str(e.get("surface_form", "")).lower()
        for tok in re.findall(r"[a-z]+", surf):
            if len(tok) >= 4:
                m[canon].add(tok)
    return dict(m)


def _get_canonical_id_info(entities: list[dict]) -> dict[str, dict]:
    """canonical_id → {surface_form, type} (most frequent surface_form)."""
    info: dict[str, dict] = {}
    freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in entities:
        cid = e.get("canonical_id", "")
        if not cid:
            continue
        surf = e.get("surface_form", "")
        typ = e.get("type", "Other")
        freq[cid][surf] += 1
        type_counts[cid][typ] += 1
    for cid in freq:
        best_surf = max(freq[cid].keys(), key=lambda s: freq[cid][s])
        best_type = max(type_counts[cid].keys(), key=lambda t: type_counts[cid][t])
        info[cid] = {"surface_form": best_surf, "type": best_type}
    return info


def check_materialization(u: str, v: str,
                          ent_surfaces: dict[str, set[str]],
                          paper_idx: dict[str, set[str]],
                          ) -> bool:
    """Return True if both u and v's surface tokens co-occur in any post-cutoff paper."""
    u_surf = ent_surfaces.get(u, set())
    v_surf = ent_surfaces.get(v, set())
    if not u_surf or not v_surf:
        return False
    for pid, p_tokens in paper_idx.items():
        if u_surf & p_tokens and v_surf & p_tokens:
            return True
    return False


def load_gaps(gaps_path: Path) -> list[dict]:
    """Load a gaps JSON file.  Returns list of gap dicts, or empty list on error."""
    if not gaps_path.exists():
        return []
    try:
        return json.loads(gaps_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def evaluate_gaps(gap_list: list[dict], ent_surfaces: dict[str, set[str]],
                  paper_idx: dict[str, set[str]],
                  ) -> dict:
    """Score a list of gap dicts against post-cutoff paper index.

    Returns {hits, misses, candidate_gaps_scored, hit_rate, top_hits}.
    """
    hits = []
    misses = []
    for g in gap_list:
        u = g.get("entity_a", "")
        v = g.get("entity_b", "")
        if not u or not v:
            continue
        materialized = check_materialization(u, v, ent_surfaces, paper_idx)
        record = {
            "entity_a": u,
            "entity_b": v,
            "materialized_post_cutoff": materialized,
        }
        # Copy over any extra fields from the gap
        for key in ("structural_score", "semantic_score", "fused_score",
                     "alpha", "year", "pre_cutoff_cooccurrences"):
            if key in g:
                record[key] = g[key]
        if materialized:
            hits.append(record)
        else:
            misses.append(record)

    scored = len(hits) + len(misses)
    hit_rate = len(hits) / scored if scored > 0 else 0.0
    return {
        "hits": len(hits),
        "misses": len(misses),
        "candidate_gaps_scored": scored,
        "hit_rate": round(hit_rate, 4),
        "top_hits": hits[:TOP_K_EVAL],
        "top_misses": misses[:TOP_K_EVAL],
    }


def evaluate_random_baseline(gap_list: list[dict], ent_surfaces: dict[str, set[str]],
                             paper_idx: dict[str, set[str]],
                             seeds: list[int] = RANDOM_SEEDS,
                             ) -> dict:
    """Shuffle gap pairs per seed, TRUNCATE to TOP_K_EVAL (matching how the
    real alpha-sweep settings are evaluated), then score. Mean hit_rate over
    seeds gives a genuine random-draw baseline.

    Previous version scored the FULL (untruncated) shuffled list every seed.
    Since evaluate_gaps scores the whole list regardless of order, shuffling
    without truncating is a no-op — every seed produced an identical
    hit_rate over the full ~500-candidate pool, which is a different
    quantity (population materialization rate) than "expected hit rate of a
    random top-K draw," and isn't comparable to the alpha-sweep numbers
    (which ARE top-K). This also explains why the printed (hits/scored)
    looked inconsistent: hits summed each seed's first-K hits found while
    the unrelated candidate_gaps_scored was just min(TOP_K_EVAL, len(list)),
    not what was actually scored.
    """
    if len(gap_list) < 2:
        return {
            "hits": 0, "misses": 0, "candidate_gaps_scored": 0,
            "hit_rate": 0.0, "top_hits": [],
            "note": "fewer than 2 gaps for random baseline",
        }

    per_seed_results = []
    for seed in seeds:
        shuffled = list(gap_list)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        top_k_random = shuffled[:TOP_K_EVAL]  # truncate BEFORE scoring, same as alpha branches
        result = evaluate_gaps(top_k_random, ent_surfaces, paper_idx)
        per_seed_results.append(result)

    hit_rates = [r["hit_rate"] for r in per_seed_results]
    mean_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    mean_hits = sum(r["hits"] for r in per_seed_results) / len(per_seed_results)
    mean_scored = sum(r["candidate_gaps_scored"] for r in per_seed_results) / len(per_seed_results)

    return {
        "hits": round(mean_hits, 2),
        "misses": round(mean_scored - mean_hits, 2),
        "candidate_gaps_scored": round(mean_scored, 2),
        "hit_rate": round(mean_hr, 4),
        "per_seed_hit_rates": hit_rates,
        "top_hits": per_seed_results[0]["top_hits"] if per_seed_results else [],
        "seeds": seeds,
        "note": f"mean over {len(seeds)} random shuffles, each truncated to "
                f"top-{TOP_K_EVAL} (matching alpha-sweep evaluation size)",
    }


# ── main evaluation per (domain, cutoff) ──────────────────────────────────

def evaluate_domain_cutoff(domain: str, cutoff: int) -> dict:
    """Run full alpha sweep + ablations for one (domain, cutoff)."""
    t_start = time.time()

    # Load Component 2 data
    entities, extracted, relations = _load_component2_data(domain)
    ent_surfaces = _entity_surface_map(entities)
    paper_idx = _build_paper_index(extracted, cutoff)
    canon_info = _get_canonical_id_info(entities)

    print(f"    Loaded: {len(entities)} entities, {len(extracted)} papers, "
          f"{len(relations)} relations")
    print(f"    Post-cutoff papers (year > {cutoff}): {len(paper_idx)}")
    if len(paper_idx) == 0:
        print(f"    [WARNING] No post-cutoff papers — all evaluations will be 0")

    # Alpha sweep
    sweep_results: dict[str, dict] = {}
    for alpha in ALPHA_SWEEP:
        alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
        gaps_path = GAPS_DIR / f"{domain}_{cutoff}_alpha{alpha_str}.json"
        gaps = load_gaps(gaps_path)

        if not gaps:
            print(f"    [WARNING] No gaps file for alpha={alpha} at cutoff {cutoff}: "
                  f"{gaps_path.name} — skipping")
            sweep_results[str(alpha)] = {
                "hit_rate": 0.0, "hits": 0, "misses": 0,
                "candidate_gaps_scored": 0, "elapsed_seconds": 0.0,
                "top_hits": [], "note": "no gaps file",
            }
            continue

        # Take top TOP_K_EVAL
        top_gaps = gaps[:TOP_K_EVAL]
        if len(top_gaps) < DEGENERATE_WARN_PAIRS:
            print(f"    [WARNING] Only {len(top_gaps)} gaps for alpha={alpha} "
                  f"at cutoff {cutoff} (fewer than {DEGENERATE_WARN_PAIRS})")

        t0 = time.time()
        result = evaluate_gaps(top_gaps, ent_surfaces, paper_idx)
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        result["alpha"] = alpha
        result["gaps_file"] = gaps_path.name
        sweep_results[str(alpha)] = result

        hr = result["hit_rate"]
        h = result["hits"]
        s = result["candidate_gaps_scored"]
        print(f"    alpha={alpha}: hit_rate={hr:.2%} ({h}/{s}) "
              f"time={result['elapsed_seconds']}s")

    # Ablations
    ablation_results: dict[str, dict] = {}

    # structural_only = alpha=1.0
    structural = sweep_results.get("1.0", {}).copy()
    structural["setting"] = "structural_only"
    ablation_results["structural_only"] = structural

    # semantic_only = alpha=0.0
    semantic = sweep_results.get("0.0", {}).copy()
    semantic["setting"] = "semantic_only"
    ablation_results["semantic_only"] = semantic

    # random_baseline: shuffle alpha=0.5 gaps
    alpha05_path = GAPS_DIR / f"{domain}_{cutoff}_alpha0.5.json"
    alpha05_gaps = load_gaps(alpha05_path)
    if alpha05_gaps:
        t0 = time.time()
        random_result = evaluate_random_baseline(alpha05_gaps, ent_surfaces, paper_idx)
        random_result["elapsed_seconds"] = round(time.time() - t0, 2)
        random_result["setting"] = "random_baseline"
        ablation_results["random_baseline"] = random_result
        print(f"    random_baseline: hit_rate={random_result['hit_rate']:.2%} "
              f"(mean {random_result['hits']}/{random_result['candidate_gaps_scored']} "
              f"hits per seed) seeds={random_result['seeds']} "
              f"per_seed_rates={random_result.get('per_seed_hit_rates')}")
    else:
        ablation_results["random_baseline"] = {
            "hit_rate": 0.0, "hits": 0, "misses": 0,
            "candidate_gaps_scored": 0, "elapsed_seconds": 0.0,
            "top_hits": [], "setting": "random_baseline",
            "note": "no alpha=0.5 gaps file",
        }

    elapsed = time.time() - t_start
    print(f"    Cutoff {cutoff} done in {elapsed:.1f}s")

    return {
        "cutoff_year": cutoff,
        "post_cutoff_range": f"{cutoff + 1}–{LATEST_YEAR[domain]}",
        "alpha_sweep": sweep_results,
        "ablations": ablation_results,
        "elapsed_seconds": round(elapsed, 2),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 5 Step 5 — evaluate gaps via retrospective validation")
    parser.add_argument("--domain", type=str, choices=["NLP", "COVID"],
                        help="single domain (default: both)")
    parser.add_argument("--cutoff", type=int,
                        help="single cutoff year (default: all for domain)")
    parser.add_argument("--force", action="store_true",
                        help="redo evaluation even if ablation JSON exists")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    np.random.seed(42)
    random.seed(42)

    domains = [args.domain] if args.domain else ["NLP", "COVID"]

    all_results: dict[str, dict] = {}

    for domain in domains:
        cutoffs = [args.cutoff] if args.cutoff else CUTOFF_YEARS[domain]
        domain_results: dict[int, dict] = {}

        print(f"\n{'='*60}")
        print(f"  Domain: {domain}")
        print(f"{'='*60}")

        # Check that gaps files exist for all required (cutoff, alpha) combos
        missing_files: list[str] = []
        for cutoff in cutoffs:
            for alpha in ALPHA_SWEEP + [1.0, 0.0]:
                alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
                gp = GAPS_DIR / f"{domain}_{cutoff}_alpha{alpha_str}.json"
                if not gp.exists():
                    missing_files.append(str(gp))
        if missing_files and not args.force:
            raise FileNotFoundError(
                f"Missing gaps files for {domain}.  Run 04_fuse_and_rank.py first.\n"
                + "\n".join(f"  - {m}" for m in missing_files[:5])
                + (f"\n  ... (+{len(missing_files)-5} more)" if len(missing_files) > 5 else "")
            )

        for cutoff in cutoffs:
            print(f"\n  Cutoff year: {cutoff}")
            if args.cutoff is None and not args.force:
                # Check if already done
                ablation_json = EVAL_DIR / f"ablation_{domain}.json"
                if ablation_json.exists():
                    existing = json.loads(ablation_json.read_text(encoding="utf-8"))
                    if str(cutoff) in existing:
                        print(f"    Already evaluated (cutoff {cutoff} in ablation JSON) "
                              f"— skipping (use --force to redo)")
                        domain_results[cutoff] = existing[str(cutoff)]
                        continue

            result = evaluate_domain_cutoff(domain, cutoff)
            domain_results[cutoff] = result

        all_results[domain] = {
            "domain": domain,
            "cutoffs": domain_results,
            "total_time_s": 0,  # filled below
        }

    # Write per-domain ablation JSON
    for domain in domains:
        ablation_json = EVAL_DIR / f"ablation_{domain}.json"
        # Restructure: {cutoff_year: {setting: result}}
        cutoffs_data = all_results[domain]["cutoffs"]
        restructured: dict[str, dict] = {}
        for cutoff, cr in cutoffs_data.items():
            restructured[str(cutoff)] = {
                **cr["alpha_sweep"],
                **cr["ablations"],
            }
        # Add metadata
        restructured["_metadata"] = {
            "domain": domain,
            "cutoffs_evaluated": list(cutoffs_data.keys()),
            "alpha_sweep": ALPHA_SWEEP,
            "ablations": [s[0] for s in ABLATION_SETTINGS],
            "top_k_evaluated": TOP_K_EVAL,
            "random_seeds": RANDOM_SEEDS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ablation_json.write_text(json.dumps(restructured, indent=2), encoding="utf-8")
        print(f"\n  Ablation JSON: {ablation_json}")

    # Write per-domain markdown table
    for domain in domains:
        md_path = EVAL_DIR / f"ablation_{domain}.md"
        cutoffs_data = all_results[domain]["cutoffs"]
        settings_order = [str(a) for a in ALPHA_SWEEP] + \
                         [s[0] for s in ABLATION_SETTINGS]

        lines = [
            f"# Component 5 — Ablation Results: {domain}",
            "",
            f"**Domain:** {domain}  ",
            f"**Gap source:** `04_fuse_and_rank.py` (FAISS-ranked, top-{TOP_K_EVAL} per setting)  ",
            f"**Validation:** Post-cutoff co-mention check (Component 3 methodology)  ",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
            "## Hit Rate by Setting × Cutoff",
            "",
            "| Setting | " + " | ".join(f"Cutoff {c}" for c in cutoffs_data) + " |",
            "|---|---|---" + "---|---" * (len(cutoffs_data) - 1),
        ]

        best_per_cutoff: dict[int, float] = {}
        for cutoff in cutoffs_data:
            best_per_cutoff[cutoff] = 0.0
            for setting in settings_order:
                if setting in cutoffs_data[cutoff].get("alpha_sweep", {}):
                    hr = cutoffs_data[cutoff]["alpha_sweep"][setting]["hit_rate"]
                    if hr > best_per_cutoff[cutoff]:
                        best_per_cutoff[cutoff] = hr
                elif setting in cutoffs_data[cutoff].get("ablations", {}):
                    hr = cutoffs_data[cutoff]["ablations"][setting]["hit_rate"]
                    if hr > best_per_cutoff[cutoff]:
                        best_per_cutoff[cutoff] = hr

        for setting in settings_order:
            cells = []
            for cutoff in cutoffs_data:
                cr = cutoffs_data[cutoff]
                val = None
                if setting in cr.get("alpha_sweep", {}):
                    val = cr["alpha_sweep"][setting]
                elif setting in cr.get("ablations", {}):
                    val = cr["ablations"][setting]
                if val is None:
                    cells.append("—")
                else:
                    hr = val.get("hit_rate", 0.0)
                    h = val.get("hits", 0)
                    s = val.get("candidate_gaps_scored", 0)
                    is_best = (hr == best_per_cutoff.get(cutoff, -1) and hr > 0)
                    cell = f"{hr:.1%} ({h}/{s})"
                    if is_best:
                        cell = f"**{cell}**"
                    cells.append(cell)
            display_name = setting
            if setting == "1.0":
                display_name = "α=1.0 (structural-only)"
            elif setting == "0.0":
                display_name = "α=0.0 (semantic-only)"
            elif setting == "0.25":
                display_name = "α=0.25"
            elif setting == "0.5":
                display_name = "α=0.5 (default)"
            elif setting == "0.75":
                display_name = "α=0.75"
            lines.append(f"| {display_name} | " + " | ".join(cells) + " |")

        lines += [
            "",
            "## Top Gaps — Latest Year (alpha=0.5)",
            "",
        ]

        latest = LATEST_YEAR[domain]
        top_gaps_path = GAPS_DIR / f"{domain}_{latest}_alpha0.5.json"
        if top_gaps_path.exists():
            top_gaps = json.loads(top_gaps_path.read_text(encoding="utf-8"))[:10]
            canon_info = _get_canonical_id_info(
                json.loads((COMP2_OUT / f"{domain.split()[0].lower().capitalize()}_entities.json").read_text(encoding="utf-8"))
            )

            lines.append(f"| Rank | Entity A | Entity B | Structural | Semantic | Fused |")
            lines.append("|---|---|---" + "---|---|---")
            for i, g in enumerate(top_gaps, 1):
                u = g.get("entity_a", "")
                v = g.get("entity_b", "")
                info_u = canon_info.get(u, {})
                info_v = canon_info.get(v, {})
                lines.append(
                    f"| {i} | {u}<br><small>{info_u.get('surface_form', '')} [{info_u.get('type', '')}]</small> | "
                    f"{v}<br><small>{info_v.get('surface_form', '')} [{info_v.get('type', '')}]</small> | "
                    f"{g.get('structural_score', 0):.4f} | "
                    f"{g.get('semantic_score', 0):.4f} | "
                    f"{g.get('fused_score', 0):.4f} |"
                )
        else:
            lines.append(f"(No gaps file for {domain} {latest} alpha=0.5)")

        lines += [
            "",
            "## Notes",
            "",
            f"- **Alpha sweep:** {ALPHA_SWEEP}  ",
            f"- **Ablations:** {', '.join(s[0] for s in ABLATION_SETTINGS)}  ",
            f"- **Top-K evaluated per setting:** {TOP_K_EVAL}  ",
            f"- **Random baseline seeds:** {RANDOM_SEEDS}  ",
            "- **Bold cells** indicate the best hit rate for that cutoff.  ",
            "- **Structural-only** (α=1.0) and **semantic-only** (α=0.0) are included in the alpha sweep.  ",
            "- **Random baseline** shuffles the top-75 α=0.5 pairs and reports mean hit rate over seeds.  ",
            "",
            "## Input Files",
            "",
            "Gap files: `component5/output/gaps/{domain}_{cutoff}_alpha{alpha}.json` (from 04)  ",
            "Component 2 data: `{domain}_entities.json`, `{domain}_extracted.json`, `{domain}_relations.json`  ",
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Ablation markdown: {md_path}")

        # Write top gaps for latest year
        latest = LATEST_YEAR[domain]
        top_gaps_path = GAPS_DIR / f"{domain}_{latest}_alpha0.5.json"
        top10_out_path = EVAL_DIR / f"top_gaps_{domain}_latest.json"
        if top_gaps_path.exists():
            top_gaps = json.loads(top_gaps_path.read_text(encoding="utf-8"))[:10]
            canon_info = _get_canonical_id_info(
                json.loads((COMP2_OUT / f"{domain.split()[0].lower().capitalize()}_entities.json").read_text(encoding="utf-8"))
            )
            enriched = []
            for g in top_gaps:
                u = g.get("entity_a", "")
                v = g.get("entity_b", "")
                enriched.append({
                    "entity_a": u,
                    "entity_b": v,
                    "surface_form_a": canon_info.get(u, {}).get("surface_form", ""),
                    "surface_form_b": canon_info.get(v, {}).get("surface_form", ""),
                    "type_a": canon_info.get(u, {}).get("type", ""),
                    "type_b": canon_info.get(v, {}).get("type", ""),
                    "structural_score": g.get("structural_score"),
                    "semantic_score": g.get("semantic_score"),
                    "fused_score": g.get("fused_score"),
                    "alpha": g.get("alpha", 0.5),
                    "year": g.get("year", latest),
                })
            top10_out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
            print(f"  Top-10 latest gaps: {top10_out_path}")
        else:
            print(f"  [WARNING] No gaps file for {domain} {latest} alpha=0.5 — "
                  f"skipping top_gaps output")

    elapsed_total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  Done.  Total time: {elapsed_total:.1f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f} MiB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
