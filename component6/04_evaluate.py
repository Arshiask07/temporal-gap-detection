#!/usr/bin/env python3
"""
Component 6 — Step 4: Evaluate velocity re-ranking via retrospective validation.

Compares three ranking strategies per (domain, cutoff):
  1. fused_only  — Component 5's alpha=0.5 gaps (baseline)
  2. velocity    — Component 6's velocity-reranked gaps (priority-sorted)
  3. random      — random draw of 75 from the candidate pool

Uses the same post-cutoff co-mention methodology as Component 5's 05_evaluate.py
and Component 3 — kept standalone, no imports from those packages.

Output per domain:
  - component6/output/ablation_velocity_{domain}.json
  - component6/output/ablation_velocity_{domain}.md
  - component6/output/top_velocity_gaps_{domain}_latest.json

Input:
  - component6/output/reranked/{domain}_{cutoff}_alpha0.5.json  (velocity gaps)
  - component5/output/gaps/{domain}_{cutoff}_alpha0.5.json       (fused baseline)
  - component2_entity_relation_extraction/output/{domain}_entities.json
  - component2_entity_relation_extraction/output/{domain}_extracted.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"
COMP5_OUT = PROJECT_ROOT / "component5" / "output"
COMP5_GAPS = COMP5_OUT / "gaps"
COMP6_OUT = PROJECT_ROOT / "component6" / "output"
RERANKED_DIR = COMP6_OUT / "reranked"
EVAL_DIR = COMP6_OUT  # ablation files go directly under component6/output/

# ── constants ────────────────────────────────────────────────────────────────

CUTOFF_YEARS = {
    "NLP": [2021, 2022, 2023],
    "COVID": [2020, 2021, 2022],
}

LATEST_YEAR = {"NLP": 2024, "COVID": 2024}

TOP_K_EVAL = 75
RANDOM_SEEDS = [42, 123, 456]
DEGENERATE_WARN_PAIRS = 10

# ── helpers (copied from component5/05_evaluate.py, kept standalone) ────────

def _load_component2_data(domain: str):
    """Load entities + extracted JSONs for *domain*."""
    short = _domain_short(domain)
    ent_path = COMP2_OUT / f"{short}_entities.json"
    ext_path = COMP2_OUT / f"{short}_extracted.json"
    for p in [ent_path, ext_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Component 2 input missing: {p}\n"
                f"Run Component 2 extraction first, or check the path."
            )
    entities = json.loads(ent_path.read_text(encoding="utf-8"))
    extracted = json.loads(ext_path.read_text(encoding="utf-8"))
    return entities, extracted


def _domain_short(domain: str) -> str:
    s = domain.split()[0].lower().capitalize()
    if s.lower() == "nlp":
        return "NLP"
    if s.lower().startswith("covid"):
        return "COVID"
    return s


def _entity_surface_map(entities: list[dict]) -> dict[str, set[str]]:
    """canonical_id → set of lowercase surface tokens (≥4 chars)."""
    m: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        canon = e.get("canonical_id", "")
        surf = str(e.get("surface_form", "")).lower()
        for tok in re.findall(r"[a-z]+", surf):
            if len(tok) >= 4:
                m[canon].add(tok)
    return dict(m)


def _build_paper_index(papers: list[dict], cutoff: int) -> dict[str, set[str]]:
    """paper_id → set of lowercase tokens (≥4 chars) for post-cutoff papers."""
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


def check_materialization(u: str, v: str,
                          ent_surfaces: dict[str, set[str]],
                          paper_idx: dict[str, set[str]]) -> bool:
    """Return True if both u and v's surface tokens co-occur in any post-cutoff paper."""
    u_surf = ent_surfaces.get(u, set())
    v_surf = ent_surfaces.get(v, set())
    if not u_surf or not v_surf:
        return False
    for pid, p_tokens in paper_idx.items():
        if u_surf & p_tokens and v_surf & p_tokens:
            return True
    return False


def load_gaps(path: Path) -> list[dict]:
    """Load a gaps JSON file.  Returns list of gap dicts, or empty list on error."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def evaluate_gaps(gap_list: list[dict], ent_surfaces: dict[str, set[str]],
                  paper_idx: dict[str, set[str]]) -> dict:
    """Score a list of gap dicts against post-cutoff paper index."""
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
        for key in ("structural_score", "semantic_score", "fused_score",
                     "alpha", "year", "vel_u", "vel_v", "priority"):
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
                             seeds: list[int] = RANDOM_SEEDS) -> dict:
    """Random draw of 75 from candidate pool, mean hit_rate over seeds."""
    if len(gap_list) < 2:
        return {
            "hits": 0, "misses": 0, "candidate_gaps_scored": 0,
            "hit_rate": 0.0, "top_hits": [],
            "note": "fewer than 2 gaps for random baseline",
        }

    per_seed_results = []
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = list(gap_list)
        rng.shuffle(shuffled)
        top_k_random = shuffled[:TOP_K_EVAL]
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
        "note": f"mean over {len(seeds)} random draws of {TOP_K_EVAL} from candidate pool",
    }


def _get_canonical_id_info(entities: list[dict]) -> dict[str, dict]:
    """canonical_id → {surface_form, type} (most frequent)."""
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


# ── main evaluation per (domain, cutoff) ────────────────────────────────────

def evaluate_domain_cutoff(domain: str, cutoff: int) -> dict:
    """Run fused-only vs velocity vs random for one (domain, cutoff)."""
    t_start = time.time()

    entities, extracted = _load_component2_data(domain)
    ent_surfaces = _entity_surface_map(entities)
    paper_idx = _build_paper_index(extracted, cutoff)
    canon_info = _get_canonical_id_info(entities)

    print(f"    Loaded: {len(entities)} entities, {len(extracted)} papers")
    print(f"    Post-cutoff papers (year > {cutoff}): {len(paper_idx)}")
    if len(paper_idx) == 0:
        print(f"    [WARNING] No post-cutoff papers — all evaluations will be 0")

    # 1. Fused-only (Component 5 baseline, alpha=0.5)
    fused_path = COMP5_GAPS / f"{domain}_{cutoff}_alpha0.5.json"
    fused_gaps = load_gaps(fused_path)
    fused_result = None
    if fused_gaps:
        fused_top = fused_gaps[:TOP_K_EVAL]
        if len(fused_top) < DEGENERATE_WARN_PAIRS:
            print(f"    [WARNING] Only {len(fused_top)} fused gaps for cutoff {cutoff}")
        t0 = time.time()
        fused_result = evaluate_gaps(fused_top, ent_surfaces, paper_idx)
        fused_result["elapsed_seconds"] = round(time.time() - t0, 2)
        fused_result["gap_source"] = "fused_only (Component 5 alpha=0.5)"
        hr = fused_result["hit_rate"]
        print(f"    fused_only: hit_rate={hr:.2%} ({fused_result['hits']}/{fused_result['candidate_gaps_scored']})")
    else:
        print(f"    [WARNING] No fused gaps file for cutoff {cutoff}")

    # 2. Velocity-reranked (Component 6, priority-sorted, alpha=0.5)
    vel_path = RERANKED_DIR / f"{domain}_{cutoff}_alpha0.5.json"
    vel_gaps = load_gaps(vel_path)
    vel_result = None
    if vel_gaps:
        vel_top = vel_gaps[:TOP_K_EVAL]
        if len(vel_top) < DEGENERATE_WARN_PAIRS:
            print(f"    [WARNING] Only {len(vel_top)} velocity gaps for cutoff {cutoff}")
        t0 = time.time()
        vel_result = evaluate_gaps(vel_top, ent_surfaces, paper_idx)
        vel_result["elapsed_seconds"] = round(time.time() - t0, 2)
        vel_result["gap_source"] = "velocity (Component 6 priority-ranked)"
        hr = vel_result["hit_rate"]
        print(f"    velocity:   hit_rate={hr:.2%} ({vel_result['hits']}/{vel_result['candidate_gaps_scored']})")
    else:
        print(f"    [WARNING] No velocity gaps file for cutoff {cutoff}")

    # 3. Random baseline (from velocity gaps pool — the re-ranked candidate set)
    random_result = None
    pool = vel_gaps if vel_gaps else fused_gaps
    if pool:
        t0 = time.time()
        random_result = evaluate_random_baseline(pool, ent_surfaces, paper_idx)
        random_result["elapsed_seconds"] = round(time.time() - t0, 2)
        random_result["gap_source"] = "random_baseline (random draw of 75 from candidate pool)"
        hr = random_result["hit_rate"]
        print(f"    random:     hit_rate={hr:.2%} "
              f"(mean {random_result['hits']}/{random_result['candidate_gaps_scored']} "
              f"per seed) seeds={random_result['seeds']}")
    else:
        print(f"    [WARNING] No gap pool for random baseline at cutoff {cutoff}")

    elapsed = time.time() - t_start
    print(f"    Cutoff {cutoff} done in {elapsed:.1f}s")

    return {
        "cutoff_year": cutoff,
        "post_cutoff_range": f"{cutoff + 1}–{LATEST_YEAR[domain]}",
        "fused_only": fused_result or _empty_result("fused_only"),
        "velocity": vel_result or _empty_result("velocity"),
        "random_baseline": random_result or _empty_result("random_baseline"),
        "elapsed_seconds": round(elapsed, 2),
    }


def _empty_result(setting: str) -> dict:
    return {
        "hit_rate": 0.0, "hits": 0, "misses": 0,
        "candidate_gaps_scored": 0, "elapsed_seconds": 0.0,
        "top_hits": [], "gap_source": setting,
        "note": "no gaps available",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 6 Step 4 — evaluate velocity re-ranking")
    parser.add_argument("--domain", choices=["NLP", "COVID"],
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

        # Check input files exist
        missing = []
        for cutoff in cutoffs:
            for src in [COMP5_GAPS / f"{domain}_{cutoff}_alpha0.5.json",
                        RERANKED_DIR / f"{domain}_{cutoff}_alpha0.5.json"]:
                if not src.exists():
                    missing.append(str(src))
        if missing and not args.force:
            raise FileNotFoundError(
                f"Missing input files for {domain}. Run 03_rerank.py first.\n"
                + "\n".join(f"  - {m}" for m in missing[:5])
                + (f"\n  ... (+{len(missing)-5} more)" if len(missing) > 5 else "")
            )

        for cutoff in cutoffs:
            print(f"\n  Cutoff year: {cutoff}")
            if args.cutoff is None and not args.force:
                ablation_json = EVAL_DIR / f"ablation_velocity_{domain}.json"
                if ablation_json.exists():
                    existing = json.loads(ablation_json.read_text(encoding="utf-8"))
                    if str(cutoff) in existing:
                        print(f"    Already evaluated (cutoff {cutoff}) — skipping "
                              f"(use --force to redo)")
                        domain_results[cutoff] = existing[str(cutoff)]
                        continue

            domain_results[cutoff] = evaluate_domain_cutoff(domain, cutoff)

        all_results[domain] = {
            "domain": domain,
            "cutoffs": domain_results,
            "total_time_s": 0,  # filled below
        }

    # Write per-domain ablation JSON
    for domain in domains:
        ablation_json = EVAL_DIR / f"ablation_velocity_{domain}.json"
        cutoffs_data = all_results[domain]["cutoffs"]
        restructured: dict[str, dict] = {}
        for cutoff, cr in cutoffs_data.items():
            restructured[str(cutoff)] = {
                "fused_only": cr["fused_only"],
                "velocity": cr["velocity"],
                "random_baseline": cr["random_baseline"],
            }
        restructured["_metadata"] = {
            "domain": domain,
            "cutoffs_evaluated": list(cutoffs_data.keys()),
            "vel_window": "2022-2024",
            "top_k_evaluated": TOP_K_EVAL,
            "random_seeds": RANDOM_SEEDS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ablation_json.write_text(json.dumps(restructured, indent=2), encoding="utf-8")
        print(f"\n  Ablation JSON: {ablation_json}")

    # Write per-domain markdown table
    for domain in domains:
        md_path = EVAL_DIR / f"ablation_velocity_{domain}.md"
        cutoffs_data = all_results[domain]["cutoffs"]
        settings_order = ["fused_only", "velocity", "random_baseline"]

        vel_info = _velocity_path_info(domain)
        cit_path_info = _citation_path_info(domain)

        lines = [
            f"# Component 6 — Ablation Results: {domain}",
            "",
            f"**Domain:** {domain}  ",
            f"**Citation path:** {cit_path_info}  ",
            f"**Gap source:** velocity re-ranked (priority = fused_score × mean(vel_u, vel_v))  ",
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
            for cutoff in cutoffs_data:
                val = cutoffs_data[cutoff].get(setting)
                if val and val.get("hit_rate", 0) > best_per_cutoff[cutoff]:
                    best_per_cutoff[cutoff] = val["hit_rate"]

        for setting in settings_order:
            cells = []
            for cutoff in cutoffs_data:
                val = cutoffs_data[cutoff].get(setting)
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
            display = {
                "fused_only": "fused_only (C5 baseline)",
                "velocity": "velocity (C6 re-ranked)",
                "random_baseline": "random_baseline",
            }.get(setting, setting)
            lines.append(f"| {display} | " + " | ".join(cells) + " |")

        lines += [
            "",
            "### Top Velocity Gaps — Latest Year",
            "",
        ]

        latest = LATEST_YEAR[domain]
        vel_latest_path = RERANKED_DIR / f"{domain}_{latest}_alpha0.5.json"
        if vel_latest_path.exists():
            vel_gaps = json.loads(vel_latest_path.read_text(encoding="utf-8"))[:10]
            canon_info = _get_canonical_id_info(
                json.loads((COMP2_OUT / f"{_domain_short(domain)}_entities.json").read_text(encoding="utf-8"))
            )

            lines.append(f"| Rank | Entity A | Entity B | Fused | vel_u | vel_v | Priority |")
            lines.append("|---|---|---" + "---|---|---")

            for i, g in enumerate(vel_gaps, 1):
                u = g.get("entity_a", "")
                v = g.get("entity_b", "")
                info_u = canon_info.get(u, {})
                info_v = canon_info.get(v, {})
                lines.append(
                    f"| {i} | {u}<br><small>{info_u.get('surface_form', '')} [{info_u.get('type', '')}]</small> | "
                    f"{v}<br><small>{info_v.get('surface_form', '')} [{info_v.get('type', '')}]</small> | "
                    f"{g.get('fused_score', 0):.4f} | "
                    f"{g.get('vel_u', 0):.4f} | "
                    f"{g.get('vel_v', 0):.4f} | "
                    f"{g.get('priority', 0):.4f} |"
                )
        else:
            lines.append(f"(No velocity gaps file for {domain} {latest})")

        lines += [
            "",
            "## Notes",
            "",
            f"- **Velocity window:** 2022–2024  ",
            f"- **Velocity formula:** vel(e) = (c_e[2024] − c_e[2022]) / 2  ",
            f"- **Priority formula:** priority = fused_score × mean(vel_u, vel_v)  ",
            f"- **Citation path:** {cit_path_info}  ",
            f"- **Fused-only** uses Component 5's alpha=0.5 gaps (top-75)  ",
            f"- **Velocity** uses Component 6's priority-sorted gaps (top-75)  ",
            f"- **Random baseline** draws 75 pairs uniformly from the candidate pool, "
            f"mean over seeds {RANDOM_SEEDS}  ",
            f"- **Random baseline pool:** {vel_info}  ",
            "- **Bold cells** indicate the best hit rate for that cutoff.  ",
            "",
            "## Input Files",
            "",
            "Velocity gaps: `component6/output/reranked/{domain}_{cutoff}_alpha0.5.json` (from 03_rerank.py)  ",
            "Fused baseline: `component5/output/gaps/{domain}_{cutoff}_alpha0.5.json` (from 04_fuse_and_rank.py)  ",
            "Component 2 data: `{domain}_entities.json`, `{domain}_extracted.json`  ",
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Ablation markdown: {md_path}")

        # Write top velocity gaps for latest year
        latest = LATEST_YEAR[domain]
        vel_latest_path = RERANKED_DIR / f"{domain}_{latest}_alpha0.5.json"
        top10_out = EVAL_DIR / f"top_velocity_gaps_{domain}_latest.json"
        if vel_latest_path.exists():
            vel_gaps = json.loads(vel_latest_path.read_text(encoding="utf-8"))[:10]
            canon_info = _get_canonical_id_info(
                json.loads((COMP2_OUT / f"{_domain_short(domain)}_entities.json").read_text(encoding="utf-8"))
            )
            enriched = []
            for g in vel_gaps:
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
                    "vel_u": g.get("vel_u"),
                    "vel_v": g.get("vel_v"),
                    "priority": g.get("priority"),
                    "alpha": g.get("alpha", 0.5),
                    "year": g.get("year", latest),
                })
            top10_out.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
            print(f"  Top-10 latest velocity gaps: {top10_out}")
        else:
            print(f"  [WARNING] No velocity gaps file for {domain} {latest}")

    elapsed_total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for domain in domains:
        total_s = sum(cr["elapsed_seconds"] for cr in all_results[domain]["cutoffs"].values())
        all_results[domain]["total_time_s"] = round(total_s, 2)

    print(f"\n{'='*60}")
    print(f"  Done.  Total time: {elapsed_total:.1f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f} MiB")
    print(f"{'='*60}")


def _velocity_path_info(domain: str) -> str:
    """Return a human-readable description of the velocity path used."""
    vp = COMP6_OUT / f"entity_velocity_{domain}.json"
    if not vp.exists():
        return "unknown"
    data = json.loads(vp.read_text(encoding="utf-8"))
    return data.get("path", "unknown")


def _citation_path_info(domain: str) -> str:
    """Return a human-readable citation path label."""
    fb = COMP6_OUT / "citations" / f"{domain}_FALLBACK.txt"
    if fb.exists():
        return "mention-velocity fallback (S2 API unreachable)"
    return "Semantic Scholar API"


if __name__ == "__main__":
    main()
