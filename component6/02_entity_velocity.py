#!/usr/bin/env python3
"""
Component 6 — Step 2: Aggregate paper-level citations into entity×year velocity.

Reads:
  - component6/output/citations/{domain}_paper_citations.json  (or fallback flag)
  - component2_entity_relation_extraction/output/{domain}_entities.json  (entity→paper→year)

Produces:
  - component6/output/entity_velocity_{domain}.json
      {canonical_id: {year: velocity, ...}, ...}
      velocity(e) = (c_e[y2] − c_e[y1]) / (y2 − y1) where (y1, y2) = VEL_WINDOW

Velocity is computed two ways depending on what's available:

PRIMARY (citation path): c_e[y] = sum of citation counts of all papers mentioning
    entity e in year y.  Read from S2 citation cache.

FALLBACK (mention-velocity path): c_e[y] = count of papers mentioning entity e in
    year y (from entities.json paper_id/year fields).  Used when S2 API is
    unreachable and the fallback flag exists.

Usage:
    python3 component6/02_entity_velocity.py [--force] [--domain NLP|COVID] [--vel-window Y1 Y2]
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"
COMP6_OUT = PROJECT_ROOT / "component6" / "output"
CITATIONS_DIR = COMP6_OUT / "citations"

# ── defaults ─────────────────────────────────────────────────────────────────

VEL_WINDOW_DEFAULT = (2022, 2024)  # must match dashboard/config.py VEL_WINDOW


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_entities(domain: str) -> list[dict]:
    path = COMP2_OUT / f"{domain}_entities.json"
    if not path.exists():
        raise FileNotFoundError(f"Component 2 entities missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_fallback(domain: str) -> bool:
    return (CITATIONS_DIR / f"{domain}_FALLBACK.txt").exists()


def _load_citation_cache(domain: str) -> dict[str, int]:
    """Load S2 citation cache.  Returns {paper_id: citation_count}."""
    path = CITATIONS_DIR / f"{domain}_paper_citations.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items() if v > 0}


def _build_entity_year_citations(
    entities: list[dict],
    citation_cache: dict[str, int] | None,
    use_citations: bool,
) -> dict[str, dict[int, float]]:
    """
    Build entity→{year: aggregate} matrix.

    If use_citations=True: aggregate = sum of citation counts of papers
        mentioning the entity in that year (from citation_cache).
    If use_citations=False: aggregate = count of papers mentioning the entity
        in that year (mention velocity — fully from entities.json).
    """
    # entity → {year: [citation_counts or 1s]}
    e_yr_vals: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for e in entities:
        canon = e.get("canonical_id", "")
        if not canon:
            continue
        pid = e.get("paper_id", "")
        yr = int(e.get("year", 0)) or 0
        if not pid or not yr:
            continue

        if use_citations and citation_cache is not None:
            # Primary path: citation count of the paper (may be 0 if not cached)
            val = float(citation_cache.get(pid, 0.0))
        else:
            # Fallback path: paper counts as 1 mention
            val = 1.0

        e_yr_vals[canon][yr].append(val)

    # Aggregate: sum per (entity, year)
    result: dict[str, dict[int, float]] = {}
    for canon, yr_map in e_yr_vals.items():
        result[canon] = {}
        for yr, vals in yr_map.items():
            result[canon][yr] = sum(vals)

    return result


def _compute_velocities(
    entity_yr: dict[str, dict[int, float]],
    y1: int,
    y2: int,
) -> dict[str, float]:
    """
    velocity(e) = (c_e[y2] − c_e[y1]) / (y2 − y1)

    Entities not present in either year get velocity 0.0 (they have no
    measurable velocity in the window — excluded from re-ranking boost).
    """
    denom = float(y2 - y1)
    if denom <= 0:
        raise ValueError(f"VEL_WINDOW must have y2 > y1, got ({y1}, {y2})")

    result: dict[str, float] = {}
    for canon, yr_map in entity_yr.items():
        c_y1 = yr_map.get(y1, 0.0)
        c_y2 = yr_map.get(y2, 0.0)
        vel = (c_y2 - c_y1) / denom
        if vel != 0.0 or c_y1 > 0 or c_y2 > 0:
            result[canon] = vel
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 6 Step 2 — entity citation/mention velocity"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo even if velocity JSON exists",
    )
    parser.add_argument(
        "--domain",
        choices=["NLP", "COVID"],
        help="Single domain (default: both)",
    )
    parser.add_argument(
        "--vel-window",
        nargs=2,
        type=int,
        default=list(VEL_WINDOW_DEFAULT),
        metavar=("Y1", "Y2"),
        help=f"Velocity window start/end year (default: {VEL_WINDOW_DEFAULT[0]} {VEL_WINDOW_DEFAULT[1]})",
    )
    args = parser.parse_args()

    y1, y2 = args.vel_window
    if y2 <= y1:
        raise ValueError(f"--vel-window requires y2 > y1, got ({y1}, {y2})")

    tracemalloc.start()
    t_start = time.time()

    domains = [args.domain] if args.domain else ["NLP", "COVID"]
    vel_window = (y1, y2)

    for domain in domains:
        _compute_domain(domain, vel_window, args.force)

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDone.  Total time: {elapsed:.1f}s   Peak memory: {peak / 1024 / 1024:.1f} MiB")


def _compute_domain(domain: str, vel_window: tuple[int, int], force: bool) -> None:
    y1, y2 = vel_window
    vel_path = COMP6_OUT / f"entity_velocity_{domain}.json"
    build_log_path = COMP6_OUT / f"entity_velocity_{domain}_build_log.json"

    if vel_path.exists() and not force:
        print(f"[SKIP] {domain}: velocity cache exists ({vel_path}) — use --force to redo")
        return

    entities = _load_entities(domain)
    fallback = _has_fallback(domain)
    citation_cache = _load_citation_cache(domain) if not fallback else {}

    use_citations = not fallback and bool(citation_cache)
    path_label = "citation" if use_citations else "mention-velocity (fallback)"

    # Build entity×year aggregate matrix
    t0 = time.time()
    entity_yr = _build_entity_year_citations(entities, citation_cache if use_citations else None, use_citations)
    t_agg = time.time() - t0

    # Compute velocities
    t0 = time.time()
    velocities = _compute_velocities(entity_yr, y1, y2)
    t_vel = time.time() - t0

    # Coverage stats
    total_entities = len({e.get("canonical_id", "") for e in entities if e.get("canonical_id")})
    entities_with_vel = len(velocities)
    entities_with_pos_vel = sum(1 for v in velocities.values() if v > 0)
    entities_with_neg_vel = sum(1 for v in velocities.values() if v < 0)

    # Write velocity JSON
    vel_data = {
        "vel_window": list(vel_window),
        "path": path_label,
        "velocities": velocities,
    }
    vel_path.write_text(json.dumps(vel_data, indent=2), encoding="utf-8")
    print(f"  {domain}: wrote {len(velocities)} entity velocities to {vel_path}")

    # Write build log
    log = {
        "domain": domain,
        "vel_window": list(vel_window),
        "citation_path": path_label,
        "fallback_active": fallback,
        "total_entities_in_json": total_entities,
        "entities_with_velocity": entities_with_vel,
        "entities_with_positive_velocity": entities_with_pos_vel,
        "entities_with_negative_velocity": entities_with_neg_vel,
        "entities_with_zero_velocity": total_entities - entities_with_vel,
        "aggregate_build_seconds": round(t_agg, 3),
        "velocity_compute_seconds": round(t_vel, 3),
        "citation_cache_size": len(citation_cache),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    build_log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  {domain}: build log → {build_log_path}")
    print(f"    Path: {path_label}  |  Window: {y1}-{y2}  |  "
          f"Entities w/ vel: {entities_with_vel}/{total_entities}  |  "
          f"Positive: {entities_with_pos_vel}  Negative: {entities_with_neg_vel}")


if __name__ == "__main__":
    main()
