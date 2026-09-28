#!/usr/bin/env python3
"""
Component 6 — Step 3: Re-rank Component 5 gaps by citation velocity.

For every Component 5 gap file, joins entity velocities and computes:

    priority = fused_score × mean(vel_u, vel_v)

Writes re-ranked gaps to component6/output/reranked/{DOMAIN}_{YEAR}_alpha{ALPHA}.json,
preserving all original fields plus vel_u, vel_v, priority.  Sorted by priority
descending, capped at top-500.

Uses the --min-vel filter: entities with velocity < min_vel get their velocity
clamped to 0, so pairs where both entities have low velocity get priority ≈ 0
and sink to the bottom.

Usage:
    python3 component6/03_rerank.py [--force] [--domain NLP|COVID] [--year Y] [--alpha A] [--min-vel V]
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMP5_OUT = PROJECT_ROOT / "component5" / "output"
COMP5_GAPS = COMP5_OUT / "gaps"

COMP6_OUT = PROJECT_ROOT / "component6" / "output"
RERANKED_DIR = COMP6_OUT / "reranked"

LATEST_YEAR = {"NLP": 2024, "COVID": 2024}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

NLP_YEARS = list(range(2018, 2025))
COVID_YEARS = list(range(2019, 2025))


# ── helpers ──────────────────────────────────────────────────────────────────

def _velocity_path(domain: str) -> Path:
    return COMP6_OUT / f"entity_velocity_{domain}.json"


def _load_velocities(domain: str) -> dict[str, float]:
    """Return {canonical_id: velocity} for the domain."""
    vp = _velocity_path(domain)
    if not vp.exists():
        raise FileNotFoundError(
            f"Entity velocity missing for {domain}: {vp}\n"
            f"Run component6/02_entity_velocity.py first."
        )
    data = json.loads(vp.read_text(encoding="utf-8"))
    return data.get("velocities", {})


def _alpha_str(alpha: float) -> str:
    s = f"{alpha:.2f}".rstrip("0").rstrip(".")
    return s


def _gaps_path(domain: str, year: int, alpha: float) -> Path:
    return COMP5_GAPS / f"{domain}_{year}_alpha{_alpha_str(alpha)}.json"


def _reranked_path(domain: str, year: int, alpha: float) -> Path:
    return RERANKED_DIR / f"{domain}_{year}_alpha{_alpha_str(alpha)}.json"


def _load_gaps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 6 Step 3 — re-rank gaps by citation velocity"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo even if re-ranked file exists",
    )
    parser.add_argument(
        "--domain",
        choices=["NLP", "COVID"],
        help="Single domain (default: both)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year (default: all for domain)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Single alpha (default: all ALPHAS)",
    )
    parser.add_argument(
        "--min-vel",
        type=float,
        default=0.0,
        help="Minimum velocity threshold (default: 0.0).  Entities below this "
             "get velocity clamped to 0.",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()

    domains = [args.domain] if args.domain else ["NLP", "COVID"]
    years_map = {"NLP": NLP_YEARS, "COVID": COVID_YEARS}
    alphas = [args.alpha] if args.alpha is not None else ALPHAS

    for domain in domains:
        years = [args.year] if args.year else years_map[domain]
        _rerank_domain(domain, years, alphas, args.min_vel, args.force)

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDone.  Total time: {elapsed:.1f}s   Peak memory: {peak / 1024 / 1024:.1f} MiB")


def _rerank_domain(
    domain: str,
    years: list[int],
    alphas: list[float],
    min_vel: float,
    force: bool,
) -> None:
    RERANKED_DIR.mkdir(parents=True, exist_ok=True)

    velocities = _load_velocities(domain)
    print(f"\n  Domain: {domain}  |  "
          f"velocities loaded: {len(velocities)}  |  "
          f"years: {years}  |  alphas: {alphas}  |  "
          f"min_vel: {min_vel}")

    stats = {"files_processed": 0, "files_skipped": 0,
             "total_gaps_read": 0, "total_gaps_written": 0,
             "vel_u_missing": 0, "vel_v_missing": 0}

    for year in years:
        for alpha in alphas:
            src = _gaps_path(domain, year, alpha)
            dst = _reranked_path(domain, year, alpha)

            if not src.exists():
                stats["files_skipped"] += 1
                print(f"    [SKIP] {src.name}: source gaps file missing")
                continue

            if dst.exists() and not force:
                stats["files_skipped"] += 1
                print(f"    [SKIP] {dst.name}: already re-ranked (--force to redo)")
                continue

            gaps = _load_gaps(src)
            if not gaps:
                stats["files_skipped"] += 1
                print(f"    [SKIP] {dst.name}: empty gaps file")
                continue

            stats["total_gaps_read"] += len(gaps)

            # Re-rank
            t0 = time.time()
            reranked, vu_missing, vv_missing = _rerank_gaps(gaps, velocities, min_vel)
            stats["total_gaps_written"] += len(reranked)
            stats["vel_u_missing"] += vu_missing
            stats["vel_v_missing"] += vv_missing

            dst.write_text(json.dumps(reranked, indent=2), encoding="utf-8")
            stats["files_processed"] += 1
            dt = time.time() - t0

            # Summary line
            if reranked:
                t0 = reranked[0]
                print(f"    {dst.name}: {len(gaps)}→{len(reranked)} gaps, "
                      f"top priority={t0.get('priority', 0):.4f} "
                      f"({str(t0.get('entity_a', '?'))[:20]}… × {str(t0.get('entity_b', '?'))[:20]}…), "
                      f"time={dt:.2f}s")
            else:
                print(f"    {dst.name}: {len(gaps)}→0 gaps (all filtered), time={dt:.2f}s")

    # Print stats
    print(f"  {domain} re-rank summary: processed={stats['files_processed']}, "
          f"skipped={stats['files_skipped']}, "
          f"gaps_read={stats['total_gaps_read']}, gaps_written={stats['total_gaps_written']}")


def _rerank_gaps(gaps: list[dict], velocities: dict[str, float], min_vel: float) -> tuple[list[dict], int, int]:
    """
    priority = fused_score × mean(vel_u, vel_v)

    vel_u, vel_v are the raw velocities (can be negative).
    If either entity is missing from the velocity dict, its velocity is 0.
    min_vel filter: velocities below min_vel are clamped to 0 (so pairs of
    low-velocity entities sink to the bottom).
    """
    results: list[dict] = []
    vel_u_missing = 0
    vel_v_missing = 0

    for g in gaps:
        u = g.get("entity_a", "")
        v = g.get("entity_b", "")
        fused = g.get("fused_score", 0.0)

        if not u or not v or fused <= 0:
            continue

        vel_u = velocities.get(u, 0.0)
        vel_v = velocities.get(v, 0.0)

        if u not in velocities:
            vel_u_missing += 1
        if v not in velocities:
            vel_v_missing += 1

        # min_vel clamp
        if vel_u < min_vel:
            vel_u = 0.0
        if vel_v < min_vel:
            vel_v = 0.0

        vel_mean = (vel_u + vel_v) / 2.0
        priority = fused * vel_mean

        entry = dict(g)  # preserve all original fields
        entry["vel_u"] = vel_u
        entry["vel_v"] = vel_v
        entry["priority"] = round(priority, 6)
        results.append(entry)

    # Sort by priority descending, cap at 500
    results.sort(key=lambda r: r["priority"], reverse=True)
    top = results[:500]

    return top, vel_u_missing, vel_v_missing


if __name__ == "__main__":
    main()
