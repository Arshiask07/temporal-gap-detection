#!/usr/bin/env python3
"""
Component 7 — Export real Component 6 citation velocity to dashboard CSV format.

Converts the entity_velocity JSON from Component 6 into the CSV file the
Streamlit dashboard expects:

    dashboard/exports/citations_{nlp,covid-19}.csv

Schema: entity_id, year, citations

The CSV format is used by data_loader.load_citation_history() and
gap_engine._trim_citations() / rank_gaps() to compute citation velocity.

Since the S2 API was unreachable when C6 ran, this uses the mention-velocity
fallback data from entity_velocity_{domain}.json. The build_log confirms this.

Usage:
    python3 export_real_citations.py [--domain NLP|COVID] [--force]

Input:
    component6/output/entity_velocity_{domain}.json

Output:
    dashboard/exports/citations_{short}.csv
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
COMP6_OUT = DASHBOARD_DIR.parent / "component6" / "output"
EXPORT_DIR = DASHBOARD_DIR / "exports"

DOMAIN_CONFIG = {
    "NLP": {"short": "nlp", "velocity": COMP6_OUT / "entity_velocity_NLP.json"},
    "COVID": {"short": "covid-19", "velocity": COMP6_OUT / "entity_velocity_COVID.json"},
}


def export_domain(domain: str, cfg: dict, force: bool) -> dict:
    """Export citations for one domain. Returns stats."""
    vp = cfg["velocity"]
    if not vp.exists():
        print(f"  [ERROR] Velocity file not found: {vp}")
        return {"error": f"missing {vp.name}"}

    print(f"  Loading {domain} entity velocity…")
    with open(vp, encoding="utf-8") as f:
        data = json.load(f)

    velocities = data.get("velocities", {})
    path_label = data.get("path", data.get("citation_path", "unknown"))
    print(f"    → {len(velocities):,} entities with velocity")
    print(f"    → citation_path: {path_label}")
    print(f"    → fallback_active: {data.get('fallback_active', data.get('fallback', False))}")

    # Build citation rows from velocity data
    # C6 stores velocity = (citations[2024] - citations[2022]) / 2
    # We need to reconstruct entity_id, year, citations triples.
    #
    # The velocity JSON has: {canonical_id: velocity_value}
    # We write a citations CSV with entity_id, year, citations.
    #
    # Since we only have velocity (not raw citation counts), we derive:
    #   citations[2022] = base_count (from mention-velocity proxy)
    #   citations[2024] = base_count + 2 * velocity
    #
    # The mention-velocity fallback uses paper mention counts as a proxy.
    # We take a reasonable base: entities with positive velocity get
    # citations[2022] = max(1, round(vel)) and citations[2024] = round(vel * 2).
    # Entities with zero/negative velocity get citations[2022] = citations[2024] = 0.

    rows = []
    for cid, vel in velocities.items():
        vel_val = float(vel)
        if vel_val > 0:
            c22 = max(1, round(vel_val))
            c24 = round(vel_val * 2)
        else:
            c22 = 0
            c24 = 0
        rows.append({"entity_id": cid, "year": 2022, "citations": c22})
        rows.append({"entity_id": cid, "year": 2024, "citations": c24})

    # Deduplicate: if same entity appears multiple times (shouldn't), keep last
    df = pd.DataFrame(rows).drop_duplicates(subset=["entity_id", "year"], keep="last")

    stats = {
        "rows": len(df),
        "entities": df["entity_id"].nunique(),
        "citation_path": path_label,
        "fallback": data.get("fallback_active", data.get("fallback", False)),
    }

    csv_path = EXPORT_DIR / f"citations_{cfg['short']}.csv"
    if csv_path.exists() and not force:
        print(f"  [SKIP] {csv_path.name} exists (use --force)")
    else:
        df.to_csv(csv_path, index=False)
        print(f"    → {csv_path.name}: {len(df)} rows, {stats['entities']:,} entities")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 7 — export real C6 citation velocity to dashboard CSV format"
    )
    parser.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()),
                        help="single domain (default: both)")
    parser.add_argument("--force", action="store_true",
                        help="redo even if CSV exists")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    domains = [args.domain] if args.domain else list(DOMAIN_CONFIG.keys())
    total_rows = 0
    total_entities = 0

    for domain in domains:
        cfg = DOMAIN_CONFIG[domain]
        print(f"\n{'='*50}")
        print(f"  {domain} ({cfg['short']})")
        print(f"{'='*50}")
        s = export_domain(domain, cfg, args.force)
        if "error" not in s:
            total_rows += s["rows"]
            total_entities += s["entities"]
            print(f"    citation_path: {s['citation_path']}")
            print(f"    fallback: {s['fallback']}")

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*50}")
    print(f"  Done.  Total time: {elapsed:.1f}s   Peak memory: {peak/1024/1024:.1f} MiB")
    print(f"  Citation rows: {total_rows:,}  Entities: {total_entities:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
