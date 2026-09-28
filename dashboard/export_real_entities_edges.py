#!/usr/bin/env python3
"""
Component 7 — Export real Component 2 entities + relations to dashboard CSV format.

Converts the JSON outputs from Component 2 into the CSV files the Streamlit
dashboard expects:

    dashboard/exports/entities_{nlp,covid-19}.csv
    dashboard/exports/edges_{nlp,covid-19}.csv

entities CSV schema: entity_id, label, type, first_year
    → entity_id = canonical_id
    → label = most frequent surface_form for that canonical_id
    → type = most common entity type for that canonical_id
    → first_year = earliest year this canonical_id appears

edges CSV schema: source, relation, target, first_observed
    → source/target = canonical_ids (remapped from entity_ids)
    → relation = relation_type from C2
    → first_observed = year from C2

Usage:
    python3 export_real_entities_edges.py [--domain NLP|COVID] [--force]

Input:
    component2_entity_relation_extraction/output/{domain}_entities.json
    component2_entity_relation_extraction/output/{domain}_relations.json

Output:
    dashboard/exports/entities_{short}.csv
    dashboard/exports/edges_{short}.csv
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
COMP2_OUT = DASHBOARD_DIR.parent / "component2_entity_relation_extraction" / "output"
EXPORT_DIR = DASHBOARD_DIR / "exports"

DOMAIN_CONFIG = {
    "NLP": {"short": "nlp", "entities": COMP2_OUT / "NLP_entities.json",
             "relations": COMP2_OUT / "NLP_relations.json"},
    "COVID": {"short": "covid-19", "entities": COMP2_OUT / "COVID_entities.json",
              "relations": COMP2_OUT / "COVID_relations.json"},
}


def entity_id_to_canon(entities: list[dict]) -> dict[str, str]:
    """Return {entity_id: canonical_id} lookup from entities JSON."""
    mapping = {}
    for e in entities:
        eid = e.get("entity_id", "")
        cid = e.get("canonical_id", "")
        if eid and cid:
            mapping[eid] = cid
    return mapping


def aggregate_entities(entities: list[dict]) -> list[dict]:
    """Aggregate per canonical_id: most common type, most frequent surface_form, first_year."""
    # Group by canonical_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        cid = e.get("canonical_id", "")
        if cid:
            groups[cid].append(e)

    result = []
    for cid, records in groups.items():
        # Most common type
        type_counts = Counter(r.get("type", "Other") for r in records)
        best_type = type_counts.most_common(1)[0][0]

        # Most frequent surface_form (tie-break: first occurrence)
        surf_counts: dict[str, int] = {}
        surf_order: dict[str, int] = {}
        order = 0
        for r in records:
            sf = r.get("surface_form", "")
            if sf:
                surf_counts[sf] = surf_counts.get(sf, 0) + 1
                if sf not in surf_order:
                    surf_order[sf] = order
                    order += 1
        if surf_counts:
            best_sf = max(surf_counts.keys(),
                          key=lambda s: (surf_counts[s], -surf_order[s]))
        else:
            best_sf = cid

        # First year
        years = [r.get("year", 0) for r in records if r.get("year")]
        first_year = min(years) if years else 0

        result.append({
            "entity_id": cid,
            "label": best_sf,
            "type": best_type,
            "first_year": first_year,
        })

    result.sort(key=lambda r: r["entity_id"])
    return result


def build_edges(relations: list[dict], eid_to_canon: dict[str, str]) -> list[dict]:
    """Remap entity_ids to canonical_ids, build edge rows."""
    rows = []
    unresolved = 0

    for r in relations:
        src_raw = r.get("source_entity_id", "")
        tgt_raw = r.get("target_entity_id", "")
        src = eid_to_canon.get(src_raw, "")
        tgt = eid_to_canon.get(tgt_raw, "")

        if not src or not tgt:
            unresolved += 1
            continue

        rows.append({
            "source": src,
            "relation": r.get("relation_type", ""),
            "target": tgt,
            "first_observed": r.get("year", 0),
        })

    if unresolved:
        print(f"    [warn] {unresolved} relations could not be remapped (skipped)")
    return rows


def export_domain(domain: str, cfg: dict, force: bool) -> dict:
    """Export entities + edges for one domain. Returns stats."""
    stats = {}

    # Load entities
    print(f"  Loading {domain} entities…")
    with open(cfg["entities"], encoding="utf-8") as f:
        entities = json.load(f)
    print(f"    → {len(entities):,} entity records")

    # Load relations
    print(f"  Loading {domain} relations…")
    with open(cfg["relations"], encoding="utf-8") as f:
        relations = json.load(f)
    print(f"    → {len(relations):,} relation records")

    # Build entity_id → canonical_id lookup
    eid_to_canon = entity_id_to_canon(entities)
    print(f"    → {len(eid_to_canon):,} entity_id → canonical_id mappings")

    # Aggregate entities
    agg_entities = aggregate_entities(entities)
    stats["entities"] = len(agg_entities)

    # Write entities CSV
    ent_path = EXPORT_DIR / f"entities_{cfg['short']}.csv"
    if ent_path.exists() and not force:
        print(f"  [SKIP] {ent_path.name} exists (use --force)")
    else:
        pd.DataFrame(agg_entities).to_csv(ent_path, index=False)
        print(f"    → {ent_path.name}: {len(agg_entities)} entities")
    stats["entities_file"] = ent_path.name

    # Build edges
    edges = build_edges(relations, eid_to_canon)
    stats["edges"] = len(edges)
    stats["unresolved"] = sum(1 for e in edges if not e["source"] or not e["target"])

    # Write edges CSV
    edge_path = EXPORT_DIR / f"edges_{cfg['short']}.csv"
    if edge_path.exists() and not force:
        print(f"  [SKIP] {edge_path.name} exists (use --force)")
    else:
        pd.DataFrame(edges).to_csv(edge_path, index=False)
        print(f"    → {edge_path.name}: {len(edges)} edges")
    stats["edges_file"] = edge_path.name

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 7 — export real C2 entities/relations to dashboard CSV format"
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
    total = {"entities": 0, "edges": 0, "unresolved": 0}

    for domain in domains:
        cfg = DOMAIN_CONFIG[domain]
        print(f"\n{'='*50}")
        print(f"  {domain} ({cfg['short']})")
        print(f"{'='*50}")
        s = export_domain(domain, cfg, args.force)
        for k in total:
            total[k] += s.get(k, 0)

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*50}")
    print(f"  Done.  Total time: {elapsed:.1f}s   Peak memory: {peak/1024/1024:.1f} MiB")
    print(f"  Entities: {total['entities']:,}  Edges: {total['edges']:,}  Unresolved: {total['unresolved']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
