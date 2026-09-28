#!/usr/bin/env python3
"""
Component 5 — Step 1: Build per-year cumulative NetworkX slices.

For each domain (NLP: 2018–2024, COVID: 2019–2024) and each year t in range,
builds a directed nx.DiGraph where an edge (u→c, v→c) exists iff there is ANY
relation record with year <= t.  Nodes are canonical_ids.

Domain isolation: NLP and COVID are processed completely separately.  The 5,099
cross-domain canonical_id overlap is therefore a non-issue at this stage — within
one domain's slice, canonical_id alone is a safe node key.

Input (read-only, from Component 2 — active tree only):
    component2_entity_relation_extraction/output/
    ├── NLP_entities.json        (28,463 records)
    ├── NLP_relations.json       (22,690 records)
    ├── COVID_entities.json      (11,918 records)
    └── COVID_relations.json     (8,081 records)

Output:
    component5/output/slices/{domain}_{year}.gpickle   — one DiGraph per year
    component5/output/slices_build_log.json            — per-slice counts + timing

Runtime: CPU only, no torch/transformers/faiss/node2vec needed.
Seed: numpy.random.seed(42) (script is deterministic — no randomness used).
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import pickle

# ── paths ──────────────────────────────────────────────────────────────────
# Derived from this script's own location so the pipeline runs unchanged on
# any machine (Windows/macOS/Linux) with no hand-editing.
# Layout assumed:
#   <project_root>/component5/<this script>
#   <project_root>/component2_entity_relation_extraction/output
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"
COMP5_ROOT = PROJECT_ROOT / "component5"
COMP5_OUT = COMP5_ROOT / "output"
SLICES_DIR = COMP5_OUT / "slices"

# Expected canonical_id counts (verified on disk, component4/graph_summary.json)
EXPECTED_CANONICAL = {"NLP": 12867, "COVID": 5500}

DOMAIN_SPECS = {
    "NLP": {
        "years": list(range(2018, 2025)),
        "entities": COMP2_OUT / "NLP_entities.json",
        "relations": COMP2_OUT / "NLP_relations.json",
    },
    "COVID": {
        "years": list(range(2019, 2025)),
        "entities": COMP2_OUT / "COVID_entities.json",
        "relations": COMP2_OUT / "COVID_relations.json",
    },
}

DEGENERATE_WARN_NODES = 10   # warn if a slice has fewer than this many nodes


# ── helpers ────────────────────────────────────────────────────────────────

def load_entities(entities_path: Path) -> tuple[dict[str, str], int]:
    """Return (entity_id→canonical_id map, unique canonical_id count)."""
    with open(entities_path, encoding="utf-8") as f:
        records = json.load(f)
    eid_to_canon: dict[str, str] = {}
    canon_set: set[str] = set()
    for r in records:
        eid = r.get("entity_id", "")
        canon = r.get("canonical_id", "")
        if eid and canon:
            eid_to_canon[eid] = canon
        if canon:
            canon_set.add(canon)
    return eid_to_canon, len(canon_set)


def build_slice(relations_path: Path, eid_to_canon: dict[str, str],
                year: int, domain: str) -> nx.DiGraph:
    """Cumulative directed graph for *domain* up to *year* (inclusive).

    Edge attributes: relation_type, year, paper_id, confidence,
                     scierc_relation_type, domain.
    """
    with open(relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    G = nx.DiGraph()
    G.graph["domain"] = domain
    G.graph["cutoff_year"] = year

    nodes_added: set[str] = set()
    edges_added = 0

    for r in relations:
        r_year = r.get("year", 0)
        if r_year > year:
            continue
        src_eid = r.get("source_entity_id", "")
        tgt_eid = r.get("target_entity_id", "")
        if not src_eid or not tgt_eid:
            continue
        src = eid_to_canon.get(src_eid)
        tgt = eid_to_canon.get(tgt_eid)
        if src is None or tgt is None:
            continue
        if src not in nodes_added:
            G.add_node(src)
            nodes_added.add(src)
        if tgt not in nodes_added:
            G.add_node(tgt)
            nodes_added.add(tgt)
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt,
                       relation_type=r.get("relation_type", ""),
                       year=r_year,
                       paper_id=r.get("paper_id", ""),
                       confidence=r.get("confidence", 0.0),
                       scierc_relation_type=r.get("scierc_relation_type", ""),
                       domain=domain)
            edges_added += 1

    return G


def validate_inputs() -> None:
    """Raise FileNotFoundError if any of the 4 input JSONs is missing."""
    for domain, spec in DOMAIN_SPECS.items():
        for key, path in [("entities", spec["entities"]),
                          ("relations", spec["relations"])]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Input missing for {domain} {key}: {path}")
    print("[validate] All 4 Component 2 input JSONs found.")


def validate_canonical_counts() -> None:
    """Log WARNING if on-disk canonical_id count differs from expected."""
    for domain, spec in DOMAIN_SPECS.items():
        _, count = load_entities(spec["entities"])
        expected = EXPECTED_CANONICAL[domain]
        status = "OK" if count == expected else "WARNING"
        print(f"[{status}] {domain}: {count:,} unique canonical_ids "
              f"(expected {expected:,})")
        if count != expected:
            print(f"  → differs from graph_summary.json reference; "
                  f"continuing with on-disk data")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 5 Step 1 — build cumulative per-year slices")
    parser.add_argument("--force", action="store_true",
                        help="redo slices even if gpickle exists")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    np.random.seed(42)

    COMP5_OUT.mkdir(parents=True, exist_ok=True)
    SLICES_DIR.mkdir(parents=True, exist_ok=True)

    validate_inputs()
    validate_canonical_counts()

    build_log: list[dict] = []

    for domain, spec in DOMAIN_SPECS.items():
        entities_path = spec["entities"]
        relations_path = spec["relations"]
        years = spec["years"]

        print(f"\n{'='*60}")
        print(f"  Domain: {domain}  ({years[0]}–{years[-1]})")
        print(f"{'='*60}")

        eid_to_canon, n_canon = load_entities(entities_path)
        print(f"  Entity ID map: {len(eid_to_canon):,} entity_ids → canonical_ids")
        print(f"  Unique canonical_ids: {n_canon:,}")

        for year in years:
            slice_path = SLICES_DIR / f"{domain}_{year}.gpickle"

            if slice_path.exists() and not args.force:
                # Load to log counts even when skipping
                with open(slice_path, "rb") as fh:
                    G = pickle.load(fh)
                build_log.append({
                    "domain": domain, "year": year,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "status": "cached",
                })
                print(f"  [{year}] slice exists, skipping (--force to redo)")
                continue

            t0 = time.time()
            G = build_slice(relations_path, eid_to_canon, year, domain)
            elapsed = time.time() - t0

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            if n_nodes < DEGENERATE_WARN_NODES:
                print(f"  [WARNING] [{year}] only {n_nodes} nodes — "
                      f"fewer than {DEGENERATE_WARN_NODES}; graph may be degenerate")
            elif n_nodes == 0:
                print(f"  [WARNING] [{year}] empty slice — no relations with year ≤ {year}")
            else:
                print(f"  [{year}] nodes={n_nodes:,}  edges={n_edges:,}  "
                      f"time={elapsed:.2f}s")

            with open(slice_path, "wb") as fh:
                pickle.dump(G, fh)

            build_log.append({
                "domain": domain, "year": year,
                "nodes": n_nodes, "edges": n_edges,
                "build_time_s": round(elapsed, 3),
                "status": "built",
            })

    log_path = COMP5_OUT / "slices_build_log.json"
    log_path.write_text(json.dumps(build_log, indent=2))
    print(f"\n  Build log: {log_path}")

    elapsed_total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  Done.  Total time: {elapsed_total:.2f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f} MiB")
    print(f"{'='*60}")

    print("\n  Slice summary:")
    print(f"  {'Domain':<8} {'Year':<6} {'Nodes':>8} {'Edges':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8}")
    for e in build_log:
        print(f"  {e['domain']:<8} {e['year']:<6} "
              f"{e['nodes']:>8,} {e['edges']:>8,}")


if __name__ == "__main__":
    main()
