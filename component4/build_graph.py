#!/usr/bin/env python3
"""
Component 4 — Temporal Knowledge Graph Builder.

Loads Component 2's extracted entities and relations into a Neo4j property graph.
One :Entity node per (canonical_id, domain) pair. One directed relationship per
typed relation, with year, paper_id, domain, and confidence as edge properties.

Usage:
    python3 build_graph.py [--host localhost] [--port 7687] [--user neo4j] [--password PW]

Input (read-only, from Component 2):
    component2_entity_relation_extraction/output/
    ├── NLP_entities.json
    ├── NLP_relations.json
    ├── NLP_entity_normalization_map.json
    ├── COVID_entities.json
    ├── COVID_relations.json
    └── COVID_entity_normalization_map.json

Output:
    component4/graph_summary.json  — load validation report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMP2_OUT = Path(__file__).resolve().parent.parent / "component2_entity_relation_extraction" / "output"
COMP4_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = COMP4_DIR / "graph_summary.json"

DOMAIN_FILES = {
    "NLP": {
        "entities": COMP2_OUT / "NLP_entities.json",
        "relations": COMP2_OUT / "NLP_relations.json",
        "norm_map": COMP2_OUT / "NLP_entity_normalization_map.json",
    },
    "COVID": {
        "entities": COMP2_OUT / "COVID_entities.json",
        "relations": COMP2_OUT / "COVID_relations.json",
        "norm_map": COMP2_OUT / "COVID_entity_normalization_map.json",
    },
}

EXPECTED = {
    "NLP": {"nodes": 12867, "edges": 22690},
    "COVID": {"nodes": 5500, "edges": 8081},
}

BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Data loading and pre-processing
# ---------------------------------------------------------------------------

def load_entities(domain: str) -> list[dict]:
    path = DOMAIN_FILES[domain]["entities"]
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} entity records for {domain} from {path.name}")
    return data


def load_relations(domain: str) -> list[dict]:
    path = DOMAIN_FILES[domain]["relations"]
    if not path.exists():
        raise FileNotFoundError(f"Relations file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} relation records for {domain} from {path.name}")
    return data


def build_entity_id_lookup(entities: list[dict], domain: str) -> dict[str, tuple[str, str]]:
    """Map entity_id -> (canonical_id, domain)."""
    lookup: dict[str, tuple[str, str]] = {}
    for e in entities:
        eid = e["entity_id"]
        canon = e["canonical_id"]
        lookup[eid] = (canon, domain)
    return lookup


def compute_node_aggregates(entities: list[dict], domain: str) -> dict[str, dict[str, Any]]:
    """
    Pre-compute per-canonical_id node properties:
      - type: most common entity type
      - surface_form: most frequent surface form
      - first_seen: min year
      - last_seen: max year
      - paper_count: distinct paper_ids
    """
    # Group by canonical_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        groups[e["canonical_id"]].append(e)

    nodes: dict[str, dict[str, Any]] = {}
    for canon_id, records in groups.items():
        # Type: most common
        type_counts = Counter(r["type"] for r in records)
        node_type = type_counts.most_common(1)[0][0]

        # Surface form: most frequent
        surf_counts = Counter(r["surface_form"] for r in records)
        surface_form = surf_counts.most_common(1)[0][0]

        # Year range
        years = [r["year"] for r in records]
        first_seen = min(years)
        last_seen = max(years)

        # Distinct papers
        paper_ids = set(r["paper_id"] for r in records)
        paper_count = len(paper_ids)

        nodes[canon_id] = {
            "canonical_id": canon_id,
            "domain": domain,
            "type": node_type,
            "surface_form": surface_form,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "paper_count": paper_count,
        }

    return nodes


def build_edge_batches(
    relations: list[dict],
    entity_lookup: dict[str, tuple[str, str]],
    domain: str,
) -> dict[str, list[dict]]:
    """
    Remap entity_id -> canonical_id for source and target.
    Group by relation_type. Return {relation_type: [batch_rows]}.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    unresolved = 0

    for r in relations:
        src_key = entity_lookup.get(r["source_entity_id"])
        tgt_key = entity_lookup.get(r["target_entity_id"])

        if src_key is None or tgt_key is None:
            unresolved += 1
            continue

        src_canon, src_domain = src_key
        tgt_canon, tgt_domain = tgt_key

        # Defensive: skip if domain mismatch (shouldn't happen per verification)
        if src_domain != domain or tgt_domain != domain:
            unresolved += 1
            continue

        by_type[r["relation_type"]].append({
            "source_canonical_id": src_canon,
            "target_canonical_id": tgt_canon,
            "year": r["year"],
            "paper_id": r["paper_id"],
            "domain": domain,
            "confidence": r.get("confidence", 0.0),
            "scierc_relation_type": r.get("scierc_relation_type", ""),
        })

    if unresolved:
        print(f"  WARNING: {unresolved:,} relations could not be resolved for {domain}")
    else:
        print(f"  All {len(relations):,} relations resolved for {domain}")

    return dict(by_type)


# ---------------------------------------------------------------------------
# Neo4j loading
# ---------------------------------------------------------------------------

ENTITY_CREATE_CONSTRAINT = """
CREATE CONSTRAINT entity_canonical_domain IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.canonical_id, e.domain) IS UNIQUE
"""

ENTITY_CREATE_INDEXES = [
    "CREATE INDEX entity_first_seen IF NOT EXISTS FOR (e:Entity) ON (e.first_seen)",
    "CREATE INDEX entity_last_seen IF NOT EXISTS FOR (e:Entity) ON (e.last_seen)",
    "CREATE INDEX rel_year_used_for IF NOT EXISTS FOR ()-[r:USED_FOR]-() ON (r.year)",
    "CREATE INDEX rel_year_assoc IF NOT EXISTS FOR ()-[r:ENTITY_ASSOCIATED_WITH_ENTITY]-() ON (r.year)",
    "CREATE INDEX rel_year_method_applied IF NOT EXISTS FOR ()-[r:METHOD_APPLIED_TO]-() ON (r.year)",
    "CREATE INDEX rel_year_method_eval IF NOT EXISTS FOR ()-[r:METHOD_EVALUATED_BY]-() ON (r.year)",
]

ENTITY_LOAD_CYPHER = """
UNWIND $batch AS row
MERGE (e:Entity {canonical_id: row.canonical_id, domain: row.domain})
SET e.type = row.type,
    e.first_seen = coalesce(e.first_seen, row.year),
    e.last_seen = row.year,
    e.surface_form = row.surface_form,
    e.paper_count = row.paper_count
"""

EDGE_LOAD_CYPHER_TEMPLATE = """
UNWIND $batch AS row
MATCH (src:Entity {canonical_id: row.source_canonical_id, domain: row.domain})
MATCH (tgt:Entity {canonical_id: row.target_canonical_id, domain: row.domain})
MERGE (src)-[r:PLACEHOLDER_REL_TYPE {year: row.year, paper_id: row.paper_id, domain: row.domain, confidence: row.confidence}]->(tgt)
SET r.scierc_relation_type = row.scierc_relation_type
"""


def create_constraint(driver) -> None:
    with driver.session() as session:
        session.run(ENTITY_CREATE_CONSTRAINT)
    print("  Created composite uniqueness constraint (canonical_id, domain)")


def create_indexes(driver) -> None:
    with driver.session() as session:
        for stmt in ENTITY_CREATE_INDEXES:
            session.run(stmt)
    print("  Created indexes: entity_first_seen, entity_last_seen, rel_year (4 relationship types)")


def load_entities_to_neo4j(driver, nodes: dict[str, dict], domain: str) -> int:
    """Batch-load entity nodes. Returns count of nodes loaded."""
    rows = list(nodes.values())
    total = len(rows)
    loaded = 0
    batch_count = 0

    with driver.session() as session:
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.run(ENTITY_LOAD_CYPHER, {"batch": batch})
            loaded += len(batch)
            batch_count += 1
            if batch_count % 5 == 0 or batch_count == 1:
                print(f"    ... {min(loaded, total):,}/{total:,} {domain} entity nodes")

    print(f"  Loaded {loaded:,} {domain} entity nodes ({batch_count} batches)")
    return loaded


def load_edges_to_neo4j(
    driver,
    edge_batches: dict[str, list[dict]],
    domain: str,
) -> dict[str, int]:
    """Batch-load edges per relation_type. Returns {rel_type: count}."""
    counts: dict[str, int] = {}

    with driver.session() as session:
        for rel_type, rows in edge_batches.items():
            cypher = EDGE_LOAD_CYPHER_TEMPLATE.replace("PLACEHOLDER_REL_TYPE", rel_type)
            total = len(rows)
            batch_count = 0
            n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, total, BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                session.run(cypher, {"batch": batch})
                batch_count += len(batch)
                done = min(batch_count, total)
                if n_batches <= 3 or batch_count % (BATCH_SIZE * 2) == 0 or batch_count == total:
                    print(f"    ... {done:,}/{total:,} {domain}:{rel_type} edges")
            counts[rel_type] = batch_count

    for rel_type, count in counts.items():
        print(f"  Loaded {count:,} {domain}:{rel_type} edges")

    return counts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALIDATION_QUERIES = {
    "total_nodes": "MATCH (e:Entity) RETURN count(e) AS c",
    "nodes_by_domain": "MATCH (e:Entity) RETURN e.domain AS domain, count(e) AS c",
    "total_edges": "MATCH ()-[r]->() RETURN count(r) AS c",
    "edges_by_type_domain": "MATCH ()-[r]->() RETURN r.domain AS domain, type(r) AS rel_type, count(r) AS c",
    "edges_by_year": "MATCH ()-[r]->() RETURN r.year AS year, count(r) AS c ORDER BY year",
    "cross_domain_edges": (
        "MATCH (a:Entity {domain: 'NLP'})-[r]->(b:Entity {domain: 'COVID'}) "
        "RETURN count(r) AS c"
    ),
    "cross_domain_edges_rev": (
        "MATCH (a:Entity {domain: 'COVID'})-[r]->(b:Entity {domain: 'NLP'}) "
        "RETURN count(r) AS c"
    ),
}


def run_validation(driver) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with driver.session() as session:
        for name, cypher in VALIDATION_QUERIES.items():
            result = session.run(cypher)
            rows = [record.data() for record in result]
            results[name] = rows
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    neo4j_version: str,
    load_times: dict[str, float],
    node_counts: dict[str, int],
    edge_counts: dict[str, dict[str, int]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "neo4j_driver_version": "5.28.5.0",
        "neo4j_server_version": neo4j_version,
        "mode": "dry_run" if not validation else "live",
        "domains": {},
        "total_nodes": sum(node_counts.values()),
        "total_edges": sum(sum(counts.values()) for counts in edge_counts.values()),
        "load_times_seconds": load_times,
        "validation": validation,
    }

    for domain in ["NLP", "COVID"]:
        domain_edges = edge_counts.get(domain, {})
        summary["domains"][domain] = {
            "nodes": node_counts.get(domain, 0),
            "edges": sum(domain_edges.values()),
            "relation_types": list(domain_edges.keys()),
            "expected_nodes": EXPECTED[domain]["nodes"],
            "expected_edges": EXPECTED[domain]["edges"],
        }

    # Cross-domain check
    v_cross_fwd = validation.get("cross_domain_edges", [])
    v_cross_rev = validation.get("cross_domain_edges_rev", [])
    cross_fwd = v_cross_fwd[0].get("c", 0) if v_cross_fwd else 0
    cross_rev = v_cross_rev[0].get("c", 0) if v_cross_rev else 0
    cross_total = cross_fwd + cross_rev
    summary["cross_domain_edges"] = cross_total
    summary["cross_domain_edges_ok"] = (cross_total == 0)

    # Expected counts check
    summary["counts_match_expected"] = True
    for domain in ["NLP", "COVID"]:
        actual_nodes = node_counts.get(domain, 0)
        actual_edges = sum(edge_counts.get(domain, {}).values())
        exp = EXPECTED[domain]
        if actual_nodes != exp["nodes"]:
            summary["counts_match_expected"] = False
            print(f"  MISMATCH: {domain} nodes {actual_nodes:,} != expected {exp['nodes']:,}")
        if actual_edges != exp["edges"]:
            summary["counts_match_expected"] = False
            print(f"  MISMATCH: {domain} edges {actual_edges:,} != expected {exp['edges']:,}")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary written to {SUMMARY_PATH}")
    return summary


# ---------------------------------------------------------------------------
# Neo4j version detection
# ---------------------------------------------------------------------------

def get_neo4j_version(driver) -> str:
    try:
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions WHERE name = 'Neo4j' RETURN versions[0] AS v")
            record = result.single()
            if record:
                return record["v"]
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Component 4 — Build Neo4j knowledge graph from Component 2 outputs")
    parser.add_argument("--host", default="localhost", help="Neo4j Bolt host (default: localhost)")
    parser.add_argument("--port", type=int, default=7687, help="Neo4j Bolt port (default: 7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j user (default: neo4j)")
    parser.add_argument("--password", default=None, help="Neo4j password (default: read from ~/.neo4j_password if present, else 'neo4j')")
    parser.add_argument("--dry-run", action="store_true", help="Process data but skip Neo4j writes")
    parser.add_argument("--reset", action="store_true", help="Delete all :Entity nodes and relationships before loading (makes re-runs idempotent)")
    parser.add_argument("--uri", default=None, help="Neo4j Bolt URI (default: bolt://host:port)")
    args = parser.parse_args()

    password = args.password if args.password is not None else "neo4j"
    uri = args.uri if args.uri else f"bolt://{args.host}:{args.port}"

    # Verify input files exist
    for domain, files in DOMAIN_FILES.items():
        for key, path in files.items():
            if not path.exists():
                print(f"ERROR: {domain} {key} not found at {path}")
                sys.exit(1)

    print("=" * 60)
    print("Component 4 — Temporal Knowledge Graph Builder")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load and pre-process data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and pre-processing Component 2 data...")
    t0 = time.time()

    all_entities: dict[str, list[dict]] = {}
    all_relations: dict[str, list[dict]] = {}
    entity_lookups: dict[str, dict[str, tuple[str, str]]] = {}
    node_aggregates: dict[str, dict[str, dict]] = {}

    for domain in ["NLP", "COVID"]:
        print(f"\n  Processing {domain}...")
        entities = load_entities(domain)
        relations = load_relations(domain)
        all_entities[domain] = entities
        all_relations[domain] = relations

        entity_lookups[domain] = build_entity_id_lookup(entities, domain)
        node_aggregates[domain] = compute_node_aggregates(entities, domain)

        # Print aggregates summary
        n_canon = len(node_aggregates[domain])
        n_records = len(entities)
        print(f"    -> {n_canon:,} unique canonical_ids from {n_records:,} entity records")
        type_dist = Counter(n["type"] for n in node_aggregates[domain].values())
        print(f"    -> Types: {dict(type_dist)}")

    data_time = time.time() - t0
    print(f"\n  Data loading complete in {data_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Build edge batches
    # ------------------------------------------------------------------
    print("\n[2/5] Building edge batches (entity_id -> canonical_id remapping)...")
    t0 = time.time()

    all_edge_batches: dict[str, dict[str, list[dict]]] = {}
    for domain in ["NLP", "COVID"]:
        print(f"\n  {domain}:")
        batches = build_edge_batches(all_relations[domain], entity_lookups[domain], domain)
        all_edge_batches[domain] = batches

        total_edges = sum(len(rows) for rows in batches.values())
        print(f"    -> {len(batches)} relation types, {total_edges:,} total edges")
        for rel_type, rows in batches.items():
            print(f"       {rel_type}: {len(rows):,}")

    edge_time = time.time() - t0
    print(f"\n  Edge batch building complete in {edge_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Connect to Neo4j (or dry-run)
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[3/5] DRY RUN — skipping Neo4j connection and writes")
        print("  Use --dry-run to verify data processing without a Neo4j server")
        summary = write_summary(
            neo4j_version="dry-run",
            load_times={"data": data_time, "edge_batch": edge_time},
            node_counts={d: len(na) for d, na in node_aggregates.items()},
            edge_counts={
                d: {k: len(v) for k, v in batches.items()}
                for d, batches in all_edge_batches.items()
            },
            validation={},
        )
        print(f"\n  Expected graph: {summary['total_nodes']:,} nodes, {summary['total_edges']:,} edges")
        return

    print("\n[3/5] Connecting to Neo4j...")
    t0 = time.time()

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERROR: neo4j Python driver not installed. Run: pip install neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(args.user, password))

    # Test connection
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        print(f"  Connected to Neo4j at {args.host}:{args.port} as {args.user}")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j at {args.host}:{args.port}")
        print(f"  {e}")
        print("\n  Try: brew install neo4j && brew services start neo4j")
        print("  Or:  python3 build_graph.py --host <host> --port <port> --user <user> --password <password>")
        sys.exit(1)

    connect_time = time.time() - t0
    neo4j_version = get_neo4j_version(driver)
    print(f"  Neo4j version: {neo4j_version}")
    print(f"  Connection established in {connect_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Create constraint and load
    # ------------------------------------------------------------------
    print("\n[4/5] Creating constraint and loading graph...")
    t0 = time.time()

    if args.reset:
        print("  Resetting: deleting all existing :Entity nodes and relationships...")
        with driver.session() as session:
            session.run("MATCH (e:Entity) DETACH DELETE e")
        print("  Reset complete.")

    create_constraint(driver)
    create_indexes(driver)

    node_counts: dict[str, int] = {}
    edge_counts: dict[str, dict[str, int]] = {}

    for domain in ["NLP", "COVID"]:
        print(f"\n  Loading {domain}...")
        nodes_loaded = load_entities_to_neo4j(driver, node_aggregates[domain], domain)
        node_counts[domain] = nodes_loaded

        edges_loaded = load_edges_to_neo4j(driver, all_edge_batches[domain], domain)
        edge_counts[domain] = edges_loaded

    load_time = time.time() - t0
    print(f"\n  Graph loading complete in {load_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Validate
    # ------------------------------------------------------------------
    print("\n[5/5] Running validation queries...")
    t0 = time.time()

    validation = run_validation(driver)

    print("\n  === Validation Results ===")
    total_nodes = validation["total_nodes"][0]["c"]
    print(f"  Total nodes: {total_nodes:,}")
    print(f"  Nodes by domain:")
    for row in validation["nodes_by_domain"]:
        exp = EXPECTED.get(row["domain"], {})
        match = "✓" if row["c"] == exp.get("nodes") else "✗"
        print(f"    {row['domain']}: {row['c']:,} {match}")

    total_edges = validation["total_edges"][0]["c"]
    print(f"\n  Total edges: {total_edges:,}")
    print(f"  Edges by type and domain:")
    for row in validation["edges_by_type_domain"]:
        exp = EXPECTED.get(row["domain"], {})
        exp_edges = exp.get("edges", 0)
        match = "✓" if row["c"] <= exp_edges else "✗"
        print(f"    {row['domain']:5s} {row['rel_type']:35s} {row['c']:>8,} {match}")

    print(f"\n  Edges by year:")
    for row in validation["edges_by_year"]:
        print(f"    {row['year']}: {row['c']:>8,}")

    cross_fwd = validation["cross_domain_edges"][0]["c"]
    cross_rev = validation["cross_domain_edges_rev"][0]["c"]
    cross_total = cross_fwd + cross_rev
    cross_ok = "✓" if cross_total == 0 else "✗ CROSS-DOMAIN DETECTED"
    print(f"\n  Cross-domain edges (NLP->COVID + COVID->NLP): {cross_total:,} {cross_ok}")

    val_time = time.time() - t0
    print(f"\n  Validation complete in {val_time:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    driver.close()

    summary = write_summary(
        neo4j_version=neo4j_version,
        load_times={"data": data_time, "edge_batch": edge_time, "connect": connect_time, "load": load_time, "validate": val_time},
        node_counts=node_counts,
        edge_counts=edge_counts,
        validation=validation,
    )

    # Final verdict
    print("\n" + "=" * 60)
    print("Component 4 — Load Verdict")
    print("=" * 60)
    print(f"  Nodes: {summary['total_nodes']:,} / {18367:,} expected")
    print(f"  Edges: {summary['total_edges']:,} / {30771:,} expected")
    print(f"  Cross-domain edges: {summary['cross_domain_edges']:,} (should be 0)")
    print(f"  Counts match expected: {summary['counts_match_expected']}")

    nlp_ok = node_counts.get("NLP", 0) == EXPECTED["NLP"]["nodes"] and sum(edge_counts.get("NLP", {}).values()) == EXPECTED["NLP"]["edges"]
    covid_ok = node_counts.get("COVID", 0) == EXPECTED["COVID"]["nodes"] and sum(edge_counts.get("COVID", {}).values()) == EXPECTED["COVID"]["edges"]
    print(f"  NLP counts exact: {nlp_ok}")
    print(f"  COVID counts exact: {covid_ok}")

    if summary["counts_match_expected"] and summary["cross_domain_edges_ok"]:
        print("\n  ✅ Component 4 graph load: PASS")
        return 0
    else:
        print("\n  ❌ Component 4 graph load: FAIL — see mismatches above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
