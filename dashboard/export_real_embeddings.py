#!/usr/bin/env python3
"""
Component 7 — Export real Component 5 embeddings to dashboard CSV format.

Converts the .npy + id_map.json files from Component 5 into the CSV files
the Streamlit dashboard expects:

    dashboard/exports/node2vec_{nlp,covid-19}_{year}.csv
    dashboard/exports/specter2_{nlp,covid-19}_{year}.csv

Schema: entity_id, dim_0, dim_1, ..., dim_{EMB_DIM-1}

node2vec: one CSV per year from C5 structural embeddings (128-dim)
specter2:  C5 semantic embeddings are time-independent (768-dim),
           so the same vector is written for every year in range.

Usage:
    python3 export_real_embeddings.py [--domain NLP|COVID] [--force]

Input:
    component5/output/embeddings/structural/{domain}_{year}.npy
    component5/output/embeddings/structural/{domain}_{year}_id_map.json
    component5/output/embeddings/semantic/{domain}.npy
    component5/output/embeddings/semantic/{domain}_semantic_id_map.json

Output:
    dashboard/exports/node2vec_{short}_{year}.csv
    dashboard/exports/specter2_{short}_{year}.csv
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
COMP5_ROOT = DASHBOARD_DIR.parent / "component5"
STRUCT_DIR = COMP5_ROOT / "output" / "embeddings" / "structural"
SEM_DIR = COMP5_ROOT / "output" / "embeddings" / "semantic"
EXPORT_DIR = DASHBOARD_DIR / "exports"

DOMAIN_CONFIG = {
    "NLP": {
        "short": "nlp",
        "years": list(range(2018, 2025)),
    },
    "COVID": {
        "short": "covid-19",
        "years": list(range(2019, 2025)),
    },
}

# Derived from config.py — must match
EMB_DIM_STRUCT = 128
EMB_DIM_SEM = 768


def load_struct(domain: str, year: int) -> tuple[np.ndarray, dict[str, int]]:
    """Load structural embeddings + id_map for one (domain, year)."""
    npy_path = STRUCT_DIR / f"{domain}_{year}.npy"
    idmap_path = STRUCT_DIR / f"{domain}_{year}_id_map.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"Structural embedding missing: {npy_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"Structural id_map missing: {idmap_path}")
    matrix = np.load(npy_path)
    id_map = __import__("json").loads(idmap_path.read_text())
    return matrix.astype(np.float32), id_map


def load_semantic(domain: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load semantic (SPECTER2) embeddings + id_map — time-independent."""
    npy_path = SEM_DIR / f"{domain}.npy"
    idmap_path = SEM_DIR / f"{domain}_semantic_id_map.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"Semantic embedding missing: {npy_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"Semantic id_map missing: {idmap_path}")
    matrix = np.load(npy_path)
    id_map = __import__("json").loads(idmap_path.read_text())
    return matrix.astype(np.float32), id_map


def write_embedding_csv(
    matrix: np.ndarray,
    id_map: dict[str, int],
    export_path: Path,
    emb_dim: int,
) -> int:
    """Write embedding matrix to dashboard CSV. Returns row count."""
    if matrix.shape[0] == 0:
        # Write empty file with just the header
        cols = ["entity_id"] + [f"dim_{i}" for i in range(emb_dim)]
        pd.DataFrame(columns=cols).to_csv(export_path, index=False)
        return 0

    # Sort by id_map index for consistent ordering
    sorted_ids = sorted(id_map.keys(), key=lambda k: id_map[k])
    rows = []
    for cid in sorted_ids:
        idx = id_map[cid]
        row = {"entity_id": cid}
        for d in range(emb_dim):
            row[f"dim_{d}"] = round(float(matrix[idx, d]), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)
    return len(df)


def export_domain(domain: str, short: str, years: list[int], force: bool) -> dict:
    """Export embeddings for one domain. Returns stats dict."""
    stats = {"node2vec_files": 0, "specter2_files": 0, "node2vec_rows": 0, "specter2_rows": 0}

    # Load semantic once (time-independent)
    print(f"  Loading semantic embeddings for {domain}…")
    sem_matrix, sem_id_map = load_semantic(domain)
    print(f"    → {sem_matrix.shape[0]} entities, {sem_matrix.shape[1]}-dim")

    for year in years:
        # Node2Vec (structural, per-year)
        n2v_path = EXPORT_DIR / f"node2vec_{short}_{year}.csv"
        if n2v_path.exists() and not force:
            print(f"  [SKIP] node2vec_{short}_{year}.csv exists (use --force)")
            stats["node2vec_files"] += 1
            continue

        print(f"  Loading structural embeddings for {domain} {year}…")
        n2v_matrix, n2v_id_map = load_struct(domain, year)
        count = write_embedding_csv(n2v_matrix, n2v_id_map, n2v_path, EMB_DIM_STRUCT)
        stats["node2vec_files"] += 1
        stats["node2vec_rows"] += count
        print(f"    → node2vec_{short}_{year}.csv: {count} rows")

        # Specter2 (semantic, same for all years)
        sp_path = EXPORT_DIR / f"specter2_{short}_{year}.csv"
        if sp_path.exists() and not force:
            print(f"  [SKIP] specter2_{short}_{year}.csv exists (use --force)")
            stats["specter2_files"] += 1
            continue

        count = write_embedding_csv(sem_matrix, sem_id_map, sp_path, EMB_DIM_SEM)
        stats["specter2_files"] += 1
        stats["specter2_rows"] += count
        print(f"    → specter2_{short}_{year}.csv: {count} rows")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 7 — export real C5 embeddings to dashboard CSV format"
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
    total_stats = {"node2vec_files": 0, "specter2_files": 0,
                   "node2vec_rows": 0, "specter2_rows": 0}

    for domain in domains:
        cfg = DOMAIN_CONFIG[domain]
        print(f"\n{'='*50}")
        print(f"  {domain} ({cfg['short']}) — {len(cfg['years'])} years")
        print(f"{'='*50}")
        s = export_domain(domain, cfg["short"], cfg["years"], args.force)
        for k in total_stats:
            total_stats[k] += s[k]

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*50}")
    print(f"  Done.  Total time: {elapsed:.1f}s   Peak memory: {peak/1024/1024:.1f} MiB")
    print(f"  node2vec: {total_stats['node2vec_files']} files, {total_stats['node2vec_rows']:,} rows")
    print(f"  specter2: {total_stats['specter2_files']} files, {total_stats['specter2_rows']:,} rows")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
