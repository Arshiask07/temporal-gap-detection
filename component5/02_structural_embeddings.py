#!/usr/bin/env python3
"""
Component 5 — Step 2: Structural embeddings (Node2Vec + orthogonal Procrustes).

For each domain (NLP, COVID) and each year t, runs Node2Vec 128-dim on the
cumulative slice produced by 01_build_slices.py, then aligns embeddings across
years via orthogonal Procrustes so the same canonical_id has comparable
coordinates year-over-year.

Input:
    component5/output/slices/{domain}_{year}.gpickle   (from 01)

Output (aligned, overwriting the raw .npy with aligned coordinates):
    component5/output/embeddings/structural/{domain}_{year}.npy       — [n_nodes, 128]
    component5/output/embeddings/structural/{domain}_{year}_id_map.json — {canonical_id: row_index}
    component5/output/embeddings/structural/structural_build_log.json  — per-slice timing + alignment stats

Runtime: CPU only.  Seeded: numpy.random, random, Node2Vec(seed=42, workers=1).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
import tracemalloc
from pathlib import Path

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from scipy.linalg import orthogonal_procrustes

# ── paths ──────────────────────────────────────────────────────────────────
# Derived from this script's own location so the pipeline runs unchanged on
# any machine (Windows/macOS/Linux) with no hand-editing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP5_ROOT = PROJECT_ROOT / "component5"
COMP5_OUT = COMP5_ROOT / "output"
SLICES_DIR = COMP5_OUT / "slices"
STRUCT_DIR = COMP5_OUT / "embeddings" / "structural"

# Domain → years (must match 01_build_slices.py)
DOMAIN_SPECS = {
    "NLP": list(range(2018, 2025)),       # 2018..2024
    "COVID": list(range(2019, 2025)),     # 2019..2024
}

# Node2Vec hyperparameters (fixed, per spec)
NV_ARGS = dict(
    dimensions=128,
    walk_length=80,
    num_walks=10,
    p=1,
    q=1,
    workers=1,
    seed=42,
    quiet=True,
)

DEGENERATE_THRESHOLD = 10   # < this many nodes → placeholder, not Node2Vec


# ── helpers ────────────────────────────────────────────────────────────────

def load_slice(path: Path) -> nx.DiGraph:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def fit_node2vec(G: nx.DiGraph) -> dict[str, np.ndarray]:
    """Run Node2Vec on G, return {node: 128-dim numpy array}.

    Node2Vec.__init__ builds random walks; .fit(**kwargs) passes kwargs
    through to gensim.models.Word2Vec.  We pass window/epochs/min_count
    as fit kwargs (not Node2Vec constructor kwargs).
    """
    n2v = Node2Vec(G, **NV_ARGS)
    model = n2v.fit(window=10, epochs=10, min_count=1)
    embeddings: dict[str, np.ndarray] = {}
    for node in G.nodes:
        embeddings[node] = np.asarray(model.wv[str(node)], dtype=np.float32)
    return embeddings


def align_embeddings(prev_embs: dict[str, np.ndarray],
                     curr_embs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Orthogonal Procrustes: align curr_embs into prev_embs's coordinate frame.

    Only shared canonical_ids participate in rotation estimation.  Non-shared
    IDs in curr_embs are rotated using the same R.
    """
    shared = sorted(set(prev_embs) & set(curr_embs))
    if len(shared) < 3:
        # Too few shared points for a stable rotation; return curr unaligned
        return dict(curr_embs)

    # Build matrices: rows = shared entities, cols = 128 dims
    A = np.stack([prev_embs[s] for s in shared], dtype=np.float64)  # [S, 128]
    B = np.stack([curr_embs[s] for s in shared], dtype=np.float64)  # [S, 128]

    # orthogonal_procrustes returns (R, scale) where B_aligned = scale * B @ R
    R, scale = orthogonal_procrustes(A, B)
    # scale ~ 1.0 for unit-normed embeddings; apply R only
    aligned: dict[str, np.ndarray] = {}
    for node, vec in curr_embs.items():
        aligned[node] = np.asarray(vec, dtype=np.float64) @ R
    return aligned


def save_embeddings(emb_dict: dict[str, np.ndarray], year: int, domain: str,
                    log_entry: dict) -> None:
    """Save aligned embeddings + id map for one (domain, year)."""
    nodes = sorted(emb_dict.keys())
    idx_map = {n: i for i, n in enumerate(nodes)}
    matrix = np.stack([emb_dict[n] for n in nodes], dtype=np.float32)

    npy_path = STRUCT_DIR / f"{domain}_{year}.npy"
    np.save(npy_path, matrix)

    idmap_path = STRUCT_DIR / f"{domain}_{year}_id_map.json"
    idmap_path.write_text(json.dumps(idx_map, indent=2))

    log_entry.update({
        "nodes": len(nodes),
        "embedding_shape": list(matrix.shape),
        "id_map_path": str(idmap_path.relative_to(COMP5_ROOT)),
        "npy_path": str(npy_path.relative_to(COMP5_ROOT)),
    })


def validate_slices_exist() -> None:
    """Check that all gpickle slices exist for both domains.  Abort if any missing."""
    missing: list[str] = []
    for domain, years in DOMAIN_SPECS.items():
        for year in years:
            path = SLICES_DIR / f"{domain}_{year}.gpickle"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing slice gpickle files. Run 01_build_slices.py first.  "
            f"Missing: {missing[0]}" + (f" (+{len(missing)-1} more)" if len(missing) > 1 else "")
        )
    print("[validate] All slice gpickle files found.")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 5 Step 2 — Node2Vec + Procrustes alignment")
    parser.add_argument("--force", action="store_true",
                        help="redo embeddings even if .npy exists")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    np.random.seed(42)
    random.seed(42)

    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    validate_slices_exist()

    build_log: list[dict] = []

    for domain, years in DOMAIN_SPECS.items():
        print(f"\n{'='*60}")
        print(f"  Domain: {domain}  ({years[0]}–{years[-1]})")
        print(f"{'='*60}")

        domain_log: list[dict] = []
        prev_embs: dict[str, np.ndarray] | None = None  # anchor for alignment

        for year in years:
            slice_path = SLICES_DIR / f"{domain}_{year}.gpickle"
            npy_path = STRUCT_DIR / f"{domain}_{year}.npy"

            if npy_path.exists() and not args.force:
                print(f"  [{year}] embeddings exist, skipping (use --force to redo)")
                G = load_slice(slice_path)
                entry = {
                    "domain": domain, "year": year,
                    "status": "cached",
                    "slice_nodes": G.number_of_nodes(),
                    "slice_edges": G.number_of_edges(),
                }
                if npy_path.exists():
                    mat = np.load(npy_path)
                    entry["embedding_shape"] = list(mat.shape)
                domain_log.append(entry)
                continue

            t0 = time.time()
            G = load_slice(slice_path)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            log_entry = {
                "domain": domain, "year": year,
                "slice_nodes": n_nodes, "slice_edges": n_edges,
                "status": "built",
            }

            if n_nodes < DEGENERATE_THRESHOLD:
                print(f"  [WARNING] [{year}] only {n_nodes} nodes — "
                      f"writing placeholder, skipping Node2Vec")
                placeholder = np.zeros((1, 128), dtype=np.float32)
                np.save(npy_path, placeholder)
                (STRUCT_DIR / f"{domain}_{year}_id_map.json").write_text(
                    json.dumps({"_warning": f"degenerate slice: only {n_nodes} nodes"}, indent=2)
                )
                log_entry["degenerate"] = True
                log_entry["embedding_shape"] = [1, 128]
                domain_log.append(log_entry)
                prev_embs = None  # cannot align from degenerate
                continue

            # Fit Node2Vec
            fit_t0 = time.time()
            raw_embs = fit_node2vec(G)
            fit_elapsed = time.time() - fit_t0
            log_entry["node2vec_fit_time_s"] = round(fit_elapsed, 3)
            print(f"  [{year}] Node2Vec done: {n_nodes} nodes, {n_edges} edges, "
                  f"fit={fit_elapsed:.1f}s")

            # Align to previous year if possible
            if prev_embs is not None:
                n_shared = len(set(prev_embs) & set(raw_embs))
                log_entry["shared_with_prev_year"] = n_shared
                aligned = align_embeddings(prev_embs, raw_embs)
                log_entry["alignment_applied"] = True
                print(f"    → aligned to prev year: {n_shared} shared IDs")
            else:
                aligned = raw_embs
                log_entry["alignment_applied"] = False
                log_entry["shared_with_prev_year"] = 0
                print(f"    → first year, no alignment (anchor)")

            save_embeddings(aligned, year, domain, log_entry)
            prev_embs = aligned  # next year aligns to this one
            domain_log.append(log_entry)

        # Domain summary
        print(f"\n  {domain} summary:")
        for e in domain_log:
            align = "aligned" if e.get("alignment_applied") else "anchor"
            deg = " [DEGENERATE]" if e.get("degenerate") else ""
            print(f"    {e['year']}: {e.get('slice_nodes', 0):>5} nodes, "
                  f"{e.get('slice_edges', 0):>5} edges, "
                  f"fit={e.get('node2vec_fit_time_s', 0):.1f}s, "
                  f"{align}{deg}")

        build_log.extend(domain_log)

    # Write log
    log_path = STRUCT_DIR / "structural_build_log.json"
    log_path.write_text(json.dumps(build_log, indent=2))
    print(f"\n  Build log: {log_path}")

    elapsed_total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  Done.  Total time: {elapsed_total:.1f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f} MiB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
