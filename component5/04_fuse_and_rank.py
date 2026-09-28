#!/usr/bin/env python3
"""
Component 5 — Step 4: Fuse structural + semantic embeddings, rank gaps via FAISS.

For each (domain, year) combination, produces a ranked list of candidate gaps —
entity pairs (u, v) that have NO direct edge of any type in the cumulative slice
at year t.  Ranking is by a fused cosine score computed on vector-level fusion:

    fused_vec = alpha * pad(structural_128) + (1-alpha) * semantic_768
    fused_score = cosine(fused_vec_u, fused_vec_v)   [= FAISS IndexFlatIP distance]

where structural embeddings are the Procrustes-aligned Node2Vec embeddings from
02 (128-dim, per-year), and semantic embeddings are the time-independent
allenai/specter embeddings from 03 (768-dim, same for all years).

Vector fusion: structural (128-dim) is zero-padded to 768 dims to match
semantic dimension, then the two L2-normalized vectors are combined with the
alpha weight and the result is L2-normalized.  At alpha=1.0 the fused cosine
equals the pure structural cosine; at alpha=0.0 it equals the pure semantic
cosine; at intermediate alphas it is a genuine vector blend.

Near-duplicate filter: before top-500 selection, any candidate pair whose
representative surface forms are near-identical is excluded.  Normalization:
lowercase + strip non-alphanumerics (for Levenshtein) OR lowercase only (for
token-set, to preserve word boundaries).  Filter: normalized Levenshtein ratio
>= 0.75 OR token-set ratio >= 0.90.  Representative surface forms are the
most frequent surface_form per canonical_id from Component 2's entities JSON.

FAISS retrieval: for each alpha, build an IndexFlatIP over the L2-normalized
fused vectors, query top-K neighbors per entity, keep pairs with no direct edge
of any type in the slice, rank top-500 by fused score (FAISS distance).

Output (per domain, year, alpha):
    component5/output/gaps/{domain}_{year}_alpha{alpha}.json
        list of {entity_a, entity_b, surface_form_a, surface_form_b,
                 structural_score, semantic_score, fused_score, alpha, year}
        (top-500, descending by fused_score)
    component5/output/gaps/{domain}_{year}_alpha{alpha}_meta.json
        {domain, year, alpha, index_vector_type, total_entities_in_slice,
         total_structural_entities, total_semantic_entities, total_edges_in_slice,
         near_duplicate_filter: {threshold_lev, threshold_token_set,
                                 pairs_filtered, pairs_after_filter},
         total_candidate_pairs_considered, top_10_gaps, elapsed_seconds}

Input:
    component5/output/slices/{domain}_{year}.gpickle          (from 01)
    component5/output/embeddings/structural/{domain}_{year}.npy   (from 02)
    component5/output/embeddings/structural/{domain}_{year}_id_map.json  (from 02)
    component5/output/embeddings/semantic/{domain}.npy         (from 03)
    component5/output/embeddings/semantic/{domain}_semantic_id_map.json  (from 03)
    component2_entity_relation_extraction/output/{domain}_entities.json  (for surface forms)

Runtime: CPU only (FAISS CPU).  Seeded: numpy.random.seed(42) (FAISS IndexFlatIP
is deterministic).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
from rapidfuzz import fuzz

# ── paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP5_ROOT = PROJECT_ROOT / "component5"
COMP5_OUT = COMP5_ROOT / "output"
SLICES_DIR = COMP5_OUT / "slices"
STRUCT_DIR = COMP5_OUT / "embeddings" / "structural"
SEM_DIR = COMP5_OUT / "embeddings" / "semantic"
GAPS_DIR = COMP5_OUT / "gaps"
COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"

DOMAIN_YEARS = {
    "NLP": list(range(2018, 2025)),
    "COVID": list(range(2019, 2025)),
}

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FAISS_TOP_K = 20
GAP_TOP_N = 500
DEGENERATE_MIN_NODES = 2

# ── near-duplicate filter constants ────────────────────────────────────────

# Thresholds for the near-duplicate surface-form filter.
# Two complementary metrics with OR conjunction, each with its own normalization:
#
#   Normalized Levenshtein ratio >= 0.75
#     - Normalization: lowercase + strip non-alphanumerics
#     - Catches: "CGF" vs "CGF)" (→ "cgf" vs "cgf", ratio 1.0),
#                "ExSum" vs "ExSum)" (→ "exsum" vs "exsum", ratio 1.0),
#                "CDA" vs "CDA/S" (→ "cda" vs "cdas", ratio 0.75),
#                single-character / single-token differences.
#     - Threshold 0.75 (not 0.85) because after stripping, "CDA" vs "CDA/S"
#       yields "cda" vs "cdas" with ratio exactly 0.75; at 0.85 this would
#       slip through, but at 0.75 it's caught.  A threshold of 0.75 still
#       requires 75% character similarity — for entity names of 3-20 chars
#       this means at most 1-5 chars may differ, a meaningful distinction.
#
#   Token-set ratio >= 0.90
#     - Normalization: lowercase only (NO stripping, to preserve word boundaries)
#     - Catches: "multistep fusion schema" vs "fusion schema" (token-set
#       intersection {"fusion", "schema"} gives high ratio),
#       multi-word partial overlaps where one phrase is a near-subset of another.
#     - Stripping non-alphanumerics would collapse "multistep fusion schema"
#       into "multistepfusionschema", destroying the token structure that
#       token-set relies on — hence lowercase-only for this metric.
#
# Using both with OR ensures comprehensive coverage of the spec's four example
# cases (CGF/CGF), CDA/CDA/S, ExSum/ExSum), multistep fusion schema/fusion schema).

NEAR_DUP_LEV_THRESHOLD = 0.75
NEAR_DUP_TSR_THRESHOLD = 0.90


# ── helpers: near-duplicate filter ─────────────────────────────────────────

def normalize_lev(s: str) -> str:
    """Lowercase + strip non-alphanumerics: 'CGF)' -> 'cgf', 'CDA/S' -> 'cdas'."""
    return "".join(c.lower() for c in s if c.isalnum())


def normalize_tsr(s: str) -> str:
    """Lowercase only (preserve word boundaries for token-set): 'CDA/S' -> 'cda/s'."""
    return s.lower()


def is_near_duplicate(sf_a: str, sf_b: str) -> bool:
    """Return True if two representative surface forms are near-identical.

    Uses two complementary metrics with OR:
      - Normalized Levenshtein ratio >= 0.75 (strip + lowercase)
      - Token-set ratio >= 0.90 (lowercase only)
    A pair is filtered if EITHER metric exceeds its threshold.
    """
    if not sf_a or not sf_b:
        return False
    # Levenshtein branch: strip + lowercase
    a_lev = normalize_lev(sf_a)
    b_lev = normalize_lev(sf_b)
    if not a_lev or not b_lev:
        return False
    lev_ratio = fuzz.ratio(a_lev, b_lev) / 100.0
    if lev_ratio >= NEAR_DUP_LEV_THRESHOLD:
        return True
    # Token-set branch: lowercase only (preserve word boundaries)
    a_tsr = normalize_tsr(sf_a)
    b_tsr = normalize_tsr(sf_b)
    tsr = fuzz.token_set_ratio(a_tsr, b_tsr) / 100.0
    return tsr >= NEAR_DUP_TSR_THRESHOLD


# ── helpers: surface form loading ──────────────────────────────────────────

def load_representative_surface_forms(entities_path: Path) -> dict[str, str]:
    """Load entities JSON and return {canonical_id: most_frequent_surface_form}.

    Ties broken by first occurrence in the JSON.
    """
    with open(entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    canon_surfaces: dict[str, Counter] = defaultdict(Counter)
    canon_first: dict[str, dict[str, int]] = defaultdict(dict)
    order = 0
    for e in entities:
        cid = e.get("canonical_id", "")
        sf = e.get("surface_form", "")
        if not cid or not sf:
            continue
        canon_surfaces[cid][sf] += 1
        if sf not in canon_first[cid]:
            canon_first[cid][sf] = order
            order += 1
    result: dict[str, str] = {}
    for cid, freq in canon_surfaces.items():
        best_sf = max(freq.keys(),
                      key=lambda s: (freq[s], -canon_first[cid][s]))
        result[cid] = best_sf
    return result


# ── helpers: embedding loading ─────────────────────────────────────────────

def load_structural(domain: str, year: int) -> tuple[np.ndarray, dict[str, int]]:
    npy_path = STRUCT_DIR / f"{domain}_{year}.npy"
    idmap_path = STRUCT_DIR / f"{domain}_{year}_id_map.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"Structural embedding missing: {npy_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"Structural id_map missing: {idmap_path}")
    matrix = np.load(npy_path)
    with open(idmap_path, encoding="utf-8") as f:
        id_map = json.load(f)
    return matrix.astype(np.float32), id_map


def load_semantic(domain: str) -> tuple[np.ndarray, dict[str, int]]:
    npy_path = SEM_DIR / f"{domain}.npy"
    idmap_path = SEM_DIR / f"{domain}_semantic_id_map.json"
    if not npy_path.exists():
        raise FileNotFoundError(f"Semantic embedding missing: {npy_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"Semantic id_map missing: {idmap_path}")
    matrix = np.load(npy_path)
    with open(idmap_path, encoding="utf-8") as f:
        id_map = json.load(f)
    return matrix.astype(np.float32), id_map


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row.  Returns new array."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ── helpers: fused embedding computation ───────────────────────────────────

def compute_fused_embeddings(
    struct_norm: np.ndarray,
    sem_norm: np.ndarray,
    struct_id_map: dict[str, int],
    sem_id_map: dict[str, int],
    alpha: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute L2-normalized fused embeddings for entities in BOTH maps.

    Pipeline:
      1. Find common canonical_ids (in both struct and semantic maps)
      2. Extract and L2-normalize structural (128-dim) and semantic (768-dim)
         rows for common entities
      3. Zero-pad normalized structural to 768 dims
      4. fused = alpha * padded_norm_struct + (1-alpha) * norm_sem
      5. L2-normalize fused

    Returns (fused_matrix [n, 768], fused_id_map {canonical_id: row}).
    """
    common = sorted(set(struct_id_map) & set(sem_id_map))
    if not common:
        return np.zeros((0, sem_norm.shape[1]), dtype=np.float32), {}

    struct_dim = struct_norm.shape[1]
    sem_dim = sem_norm.shape[1]

    struct_rows = np.stack([struct_norm[struct_id_map[c]] for c in common],
                           dtype=np.float32)
    sem_rows = np.stack([sem_norm[sem_id_map[c]] for c in common],
                        dtype=np.float32)

    # Ensure normalized (should already be, but be safe)
    norm_struct = l2_normalize(struct_rows)
    norm_sem = l2_normalize(sem_rows)

    # Zero-pad structural to semantic dim
    if struct_dim < sem_dim:
        padded = np.zeros((len(common), sem_dim), dtype=np.float32)
        padded[:, :struct_dim] = norm_struct
    else:
        padded = norm_struct  # struct_dim >= sem_dim (shouldn't happen)

    fused = alpha * padded + (1 - alpha) * norm_sem
    norm_fused = l2_normalize(fused)

    fused_id_map = {cid: i for i, cid in enumerate(common)}
    return norm_fused, fused_id_map


# ── helpers: FAISS ──────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    embs = embeddings.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


# ── helpers: validation ────────────────────────────────────────────────────

def validate_inputs(domain: str, year: int,
                    entities_path: Path) -> None:
    missing: list[str] = []
    for p in [
        SLICES_DIR / f"{domain}_{year}.gpickle",
        STRUCT_DIR / f"{domain}_{year}.npy",
        STRUCT_DIR / f"{domain}_{year}_id_map.json",
        SEM_DIR / f"{domain}.npy",
        SEM_DIR / f"{domain}_semantic_id_map.json",
        entities_path,
    ]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"Missing inputs for {domain} {year}.  Run previous steps first.\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


# ── core: gap finding for one alpha (fused FAISS + near-dup filter) ────────

def find_gaps_for_alpha(
    domain: str,
    year: int,
    alpha: float,
    struct_embs: np.ndarray,
    struct_id_map: dict[str, int],
    sem_embs: np.ndarray,
    sem_id_map: dict[str, int],
    surface_form_map: dict[str, str],
    G: nx.DiGraph,
) -> tuple[list[dict], dict]:
    """Find top-GAP_TOP_N gaps for one alpha using fused-space FAISS.

    Returns (gaps, stats) where:
      gaps: list of gap dicts, descending by fused_score, truncated to GAP_TOP_N.
      stats: {pairs_before_filter, pairs_filtered, pairs_after_filter, gaps_saved}.
    """
    n_nodes = struct_embs.shape[0]
    if n_nodes < DEGENERATE_MIN_NODES:
        return [], {"pairs_before_filter": 0, "pairs_filtered": 0,
                    "pairs_after_filter": 0, "gaps_saved": 0}

    # Compute fused embeddings for this alpha
    fused_embs, fused_id_map = compute_fused_embeddings(
        struct_embs, sem_embs, struct_id_map, sem_id_map, alpha,
    )
    if fused_embs.shape[0] == 0:
        return [], {"pairs_before_filter": 0, "pairs_filtered": 0,
                    "pairs_after_filter": 0, "gaps_saved": 0}

    fused_cid_list = sorted(fused_id_map.keys())

    # Edge set for O(1) "no direct edge" check
    edge_set: set[tuple[str, str]] = set()
    for u, v in G.edges():
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Build FAISS index over fused embeddings
    index = build_faiss_index(fused_embs)

    # Query top-K fused neighbors per entity
    distances, labels = index.search(fused_embs, FAISS_TOP_K)

    # Collect candidate pairs — deduplicate, keep best fused score
    pair_scores: dict[tuple[str, str], dict] = {}
    pairs_before_filter = 0

    for i, cid in enumerate(fused_cid_list):
        u = cid
        for j in range(len(labels[i])):
            neighbor_idx = labels[i][j]
            if neighbor_idx == i:
                continue
            if neighbor_idx >= len(fused_cid_list):
                continue
            v = fused_cid_list[neighbor_idx]
            if u >= v:
                continue

            # Check: no direct edge in slice
            if (u, v) in edge_set or (v, u) in edge_set:
                continue

            pairs_before_filter += 1

            # Near-duplicate filter (BEFORE adding to pair_scores)
            sf_u = surface_form_map.get(u, "")
            sf_v = surface_form_map.get(v, "")
            if sf_u and sf_v and is_near_duplicate(sf_u, sf_v):
                continue

            # Fused score = FAISS distance (inner product of L2-normed fused vecs)
            fused_score = float(distances[i][j])

            # Individual cosines for reporting (from normalized vectors)
            u_s = struct_id_map.get(u)
            v_s = struct_id_map.get(v)
            struct_score = 0.0
            if u_s is not None and v_s is not None:
                struct_score = float(np.dot(
                    l2_normalize(struct_embs[u_s:u_s+1])[0],
                    l2_normalize(struct_embs[v_s:v_s+1])[0],
                ))
            sem_score = 0.0
            u_sem = sem_id_map.get(u)
            v_sem = sem_id_map.get(v)
            if u_sem is not None and v_sem is not None:
                sem_score = float(np.dot(
                    l2_normalize(sem_embs[u_sem:u_sem+1])[0],
                    l2_normalize(sem_embs[v_sem:v_sem+1])[0],
                ))

            pair_key = (u, v) if u < v else (v, u)
            if pair_key not in pair_scores or fused_score > pair_scores[pair_key]["fused_score"]:
                pair_scores[pair_key] = {
                    "entity_a": u,
                    "entity_b": v,
                    "surface_form_a": surface_form_map.get(u, ""),
                    "surface_form_b": surface_form_map.get(v, ""),
                    "structural_score": round(struct_score, 6),
                    "semantic_score": round(sem_score, 6),
                    "fused_score": round(fused_score, 6),
                    "alpha": alpha,
                    "year": year,
                }

    stats = {
        "pairs_before_filter": pairs_before_filter,
        "pairs_filtered": pairs_before_filter - len(pair_scores),
        "pairs_after_filter": len(pair_scores),
        "gaps_saved": 0,
    }

    ranked = sorted(pair_scores.values(), key=lambda x: -x["fused_score"])
    stats["gaps_saved"] = len(ranked)
    return ranked[:GAP_TOP_N], stats


# ── main processing per (domain, year) ─────────────────────────────────────

def process_domain_year(domain: str, year: int, alpha: float | None,
                        surface_form_map: dict[str, str]) -> dict:
    entities_path = COMP2_OUT / f"{domain}_entities.json"
    validate_inputs(domain, year, entities_path)

    with open(SLICES_DIR / f"{domain}_{year}.gpickle", "rb") as fh:
        G = pickle.load(fh)
    struct_embs, struct_id_map = load_structural(domain, year)
    sem_embs, sem_id_map = load_semantic(domain)

    # Normalize once for this (domain, year)
    struct_norm = l2_normalize(struct_embs)
    sem_norm = l2_normalize(sem_embs)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_struct = struct_embs.shape[0]
    n_semantic = sem_embs.shape[0]

    if alpha is not None:
        alphas_to_run = [alpha]
    else:
        alphas_to_run = ALPHAS

    results: dict[str, dict] = {}

    for a in alphas_to_run:
        t0 = time.time()
        gaps, stats = find_gaps_for_alpha(
            domain, year, a,
            struct_norm, struct_id_map,
            sem_norm, sem_id_map,
            surface_form_map,
            G,
        )
        elapsed = time.time() - t0

        alpha_str = f"{a:.2f}".rstrip("0").rstrip(".")
        gaps_path = GAPS_DIR / f"{domain}_{year}_alpha{alpha_str}.json"
        gaps_path.write_text(json.dumps(gaps, indent=2))

        meta = {
            "domain": domain,
            "year": year,
            "alpha": a,
            "index_vector_type": f"fused_alpha_{a}",
            "total_entities_in_slice": n_nodes,
            "total_structural_entities": n_struct,
            "total_semantic_entities": n_semantic,
            "total_edges_in_slice": n_edges,
            "near_duplicate_filter": {
                "threshold_lev": NEAR_DUP_LEV_THRESHOLD,
                "threshold_token_set": NEAR_DUP_TSR_THRESHOLD,
                "normalization_lev": "lowercase + strip non-alphanumerics",
                "normalization_tsr": "lowercase only (preserve word boundaries)",
                "pairs_filtered": stats["pairs_filtered"],
                "pairs_after_filter": stats["pairs_after_filter"],
                "pairs_before_filter": stats["pairs_before_filter"],
            },
            "total_candidate_pairs_considered": stats["pairs_after_filter"],
            "top_10_gaps": gaps[:10],
            "elapsed_seconds": round(elapsed, 3),
        }
        meta_path = GAPS_DIR / f"{domain}_{year}_alpha{alpha_str}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        print(f"  [alpha={a}] {len(gaps)} gaps, "
              f"{stats['pairs_filtered']} near-dupes filtered "
              f"(of {stats['pairs_before_filter']} candidates) → "
              f"{gaps_path.name}")

        results[f"alpha{a}"] = {
            "gaps_file": str(gaps_path.relative_to(COMP5_ROOT)),
            "meta_file": str(meta_path.relative_to(COMP5_ROOT)),
            "gap_count": len(gaps),
            "elapsed_seconds": round(elapsed, 3),
            "pairs_filtered": stats["pairs_filtered"],
        }

    return results


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 5 Step 4 — fuse + rank gaps via FAISS")
    parser.add_argument("--domain", type=str,
                        choices=["NLP", "COVID"],
                        help="domain to process (default: both)")
    parser.add_argument("--year", type=int,
                        help="year to process (default: all years for domain)")
    parser.add_argument("--alpha", type=float,
                        help="single alpha value (default: all ALPHAS)")
    parser.add_argument("--force", action="store_true",
                        help="redo gaps even if JSON exists")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    np.random.seed(42)

    GAPS_DIR.mkdir(parents=True, exist_ok=True)

    domains = [args.domain] if args.domain else list(DOMAIN_YEARS.keys())

    # Pre-load surface form maps per domain
    surface_form_maps: dict[str, dict[str, str]] = {}
    for d in domains:
        entities_path = COMP2_OUT / f"{d}_entities.json"
        if entities_path.exists():
            print(f"  Loading representative surface forms for {d}…")
            t_sf = time.time()
            surface_form_maps[d] = load_representative_surface_forms(entities_path)
            print(f"    → {len(surface_form_maps[d])} canonical_ids with surface forms "
                  f"({time.time() - t_sf:.1f}s)")
        else:
            print(f"  [WARNING] Entities JSON not found for {d}: {entities_path}")

    years_to_run: dict[str, list[int]] = {}
    for d in domains:
        all_years = DOMAIN_YEARS[d]
        if args.year is not None:
            if args.year not in all_years:
                print(f"  [warning] year {args.year} not in {d} range "
                      f"{all_years[0]}–{all_years[-1]}")
                continue
            years_to_run[d] = [args.year]
        else:
            years_to_run[d] = all_years

    total_gaps = 0
    total_filtered = 0
    for domain in domains:
        sf_map = surface_form_maps.get(domain, {})
        for year in years_to_run[domain]:
            print(f"\n{'='*60}")
            print(f"  {domain} — year {year}")
            print(f"{'='*60}")

            if not args.force:
                alpha_str = "0.5"
                gaps_path = GAPS_DIR / f"{domain}_{year}_alpha{alpha_str}.json"
                if gaps_path.exists():
                    print(f"  Gaps for alpha=0.5 already exist, skipping "
                          f"(use --force to redo)")
                    existing = json.loads(gaps_path.read_text())
                    total_gaps += len(existing)
                    continue

            results = process_domain_year(domain, year, args.alpha, sf_map)
            for k, v in results.items():
                total_gaps += v["gap_count"]
                total_filtered += v.get("pairs_filtered", 0)

    elapsed_total = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  Done.  Total gaps produced: {total_gaps:,}")
    print(f"  Total near-duplicate pairs filtered: {total_filtered:,}")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f} MiB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
