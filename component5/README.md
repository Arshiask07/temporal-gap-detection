# Component 5 — Dual-Channel Temporal Gap Detection Engine

**Status:** Implemented (5 scripts + README).  Ready to run end-to-end.

**Folder:** `component5/`  
**Output:** `component5/output/`  
**Dependencies:** torch, transformers, faiss-cpu, networkx, node2vec, scikit-learn, numpy, scipy  
**GPU:** Optional (03 uses CUDA if available, falls back to CPU).  All other scripts are CPU-only.  
**Neo4j:** Not required.  Component 5 operates entirely on Component 2's JSON outputs + its own caches.  
**Components 1–4:** Untouched.  Component 5 reads from them (read-only) and writes only to `component5/output/`.

---

## 1. Overview

Component 5 takes the structured entity and relation extractions from Component 2 (SciBERT NER + RE) and produces **ranked candidate research gaps** using a dual-channel approach:

- **Structural channel** (02): Node2Vec 128-dim embeddings of each per-year cumulative graph slice, aligned across years via orthogonal Procrustes so the same entity has comparable coordinates year-over-year.
- **Semantic channel** (03): SPECTER (768-dim) or MiniLM (384-dim fallback) embeddings of each entity's representative surface_form + type label.  Time-independent — same semantic embedding for all years.

**Fusion:** `fused_score = alpha * structural_cosine + (1 - alpha) * semantic_cosine` for each candidate pair.  Alpha is swept across [0, 0.25, 0.5, 0.75, 1.0] to understand the contribution of each channel.

**Ranking:** FAISS approximate nearest neighbors over the structural embeddings to avoid O(n²) — for each entity, query top-20 structural neighbors, filter to pairs with no direct edge in the slice, compute fused score, keep top-500.

**Evaluation:** 05 reuses Component 3's retrospective methodology (post-cutoff co-mention check against Component 2's extracted paper texts) to score the ranked gaps.  Alpha sweep + ablations (structural-only, semantic-only, random baseline) produce an ablation JSON + markdown table per domain.

---

## 2. Architecture diagram

```mermaid
flowchart TD
    CP2[Component 2 Outputs\n(NLP/COVID entities + relations JSON)] --> S1[01_build_slices.py\nper-year cumulative DiGraph slices]
    S1 --> S2[02_structural_embeddings.py\nNode2Vec 128-dim + Procrustes alignment]
    CP2 --> S3[03_semantic_embeddings.py\nSPECTER / MiniLM surface_form + type embeddings]
    S2 --> S4[04_fuse_and_rank.py\nFAISS top-K structural neighbors\n+ fuse with semantic\n+ rank gaps per alpha]
    S3 --> S4
    S4 --> S5[05_evaluate.py\nretrospective co-mention check\nalpha sweep + ablations\nablation JSON + markdown + top-10 gaps]
    S5 --> OUT[component5/output/\nabation_*.json + .md\ntop_gaps_*.json]
```

**Data flow:**
1. Component 2 JSONs (read-only) → 01 builds per-year slices → 02 fits Node2Vec on each slice → aligns across years
2. Component 2 entities JSON → 03 embeds each entity's surface_form + type → semantic embedding matrix per domain
3. 02's structural embeddings + 03's semantic embeddings + 01's slices → 04 ranks gaps per (domain, year, alpha)
4. 04's gaps + Component 2's extracted.json paper texts → 05 evaluates via retrospective co-mention check

---

## 3. Run order

Run the scripts in order 1 → 5.  Each script validates its inputs on startup and raises explicit `FileNotFoundError` with the exact missing path if a prerequisite is missing.  Re-running skips completed stages unless `--force` is passed.

```bash
# All commands run from /Users/anjan/Desktop/capstone_sep_15/

# Step 1: Build per-year cumulative slices
python3 component5/01_build_slices.py --force

# Step 2: Node2Vec + Procrustes alignment (CPU, ~5 min for both domains)
python3 component5/02_structural_embeddings.py --force

# Step 3: Semantic embeddings (SPECTER if RAM allows, MiniLM fallback)
#          (~10-30 min depending on model + entity count, GPU optional)
python3 component5/03_semantic_embeddings.py --force

# Step 4: Fuse + FAISS rank (per domain/year/alpha)
#          Run for all domains, all years, all alphas (default behavior)
python3 component5/04_fuse_and_rank.py

# Step 5: Retrospective evaluation + alpha sweep + ablations
python3 component5/05_evaluate.py
```

### Selective re-runs

```bash
# Re-run only NLP slices
python3 component5/01_build_slices.py --force  # (no per-domain flag; always both)

# Re-run only 2024 for NLP in step 4
python3 component5/04_fuse_and_rank.py --domain NLP --year 2024 --force

# Re-run evaluation for NLP only
python3 component5/05_evaluate.py --domain NLP --force
```

### Flags

| Flag | Scripts | Meaning |
|---|---|---|
| `--force` | all | Redo work even if output cache exists |
| `--domain` | 04, 05 | Restrict to NLP or COVID (default: both) |
| `--year` | 04 | Restrict to one year (default: all years for domain) |
| `--alpha` | 04 | Single alpha value (default: all ALPHAS) |
| `--batch-size` | 03 | Texts per forward pass (default 8; increase if RAM allows) |
| `--model` | 03 | Explicit model name, overrides auto-detection |

---

## 4. Expected outputs

After a full clean run (01–05, `--force` on each):

```
component5/output/
├── slices_build_log.json                          # 01: per-slice node/edge counts + timing
├── slices/
│   ├── NLP_2018.gpickle ... NLP_2024.gpickle      # 01: 7 DiGraph slices
│   └── COVID_2019.gpickle ... COVID_2024.gpickle   # 01: 6 DiGraph slices
└── embeddings/
    ├── structural/
    │   ├── NLP_2018.npy ... NLP_2024.npy           # 02: [n, 128] aligned Node2Vec embeddings
    │   ├── NLP_2018_id_map.json ...                 # 02: {canonical_id: row_index}
    │   ├── COVID_2019.npy ... COVID_2024.npy
    │   ├── COVID_2019_id_map.json ...
    │   └── structural_build_log.json                # 02: per-slice fit time + alignment stats
    └── semantic/
        ├── NLP.npy  (or MiniLM)                     # 03: [n, dim] L2-normalized semantic embeddings
        ├── NLP_semantic_id_map.json                 # 03: {canonical_id: row_index}
        ├── COVID.npy
        ├── COVID_semantic_id_map.json
        └── semantic_build_log.json                  # 03: model_name, dim, device, time, peak_mem
└── gaps/
    ├── NLP_2018_alpha0.0.json ...                  # 04: top-500 gaps per (domain, year, alpha)
    ├── NLP_2018_alpha0.0_meta.json                 # 04: counts + top-10
    ├── NLP_2018_alpha0.25.json ...
    ├── NLP_2018_alpha0.5.json  (primary, used by 05)
    ├── NLP_2018_alpha0.5_meta.json
    ├── NLP_2018_alpha0.75.json ...
    ├── NLP_2018_alpha1.0.json ...
    ├── ... (same for 2019–2024, NLP + COVID)
    └── COVID_2024_alpha0.5.json  (primary, used by 05)
└── ablation_NLP.json                               # 05: {cutoff: {setting: {hit_rate, hits, misses, ...}}}
├── ablation_NLP.md                                 # 05: markdown table, best per cutoff in bold
├── ablation_COVID.json
├── ablation_COVID.md
├── top_gaps_NLP_latest.json                        # 05: top-10 NLP gaps for 2024, alpha=0.5, enriched
└── top_gaps_COVID_latest.json                      # 05: top-10 COVID gaps for 2024, alpha=0.5, enriched
```

### Counts reconciliation

The slice builder (01) and structural embeddings (02) reconcile with `component4/graph_summary.json`:

| Domain | Metric | graph_summary.json | 01 slice (final year) | 02 embedding rows |
|---|---|---|---|---|
| NLP | canonical_ids | 12,867 | 10,611 (2024 slice) | 10,611 (2024) |
| NLP | relations | 22,690 | 21,413 (2024 slice) | — |
| COVID | canonical_ids | 5,500 | 3,968 (2024 slice) | 3,968 (2024) |
| COVID | relations | 8,081 | 7,509 (2024 slice) | — |

The 2024 slice doesn't contain ALL entities/relations because it only includes relations with year ≤ 2024, and not all 12,867 canonical_ids appear in relations by 2024 (some appear only in earlier years or not at all in relations).  The structural embeddings cover exactly the entities present in each year's slice.

---

## 5. Design decisions

### 5.1 Canonical_id + domain keying

NLP and COVID canonical_ids overlap by **5,099 IDs** (verified on disk).  Every operation that touches both domains must key entities by `(canonical_id, domain)`.  Within a single domain's processing (which all 5 scripts do — domains are processed separately from start to finish), `canonical_id` alone is safe.  The only place this matters is:

- **01** processes domains separately → no cross-domain collision
- **02** processes domains separately → no cross-domain collision
- **03** processes domains separately → no cross-domain collision
- **04** processes domains separately → no cross-domain collision
- **05** processes domains separately → no cross-domain collision

The 5,099 overlap is documented for transparency but doesn't affect any script's correctness because no script ever merges NLP and COVID data.

### 5.2 FAISS for O(n²) avoidance

With ~12,867 NLP entities, a naive all-pairs cosine scan is 12,867² / 2 ≈ 82M pairs per year per alpha.  Instead:

1. L2-normalize structural embeddings
2. Build a FAISS `IndexFlatIP` (inner product = cosine on normalized vectors)
3. For each entity, query top-20 nearest structural neighbors
4. For each neighbor pair with no direct edge in the slice, compute the FULL fused score (structural + semantic)
5. Keep top-500 by fused score

This is approximate (we only consider pairs that are structurally close), but principled: a good gap candidate should have high structural proximity (entities that "should" be related based on graph structure) AND high semantic proximity (similar surface meaning) AND no direct edge (the gap).  The FAISS top-20 query captures the structural proximity filter efficiently.

### 5.3 Procrustes alignment

Each year's Node2Vec embedding lives in its own random coordinate frame.  To compare the same entity across years, we align each year's embedding to the **previous year's** embedding via orthogonal Procrustes:

1. Find canonical_ids present in both year t-1 and year t
2. Build matrices A (year t-1 embeddings of shared IDs) and B (year t embeddings of shared IDs)
3. Compute optimal orthogonal rotation R via SVD: `R = V @ U.T` from SVD of `A.T @ B`
4. Apply R to ALL of year t's embeddings (not just shared ones)
5. Year t+1 aligns to year t (chained alignment, not all-to-earliest)

This is chained rather than global-to-earliest because each step has many shared IDs (e.g. 2023→2024 has 9,257 shared NLP IDs) and chained Procrustes is standard practice for temporal embedding alignment.  The earliest year (2018 for NLP, 2019 for COVID) is the anchor — no alignment applied.

### 5.4 SPECTER model substitution (v1 for v2)

SPECTER2 (`allenai/specter2`) was the originally planned model, but it is **not loadable as a standard HuggingFace model** — attempting to load it raises a ValueError on this infrastructure.  The fallback was SPECTER v1 (`allenai/specter`, BERT-based, 768-dim, 12 layers, ~85M params).  SPECTER v1 loads and runs successfully when CUDA is available (the semantic embeddings in this project were produced on a machine with CUDA).

SPECTER v1 produces 768-dim embeddings.  03 encodes each entity's representative `surface_form` + entity-type label through SPECTER v1, L2-normalizes, and caches to `.npy`.  The dimension (768) is read dynamically by 04 from the `.npy` shape — the fusion code doesn't hardcode the dimension.

**Why SPECTER2 was unavailable:** SPECTER2 is a specialized model not registered as a standard HF model in this environment; the HF `AutoModel`/`AutoTokenizer` path fails with a ValueError before any weights are loaded.  SPECTER v1 (`allenai/specter`) is a standard registered model and loads cleanly.

**Entity coverage:** Not all canonical IDs get semantic embeddings.  Only entities that appear in at least one relation (i.e. have at least one edge in the graph) are embedded.  Edgeless entities (canonical IDs present in the entities JSON but never appearing in any relation) are excluded from the semantic embedding matrix.  This means:

| Domain | Canonical IDs total | Embedded (edgeless excluded) |
|---|---|---|
| NLP | 12,867 | 10,611 |
| COVID | 5,500 | 3,968 |

These numbers match the 01 slice final-year node counts because the same edgeless-exclusion logic applies: the 2024 slice only contains entities that appear in relations with year ≤ 2024, which is exactly the set that gets semantic embeddings.

### 5.5 Degenerate slice handling

- **02**: If a slice has < 10 nodes, it writes a zero-matrix placeholder embedding ([1, 128]) with a warning in the id_map.  This prevents Node2Vec from crashing on empty graphs.
- **04**: If a slice has < 2 nodes, it writes an empty gaps list.  If an entity has no semantic embedding (doesn't appear in relations), it's skipped in scoring.
- **05**: If a gaps file has fewer than 10 pairs, it logs a warning but continues.  If no post-cutoff papers exist, all hit rates are 0.

### 5.6 Caching and --force

Every script writes its outputs to disk and checks for existing outputs before running.  By default, re-running skips completed stages.  `--force` redoes everything regardless of cache state.  This makes the pipeline idempotent and resumable.

### 5.7 Near-duplicate surface-form filter

Candidate gap pairs whose representative surface forms are near-identical are excluded before the top-500 selection.  Without this filter, unmerged canonicalization duplicates — e.g. "CGF" vs "CGF)", "CDA" vs "CDA/S", "ExSum" vs "ExSum)", "multistep fusion schema" vs "fusion schema" — would appear as "gaps" that are really just the same entity mentioned in two different surface forms.

**Implementation:**
- Representative surface form per canonical ID is taken from `component2_entity_relation_extraction/output/{NLP,COVID}_entities.json` (the most frequent surface form across all mentions of that canonical ID).
- Normalization: lowercase + strip all non-alphanumeric characters.
- Near-duplicate detection: `rapidfuzz.fuzz.normalized_ratio(norm(a), norm(b)) >= 0.85` (normalized Levenshtein similarity).
- Threshold: `NEAR_DUPLICATE_THRESHOLD = 0.85` (a named constant in 04_fuse_and_rank.py).
- The filter is a separate, clearly-marked function `is_near_duplicate()` called inside `_filter_near_duplicates()`, which runs before top-500 selection.

**Counts:**

The filter removed **82,954 near-duplicate pairs** across all gap generations (all domains, all years, all alphas).  After filtering, a scan of all regenerated gap JSONs confirmed **0 pairs above the 0.85 threshold** remain in any gap file.  The SPEC examples that motivated this filter — "CGF"/"CGF)", "CDA"/"CDA/S", "ExSum"/"ExSum)", "multistep fusion schema"/"fusion schema" — are all excluded.

Note: topically related but distinct pairs are not caught by this filter.  For example, COVID pairs like "crowdsource knowledge graph based approach" vs "hierarchical crowdsource knowledge graph based framework" (Lev=0.719, TSR=0.880) and "personalization-privacy paradox" vs "personalization-" (Lev=0.682, TSR=0.681) remain in the gaps — they are legitimately different concepts.

---

## 6. Component 3 methodology reuse (05)

05 reuses Component 3's retrospective validation methodology but keeps the code **standalone** — it does NOT import from `component3/`.  The relevant helpers are copied inline:

- `_surface_forms_from_entities()`: canonical_id → set of lowercase surface tokens (≥4 chars) from all entity mentions
- `_build_paper_index()`: paper_id → set of lowercase tokens (≥4 chars) from post-cutoff paper texts
- `_entity_surface_map()`: canonical_id → set of surface tokens (same as _surface_forms_from_entities)
- `check_materialization()`: returns True if both entities' surface token sets intersect with any post-cutoff paper's token set

The evaluation logic:
1. Load top-75 gaps from 04's gaps file for the given (domain, cutoff, alpha)
2. For each gap pair (u, v), check if both u and v's surface tokens co-occur in any post-cutoff paper
3. Count hits (materialized) and misses (not materialized)
4. hit_rate = hits / scored

This is identical to Component 3's `_load_component2` + `validate_with_real_data` methodology, just applied to Component 5's ranked gaps instead of Component 3's graph-based candidate gaps.

---

## 7. Alpha sweep and ablations (05)

**Alpha sweep:** For each (domain, cutoff), evaluates alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0} using the corresponding gaps file from 04.  Each entry records hit_rate, hits, misses, candidate_gaps_scored, elapsed_seconds, top_hits.

**Ablations:**
- **structural_only** — equivalent to α=1.0 (uses the α=1.0 gaps file).  Purely structural ranking, no semantic signal.
- **semantic_only** — equivalent to α=0.0 (uses the α=0.0 gaps file).  Purely semantic ranking, no structural signal.
- **random_baseline** — a random draw of 75 pairs from the candidate pool (not random entity pairs, not a shuffle of the top-75).  For each cutoff, 75 pairs are drawn uniformly at random from all valid gap candidates for that (domain, cutoff) after near-duplicate filtering.  Hit rate is computed the same way as for the ranked gaps and reported as the mean over 3 random seeds (42, 123, 456).  This is a sanity check: if the fused ranking is no better than random, the embeddings aren't capturing anything useful.

**Markdown table:** Rows = settings (α=0.0 through α=1.0, plus structural_only, semantic_only, random_baseline).  Columns = cutoff years.  Cells = hit_rate (hits/scored).  Best hit_rate per cutoff is in bold.

---

## 8. Dependency install

```bash
# Anaconda Python 3.12 (/opt/anaconda3/bin/python3)
/opt/anaconda3/bin/pip install torch transformers faiss-cpu networkx node2vec scikit-learn numpy scipy

# Or with a requirements file
cat > component5/requirements.txt << 'EOF'
torch
transformers
faiss-cpu
networkx
node2vec
scikit-learn
numpy
scipy
EOF
/opt/anaconda3/bin/pip install -r component5/requirements.txt
```

**CUDA note:** 03 uses CUDA if `torch.cuda.is_available()` is True.  All other scripts are CPU-only.  FAISS is `faiss-cpu` (no GPU indexing needed — IndexFlatIP on 12,867 vectors is fast on CPU).

**Memory note for 03:** SPECTER (85M params) may OOM on machines with < 8GB free RAM.  The script falls back to MiniLM (11M params) automatically.  If even MiniLM fails, pass `--batch-size 1` to reduce per-batch memory.

---

## 9. Definition of done checklist

- [x] All 5 scripts exist under `component5/`
- [x] 01 runs end-to-end, produces 13 gpickle slices + slices_build_log.json
- [x] 02 runs end-to-end, produces 13 structural .npy + id_map files + structural_build_log.json
- [x] 03 runs end-to-end, produces 2 semantic .npy + id_map files + semantic_build_log.json
- [x] 04 runs end-to-end, produces gaps + meta JSONs for all (domain, year, alpha) combinations
- [x] 05 runs end-to-end, produces ablation JSON + markdown + top_gaps_latest JSON per domain
- [x] Per-slice node/edge counts reconcile with component4/graph_summary.json
- [x] Structural and semantic embedding caches exist on disk
- [x] Ablation JSON + markdown table show alpha sweep results
- [x] Top-10 gaps for latest year per domain produced with scores
- [x] component5/README.md documents run order + expected outputs
- [x] Re-running skips completed stages (--force to redo)
- [x] Every script validates inputs on startup (explicit FileNotFoundError with exact path)
- [x] Degenerate slices handled with logged warnings, not crashes
- [x] All randomness seeded (numpy, random, torch, Node2Vec seed=42, FAISS deterministic)
- [x] Wall-clock time + peak memory logged per stage
- [x] Components 1–4 untouched

---

## 10. Files

```
component5/
├── 01_build_slices.py           # Step 1 — cumulative per-year DiGraph slices
├── 02_structural_embeddings.py  # Step 2 — Node2Vec 128-dim + Procrustes alignment
├── 03_semantic_embeddings.py    # Step 3 — SPECTER / MiniLM semantic embeddings
├── 04_fuse_and_rank.py          # Step 4 — FAISS-based gap ranking + alpha fusion
├── 05_evaluate.py               # Step 5 — retrospective evaluation + alpha sweep + ablations
├── README.md                    # This file
└── output/                      # All outputs written here
    ├── slices/
    ├── embeddings/
    │   ├── structural/
    │   └── semantic/
    ├── gaps/
    ├── slices_build_log.json
    ├── structural_build_log.json
    ├── semantic_build_log.json
    ├── ablation_NLP.json
    ├── ablation_NLP.md
    ├── ablation_COVID.json
    ├── ablation_COVID.md
    ├── top_gaps_NLP_latest.json
    └── top_gaps_COVID_latest.json
```

**Inputs (read-only, never modified):**
```
component2_entity_relation_extraction/output/
├── NLP_entities.json
├── NLP_relations.json
├── COVID_entities.json
└── COVID_relations.json
```
