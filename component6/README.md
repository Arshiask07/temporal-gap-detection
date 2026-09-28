# Component 6 — Citation Velocity Re-ranking Engine

**Status:** Implemented (4 scripts + README).  Ready to run end-to-end.

**Folder:** `component6/`  
**Output:** `component6/output/`  
**Dependencies:** requests, numpy, json (stdlib) — no torch/transformers/faiss needed  
**GPU:** Not required.  All scripts are CPU-only.  
**Network:** Only `01_fetch_citations.py` makes external calls (Semantic Scholar API).  
**Neo4j:** Not required.  
**Components 1–5:** Untouched.  Component 6 reads from them (read-only) and writes only to `component6/output/`.

---

## 1. Overview

Component 6 takes Component 5's ranked research gaps and re-ranks them by **citation velocity** — how fast the literature is growing around each entity.  A gap connecting two fast-growing entities gets a higher priority, surfacing the "emerging" research opportunities.

**Pipeline:**

```
Component 5 gaps  ──→ 01_fetch_citations  ──→ per-paper citation counts (S2 API)
                         │
                         └─→ [fallback] mention velocity from entities.json
                                      │
02_entity_velocity  ←─────────────────┘   entity×year velocity matrix
                                      │
03_rerank            ←─────────────────┘   priority = fused_score × mean(vel_u, vel_v)
                                      │
04_evaluate          ←─────────────────┘   retrospective eval: fused-only vs velocity vs random
```

**Velocity formula:**

```
vel(e) = (c_e[y2] − c_e[y1]) / (y2 − y1)
```

where `(y1, y2) = VEL_WINDOW = (2022, 2024)` (matching `dashboard/config.py`).

`c_e[y]` = sum of citation counts of papers mentioning entity `e` in year `y` (citation path)  
or = count of papers mentioning `e` in year `y` (mention-velocity fallback).

**Priority formula:**

```
priority = fused_score × mean(vel_u, vel_v)
```

Pairs where both entities have low velocity get `priority ≈ 0` and sink to the bottom.  A `--min-vel` filter clamps velocities below the threshold to 0 before computing priority.

---

## 2. Citation acquisition — two paths

**Critical data reality:** No citation counts exist anywhere in the corpus or Component 2 outputs.  The Semantic Scholar enrichment step (data_collection step 2a) was never run.

### Primary path: Semantic Scholar API

`01_fetch_citations.py` extracts unique `paper_id`s from `component2/entities.json`, looks up each paper's `citationCount` via the S2 search API (title match, `limit=1`), and caches the result to `component6/output/citations/{domain}_paper_citations.json`.

- API key: `S2_API_KEY` env var (optional — raises rate limit from ~100 req/5min to 1 req/s sustained)
- Rate limit: ~1 req/s (3.5s delay without key)
- Cache: one-time cost per domain; `--force` to refetch

### Fallback path: mention velocity

If the S2 API is unreachable or blocked, `01_fetch_citations.py` writes empty citation caches + a `FALLBACK.txt` flag and exits 0.  The rest of the pipeline detects the flag and uses **mention velocity** instead: `c_e[y] = count of papers mentioning entity e in year y`, computed entirely from Component 2's `entities.json` (which carries `paper_id` and `year` per entity mention).

Mention velocity is a weaker signal than citation velocity (it measures publication volume, not impact), but it is fully reproducible offline and never requires network access.

**Never silently substitute fake data.**  If neither path works, the pipeline stops with a clear blocker report instead of proceeding with fabricated inputs.

---

## 3. Run order

Run the scripts in order 1 → 4.  Each script validates its inputs on startup and raises explicit `FileNotFoundError` with the exact missing path if a prerequisite is missing.  Re-running skips completed stages unless `--force` is passed.

```bash
# All commands run from /Users/anjan/Desktop/capstone_sep_15/

# Step 1: Fetch citation counts from S2 API (or set up fallback)
python3 component6/01_fetch_citations.py --force

# Step 2: Compute entity velocities from citation cache (or mention counts)
python3 component6/02_entity_velocity.py --force

# Step 3: Re-rank every Component 5 gap file by velocity
python3 component6/03_rerank.py --force

# Step 4: Retrospective evaluation — fused-only vs velocity vs random
python3 component6/04_evaluate.py --force
```

### Selective re-runs

```bash
# Re-fetch citations for NLP only
python3 component6/01_fetch_citations.py --domain NLP --force

# Re-compute velocities with a custom window
python3 component6/02_entity_velocity.py --vel-window 2021 2024 --force

# Re-rank only NLP 2024 alpha=0.5
python3 component6/03_rerank.py --domain NLP --year 2024 --alpha 0.5 --force

# Re-evaluate only COVID
python3 component6/04_evaluate.py --domain COVID --force
```

### Flags

| Flag | Scripts | Meaning |
|---|---|---|
| `--force` | all | Redo work even if output cache exists |
| `--domain` | 01, 02, 03, 04 | Restrict to NLP or COVID (default: both) |
| `--year` | 03 | Restrict to one year (default: all years for domain) |
| `--alpha` | 03 | Single alpha value (default: all ALPHAS [0.0, 0.25, 0.5, 0.75, 1.0]) |
| `--min-vel` | 03 | Minimum velocity threshold (default: 0.0) |
| `--vel-window Y1 Y2` | 02 | Velocity window start/end (default: 2022 2024) |

---

## 4. Expected outputs

After a full clean run (01–04, `--force` on each):

```
component6/output/
├── citations/
│   ├── NLP_paper_citations.json              # 01: {paper_id: citation_count}
│   ├── NLP_FALLBACK.txt                      # 01: present iff API unreachable
│   ├── COVID_paper_citations.json
│   └── COVID_FALLBACK.txt
├── entity_velocity_NLP.json                  # 02: {canonical_id: velocity, ...}
├── entity_velocity_NLP_build_log.json        # 02: coverage stats + timing
├── entity_velocity_COVID.json
├── entity_velocity_COVID_build_log.json
├── reranked/
│   ├── NLP_2018_alpha0.0.json ...           # 03: top-500 priority-sorted per (domain, year, alpha)
│   ├── NLP_2018_alpha0.0_meta.json
│   ├── NLP_2018_alpha0.25.json ...
│   ├── NLP_2018_alpha0.5.json  (primary)
│   ├── ... (same for 2019–2024, NLP + COVID)
│   └── COVID_2024_alpha0.5.json
├── ablation_velocity_NLP.json                # 04: {cutoff: {fused_only, velocity, random_baseline}}
├── ablation_velocity_NLP.md                 # 04: markdown table
├── ablation_velocity_COVID.json
├── ablation_velocity_COVID.md
├── top_velocity_gaps_NLP_latest.json         # 04: top-10 velocity gaps for 2024
└── top_velocity_gaps_COVID_latest.json
```

### Counts reconciliation

Entity velocities are computed per-domain separately.  The entity set is the set of canonical_ids present in `entities.json`.  Not all canonical_ids have velocities — entities that never appear in any paper within the velocity window have velocity 0.0 and are included in the output for completeness.

---

## 5. Design decisions

### 5.1 Velocity window

The velocity window `(2022, 2024)` is taken from `dashboard/config.py` (`VEL_WINDOW = (2022, 2024)`).  This is the same window used by the dashboard's gap engine for consistency.  It is configurable via `--vel-window` in `02_entity_velocity.py`.

### 5.2 Priority formula

`priority = fused_score × mean(vel_u, vel_v)` — the fused score (Component 5's ranking signal) is the base, and velocity modulates it.  A high-fused-score gap between two stagnant entities still ranks reasonably; a moderate-fused-score gap between two rapidly growing entities gets boosted.  This is the "emerging detector" semantics: it surfaces gaps the literature is actively accelerating toward.

### 5.3 min_vel filter

`--min-vel` (default 0.0) clamps entity velocities below the threshold to 0 before computing `mean(vel_u, vel_v)`.  With the default of 0.0, negative velocities are preserved — they actively penalize the priority (an entity losing citations gets `mean(vel_u, vel_v) < 1`, dragging priority below `fused_score`).  Raising `--min-vel` effectively filters out slow-moving entities.

### 5.4 Re-ranking preserves all original fields

Every re-ranked gap file preserves all fields from Component 5's gap file (`entity_a`, `entity_b`, `surface_form_a`, `surface_form_b`, `type_a`, `type_b`, `structural_score`, `semantic_score`, `fused_score`, `alpha`, `year`) plus `vel_u`, `vel_v`, `priority`.  Downstream consumers can compare structural/semantic/fused scores side-by-side with velocity.

### 5.5 Retrospective evaluation methodology

`04_evaluate.py` is **standalone** — it does NOT import from `component5/` or `component3/`.  The post-cutoff co-mention check (Component 3 methodology) is copied inline:

1. Load top-75 gaps from the velocity file for the given (domain, cutoff)
2. For each gap pair (u, v), check if both u and v's surface tokens co-occur in any post-cutoff paper
3. Count hits (materialized) and misses (not materialized)
4. `hit_rate = hits / scored`

Three settings are compared per (domain, cutoff):
- **fused_only** — Component 5's alpha=0.5 gaps (baseline)
- **velocity** — Component 6's priority-sorted gaps
- **random_baseline** — random draw of 75 from the candidate pool, mean over seeds [42, 123, 456]

### 5.6 Citation path reporting

Every output file records which citation path was used.  The ablation markdown and build logs state either "Semantic Scholar API" or "mention-velocity fallback (S2 API unreachable)".  The `FALLBACK.txt` files are the single source of truth for which path was active — if they exist, the fallback was used.

### 5.7 No network except 01

Only `01_fetch_citations.py` makes external calls.  Steps 02–04 are fully offline, reading from cached JSON files.  This makes the pipeline reproducible and resumable without network access after the initial citation fetch.

---

## 6. Relationship to Component 5

- **Input:** Component 5's gap files (`component5/output/gaps/{domain}_{year}_alpha{alpha}.json`) — read-only
- **Output:** Re-ranked gap files in `component6/output/reranked/` — same schema + velocity fields
- **Evaluation:** Component 6's `04_evaluate.py` compares its velocity-reranked gaps against Component 5's fused-only baseline using the same retrospective methodology
- **No modification:** Component 5's outputs are never modified by Component 6

---

## 7. Dependency install

```bash
# Standard library + requests only
/opt/anaconda3/bin/pip install requests
```

No torch/transformers/faiss/networkx/node2vec needed — those are Component 5's dependencies.

---

## 8. Definition of done checklist

- [ ] All 4 scripts exist under `component6/`
- [ ] 01 runs end-to-end, produces citation caches (or fallback flags)
- [ ] 02 runs end-to-end, produces entity_velocity_{NLP,COVID}.json + build logs
- [ ] 03 runs end-to-end, produces 130 re-ranked files in reranked/
- [ ] 04 runs end-to-end, produces ablation_velocity_{NLP,COVID}.json + .md + top_velocity_gaps_latest.json
- [ ] README documents run order, flags, velocity + priority formulas, citation path
- [ ] working_log.md records environment check, files created, test results, blockers
- [ ] Every script validates inputs (explicit FileNotFoundError with exact path)
- [ ] --force flag works on every script
- [ ] JSON build logs + wall-clock + tracemalloc timing per script
- [ ] NLP and COVID processed separately; canonical_id is the entity key
- [ ] Citation path (API vs mention-velocity fallback) clearly logged everywhere
- [ ] Nothing outside component6/ was modified
- [ ] Re-running skips completed stages (--force to redo)

---

## 9. Files

```
component6/
├── 01_fetch_citations.py       # Step 1 — S2 API citation fetch (or fallback setup)
├── 02_entity_velocity.py       # Step 2 — entity×year velocity matrix
├── 03_rerank.py                # Step 3 — priority re-ranking of all gap files
├── 04_evaluate.py              # Step 4 — retrospective eval: fused vs velocity vs random
├── README.md                   # This file
└── output/                     # All outputs written here
    ├── citations/
    ├── entity_velocity_{NLP,COVID}.json
    ├── entity_velocity_{NLP,COVID}_build_log.json
    ├── reranked/
    ├── ablation_velocity_{NLP,COVID}.json
    ├── ablation_velocity_{NLP,COVID}.md
    ├── top_velocity_gaps_{NLP,COVID}_latest.json
```

**Inputs (read-only, never modified):**

```
component5/output/gaps/{domain}_{year}_alpha{alpha}.json          # from 04_fuse_and_rank.py
component2_entity_relation_extraction/output/{domain}_entities.json
component2_entity_relation_extraction/output/{domain}_extracted.json
```
