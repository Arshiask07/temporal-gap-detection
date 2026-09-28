Component6/working_log.md
# Component 6 Development Log

## What Was Done

### Analysis Phase

- Read all context files: dashboard/gap_engine.py (reference velocity implementation), dashboard/config.py (VEL_WINDOW=(2022,2024)), data_collection/scripts/02_collect_semantic_scholar.py (S2 API client), component5/05_evaluate.py (retrospective eval methodology), component5/README.md + working_log.md (pipeline conventions)
- Verified the critical data reality: no citation counts exist anywhere in the corpus or Component 2 outputs — the S2 enrichment step (data_collection step 2a) was never run
- Confirmed entity→paper→year mapping is available in component2_entity_relation_extraction/output/{NLP,COVID}_entities.json (each entity mention record carries paper_id + year)
- Confirmed paper-level records (title, abstract) available in component2_entity_relation_extraction/output/{NLP,COVID}_extracted.json
- Reviewed gap file schema: entity_a, entity_b, surface_form_a, surface_form_b, structural_score, semantic_score, fused_score, alpha, year
- Confirmed 130 gap files exist (65 per domain × 5 alphas × varying years; NLP 2018-2024, COVID 2019-2024)
- Environment: requests available (stdlib json, numpy available). No torch/transformers/faiss needed for Component 6.
- S2 API: optionally key via S2_API_KEY env var; rate-limit ~1 req/s; search by title with limit=1 returns paperId + citationCount

### Design Decisions

- Two citation paths: primary = S2 API citation counts; fallback = mention velocity (papers-per-year from entities.json). Fallback triggered by FALLBACK.txt flag, never silently.
- Velocity formula: vel(e) = (c_e[y2] − c_e[y1]) / (y2 − y1), window (2022, 2024) from dashboard/config.py
- Priority formula: priority = fused_score × mean(vel_u, vel_v). Preserves all original gap fields + vel_u, vel_v, priority.
- min_vel filter: clamps velocities below threshold to 0 before computing mean. Default 0.0 preserves negative velocities (penalizes stagnant/declining entities).
- Re-ranking preserves top-500 cap per file, sorted by priority descending.
- 04_evaluate.py is standalone — copies retrospective eval helpers inline, no imports from component5/ or component3/.
- Three-way comparison per (domain, cutoff): fused_only (C5 alpha=0.5 baseline), velocity (C6 priority-sorted), random_baseline (random draw of 75 from candidate pool, mean over seeds [42, 123, 456]).
- Random baseline pool = velocity gaps if available, else fused gaps.
- Every script follows C5 conventions: argparse CLI, --force, FileNotFoundError with exact path, JSON build logs, wall-clock + tracemalloc, seed 42.
- NLP and COVID processed separately; canonical_id is the entity key.

### Files Created

1. **component6/01_fetch_citations.py** — written, syntax OK
   - Extracts unique paper_ids from entities.json, looks up citationCount via S2 search API (title match, limit=1)
   - Caches to component6/output/citations/{domain}_paper_citations.json
   - API probe on startup; if unreachable, writes empty caches + FALLBACK.txt flags and exits 0
   - --force to refetch; rate-limit 1s (3.5s without key); handles 429 with 30s backoff
   - Input validation: raises FileNotFoundError if entities.json or extracted.json missing

2. **component6/02_entity_velocity.py** — written, syntax OK
   - Builds entity×year aggregate matrix: sum of citation counts per (entity, year) from citation cache, OR count of papers per (entity, year) from entities.json (mention velocity)
   - Computes vel(e) = (c_e[y2] − c_e[y1]) / (y2 − y1) for window (2022, 2024) by default, configurable via --vel-window
   - Outputs component6/output/entity_velocity_{domain}.json + build logs with coverage stats
   - Detects FALLBACK.txt flag → uses mention velocity; logs loudly which path was used
   - --force to redo; --domain selector; --vel-window Y1 Y2

3. **component6/03_rerank.py** — written, syntax OK
   - Joins velocities onto every Component 5 gap file; computes priority = fused_score × mean(vel_u, vel_v)
   - min_vel clamp: velocities < min_vel → 0 before mean computation
   - Writes reranked/{DOMAIN}_{YEAR}_alpha{ALPHA}.json preserving all original fields + vel_u, vel_v, priority
   - Sorted by priority descending, capped at top-500
   - --force, --domain, --year, --alpha, --min-vel selectors
   - Input validation: raises FileNotFoundError if entity velocity JSON missing

4. **component6/04_evaluate.py** — written, syntax OK
   - Standalone retrospective eval: compares fused_only (C5 alpha=0.5) vs velocity (C6 priority) vs random_baseline per (domain, cutoff)
   - Uses post-cutoff co-mention check (same methodology as C5/05 and Component 3); copied inline, no imports
   - Cutoffs: NLP [2021, 2022, 2023], COVID [2020, 2021, 2022]; TOP_K_EVAL=75; seeds [42, 123, 456]
   - Outputs ablation_velocity_{domain}.json + .md, top_velocity_gaps_{domain}_latest.json
   - Ablation markdown: hit-rate table (fused_only, velocity, random_baseline × cutoffs), top-10 velocity gaps with fused/vel_u/vel_v/priority columns
   - Records citation path (API vs mention-velocity) in markdown + build logs
   - --force, --domain, --cutoff selectors
   - Input validation: raises FileNotFoundError if fused or velocity gaps files missing

### Pending / Not Yet Run

- 01_fetch_citations.py — pending (requires network access to api.semanticscholar.org; may fall back to mention velocity)
- 02_entity_velocity.py — pending (depends on 01 output)
- 03_rerank.py — pending (depends on 02 output)
- 04_evaluate.py — pending (depends on 03 output)
- README.md — written
- working_log.md — this file

### Blockers

- S2 API reachability: api.semanticscholar.org may be blocked in some environments (the original data_collection script noted HTTP 403 in a sandbox). If blocked, the fallback path (mention velocity) is fully functional and the pipeline proceeds — just with a weaker signal.
- S2_API_KEY env var: not set by default; without it, rate limit is ~3.5s per request. With NLP ~10,000+ unique papers, this could take a long time. COVID has fewer papers. A key is strongly recommended for the primary path.
- No citation counts exist anywhere — the S2 enrichment was never run, so 01 is a one-time cost to populate the citation cache.

### Verification of read-only constraint

- component6/ scripts only READ from: component2_entity_relation_extraction/output/, component5/output/gaps/
- component6/ scripts only WRITE to: component6/output/
- No modifications to components 1-5, dashboard/, or data_collection/ — verified by checking that all paths in component6/ scripts point to component6/output/ for writes and component2/component5/ for reads
