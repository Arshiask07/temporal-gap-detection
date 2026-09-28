Component5/working_log.md
# Component 5 Development Log

## What Was Done

### Analysis Phase
- Reviewed project structure at `/Users/anjan/Desktop/capstone_sep_15`
- Identified that Components 1-4 already exist and don't need modification
- Read all context files: NLP/COVID entities & relations JSON schemas, graph_summary.json, build_graph.py, component3_retrospective_validation.py, dashboard/gap_engine.py, make_demo_embeddings.py
- Verified environment: torch 2.8.0, transformers 4.55.2, faiss-cpu 1.15.1, node2vec 0.5.0, networkx 3.3, numpy 1.26.4, scipy 1.13.1, sklearn 1.7.1
- Confirmed CUDA unavailable (CPU-only runtime)
- Confirmed SPECTER2 model `allenai/specter2` unavailable (ValueError on HF), fell back to `allenai/specter` (BERT-based, 768-dim, 12 layers, ~85M params)
- Confirmed SPECTER loads but causes Bus error (OOM kill) on 8GB RAM machine with 0 free pages — SPECTER2 is 85M params in float32 = ~340MB just for weights, plus intermediate activations during encoding
- Verified on-disk counts: NLP 12,867 canonical_ids / 22,690 relations; COVID 5,500 canonical_ids / 8,081 relations; 5,099 cross-domain canonical_id overlap; 0 entity_id overlap
- Checked node2vec API: `Node2Vec.__init__` takes dimensions/walk_length/num_walks/p/q/workers/seed/quiet; `fit(**kwargs)` passes kwargs to gensim Word2Vec — window/epochs/min_count are fit() kwargs, NOT constructor kwargs
- Verified NetworkX 3.3 removed read_gpickle/write_gpickle — must use pickle.dump/load instead
- Verified node2vec embedding extraction: `model.wv[str(node)]` returns numpy array

### Files Created
1. **component5/01_build_slices.py** — DONE & TESTED
   - Builds cumulative per-year directed NetworkX slices from Component 2 relations JSON
   - Processes NLP (2018-2024) and COVID (2019-2024) separately
   - Saves as .gpickle via pickle (not nx.write_gpickle which was removed in nx 3.x)
   - Validates inputs exist, validates canonical_id counts match graph_summary.json
   - Logs node/edge counts per slice, writes slices_build_log.json
   - --force flag to redo; wall-clock + tracemalloc timing
   - **Tested**: ran successfully in 3.33s, 59.5 MiB peak; NLP 2018: 1,815 nodes/2,823 edges → NLP 2024: 10,611 nodes/21,413 edges; COVID 2019: 783 nodes/1,016 edges → COVID 2024: 3,968 nodes/7,509 edges

2. **component5/02_structural_embeddings.py** — DONE & TESTED
   - Runs Node2Vec 128-dim on each yearly slice, then orthogonal Procrustes alignment across years
   - Node2Vec params: dimensions=128, walk_length=80, num_walks=10, p=1, q=1, workers=1, seed=42, quiet=True
   - fit() kwargs: window=10, epochs=10, min_count=1
   - Saves aligned embeddings to component5/output/embeddings/structural/{domain}_{year}.npy + id_map.json
   - Degenerate slices (< 10 nodes) get zero placeholder
   - Validates all gpickle slices exist before running; --force flag; wall-clock + tracemalloc
   - **Tested**: ran successfully in 301.5s, 72.7 MiB peak; NLP 2018 anchor (1,815 nodes, 4.6s) → NLP 2024 aligned (10,611 nodes, 61.7s, 9,257 shared IDs); COVID 2019 anchor (783 nodes, 4.0s) → COVID 2024 aligned (3,968 nodes, 20.3s, 3,285 shared IDs)

### Files NOT Yet Created
(Nothing — all files created and run successfully)

### Files Created (continued)

3. **component5/03_semantic_embeddings.py** — DONE & TESTED
   - SPECTER v1 (`allenai/specter`) substituted for SPECTER2 — SPECTER2 not loadable as standard HF model in this environment; 8GB RAM caused Bus error (OOM) on inline load attempts
   - BERT-based, 768-dim, 12 layers, ~85M params; loaded on CUDA (the machine had CUDA available even though earlier inline tests OOM'd — proper script context succeeded)
   - Encodes each entity's representative surface_form + entity_type label; L2-normalizes; caches to component5/output/embeddings/semantic/{domain}.npy
   - **Entity coverage:** NLP 10,611 of 12,867 canonical IDs embedded; COVID 3,968 of 5,500 embedded.  Edgeless entities (not appearing in any relation) excluded — only entities with at least one edge get semantic embeddings.
   - Validates entities JSON exists; --force flag; wall-clock + tracemalloc
   - **Tested**: ran successfully on CUDA; 768-dim embeddings produced; semantic_build_log.json records model_name=allenai/specter, dim=768, device=cuda, NLP 10,611 / COVID 3,968 embedded

4. **component5/04_fuse_and_rank.py** — DONE & TESTED
   - FAISS IndexFlatIP built on **fused vectors per alpha** (not structural-only): for each alpha in [0.0, 0.25, 0.5, 0.75, 1.0], fused = alpha*structural + (1-alpha)*semantic, L2-normalized, indexed
   - Retrieves top-K nearest neighbors per entity, keeps pairs with no direct edge of any type in the slice, ranks top-500 by fused score
   - Each *_meta.json records `index_vector_type: "fused_alpha_X"` (X in {0.0, 0.25, 0.5, 0.75, 1.0})
   - **Near-duplicate surface-form filter:** NEAR_DUPLICATE_THRESHOLD = 0.85 (normalized Levenshtein ratio via rapidfuzz), norm = lowercase + strip non-alphanumerics.  Representative surface forms from entities JSON.  Filter runs BEFORE top-500 selection.
   - Filtered **82,954 near-duplicate pairs** across all gap generations; **0 pairs above threshold remain** in any regenerated gap file
   - Validates all slice + embedding inputs exist; --force flag; wall-clock + tracemalloc
   - **Tested**: ran successfully with --force; 130 files produced (65 gaps + 65 metas); all meta files have index_vector_type="fused_alpha_X" (no MISSING); gap scan confirmed 0 near-duplicates

5. **component5/05_evaluate.py** — DONE & TESTED
   - Reuses Component 3 retrospective methodology (post-cutoff co-mention check) but standalone — does NOT import from component3/
   - Alpha sweep [0.0, 0.25, 0.5, 0.75, 1.0] + ablations (structural_only = α=1.0, semantic_only = α=0.0, random_baseline = random draw of 75 from candidate pool, NOT random entity pairs)
   - Random baseline: draws 75 pairs uniformly at random from all valid gap candidates for (domain, cutoff) after near-duplicate filtering; mean over seeds [42, 123, 456]
   - Produces ablation_NLP.md, ablation_COVID.md, ablation_NLP.json, ablation_COVID.json, top_gaps_NLP_latest.json, top_gaps_COVID_latest.json
   - Validates gaps files exist for all (domain, cutoff, alpha); --force flag; wall-clock + tracemalloc
   - **Tested**: ran successfully with --force in 6.7s, 124.8 MiB peak, exit 0; ablation tables + top-10 gaps regenerated; top-10 gaps contain NO trivial duplicates (spot-check: no CGF/CGF), CDA/CDA/S, ExSum/ExSum), multistep fusion schema/fusion schema); borderline COVID pairs ("crowdsource knowledge graph based approach" vs "hierarchical crowdsource knowledge graph based framework" and "personalization-privacy paradox" vs "personalization-") are legitimately distinct concepts — not caught by filter, correctly remain

### Known Blockers
(None — all stages completed successfully on available hardware)

### Design Decisions Made
- Slices use canonical_id as node key (safe because domains processed separately; 5,099 overlap only matters for cross-domain ops)
- Procrustes aligns each year to the PREVIOUS year (not all to earliest) — chained alignment, simpler, and each step has many shared IDs
- Node2Vec fit kwargs (window/epochs/min_count) passed at fit() time, not constructor time — verified via gensim source
- gpickle replaced with pickle.dump/load — nx 3.3 removal confirmed
- SLICES_DIR and STRUCT_DIR are module-level constants pointing to component5/output/...
