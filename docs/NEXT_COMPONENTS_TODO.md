# Next Components — To-Do List

**Project:** Temporal Knowledge Graph Embedding for Emerging Research Gap Detection
**As of:** September 2026
**Status:** Components 1, 2, 3, 7 done. Components 4–6 partially done (architecture ready, data placeholders only).

---

## Critical context

- All project deps live in **Anaconda Python 3.12** at `/opt/anaconda3/bin/python3` (Homebrew Python 3.14 has NO project deps).
- Always run project scripts with `/opt/anaconda3/bin/python3`.
- Output path for real embedding/citation CSVs: `/Users/anjan/Desktop/scam/dashboard/exports/`
- Dashboard auto-picks up CSVs named `{node2vec,specter2,citations}_{nlp,covid}[_year].csv` from `exports/` — no code changes needed once CSVs land.

---

## Phase A — Real embeddings (Components 5 + 4, Contributions 1 + 4)

### ☐ A1. Train real Node2Vec per year (Component 5 — Channel 1)

**Goal:** Replace synthetic `node2vec_*.csv` files in `dashboard/exports/` with real 128-dim embeddings trained on each year's entity co-occurrence graph.

**Tasks:**
- [ ] Write `train_node2vec.py` (lives in `dashboard/` next to `make_demo_embeddings.py`)
  - Input: `entities_nlp.csv` / `entities_covid-19.csv` + `edges_nlp.csv` / `edges_covid-19.csv` from `exports/`
  - Build `nx.Graph` per year (use existing `gap_engine.build_graphs()`)
  - Train `gensim.models.Word2Vec` with `gensim.models.Node2Vec`:
    - dimensions=128, walk_length=80, num_walks=10, window=10, workers=4, seed=42
    - epochs=5
  - For each (domain, year), export CSV: `entity_id, dim_0, dim_1, ..., dim_127`
  - Skip years where graph has <2 nodes
- [ ] Run for both domains, all years (NLP: 2018–2024 = 7 years; COVID: 2019–2024 = 6 years)
- [ ] Verify output: 13 CSVs total, ~15K rows each, 128-dim
- [ ] Spot-check 2–3 entities: embeddings for "transformer" in 2018 vs 2024 should be visibly different

**File to create:** `dashboard/train_node2vec.py`
**Command:** `/opt/anaconda3/bin/python3 dashboard/train_node2vec.py --domain all`
**Time est.:** 2–4 hrs (incl. testing)

---

### ☐ A2. Download SPECTER2 + encode per-entity-per-year (Component 5 — Channel 2)

**Goal:** Real 768-dim SPECTER2 vectors per (entity, year), exported as CSV in same format as Node2Vec (downsampled to 128-dim or kept at full dim — see note).

**Tasks:**
- [ ] Download `allenai/specter2_base` + proximity adapter from HuggingFace
  - If S3/HF blocked (was blocked before), try `huggingface-cli download` with token, OR use `transformers` with `mirror` env var, OR fallback: SPECTER (v1, single model, no adapter)
  - ~5GB total
- [ ] Write `encode_specter2.py`
  - Input: paper abstracts from `component2_entity_relation_extraction/output/{NLP,COVID}_extracted.json`
  - Build paper_id → year map
  - For each entity: collect all (paper_id, abstract) pairs where this entity was extracted
  - Group by year
  - Encode abstracts with SPECTER2 (batched, GPU if present, else CPU with `batch_size=8`)
  - Average paper vectors → per-entity-per-year vector
  - Export CSV: `entity_id, dim_0, ..., dim_{N-1}` (768-dim native; or truncate/pca to 128-dim to match Node2Vec — see A3)
  - File naming: `specter2_{nlp,covid}_{year}.csv`
- [ ] Run for both domains, all years
- [ ] Verify: entities with more papers have tighter per-year averages (sanity)

**File to create:** `dashboard/encode_specter2.py`
**Command:** `/opt/anaconda3/bin/python3 dashboard/encode_specter2.py --domain all`
**Time est.:** 4–6 hrs (incl. model download)

---

### ☐ A3. Embedding-dim harmonization decision

SPECTER2 is 768-dim native; Node2Vec is 128-dim. Two options:
- [ ] **Option A (simpler):** Train Node2Vec at 768-dim to match. One change in `A1`.
- [ ] **Option B (cleaner):** Keep Node2Vec at 128, project SPECTER2 → 128 via PCA. Add `reduce_specter2.py` helper.
- [ ] **Decision:** Pick one (recommend Option A for time savings). Update `config.EMB_DIM` to match.

---

### ☐ A4. Run α-sweep ablation (Component 5 / Contribution 4)

**Goal:** Score gaps with fused vs single-channel at α ∈ {0.3, 0.5, 0.7} to validate the dual-channel fusion contribution.

**Tasks:**
- [ ] Write `run_alpha_ablation.py`
  - Calls `gap_engine.rank_gaps()` at each α
  - Reports: top-K gaps, hit rate via `retrospective_validate()`, mean gap score
- [ ] Run for both domains at cutoff=2021
- [ ] Write `ablation_results.md` to `dashboard/exports/`
  - Table: domain | α=0.3 hit% | α=0.5 hit% | α=0.7 hit% | n2v-only hit% | sp2-only hit%
- [ ] Compare fused vs single-channel — assert fusion > either alone (or document why not)

**File to create:** `dashboard/run_alpha_ablation.py`
**Command:** `/opt/anaconda3/bin/python3 dashboard/run_alpha_ablation.py`
**Time est.:** 30 min

---

## Phase B — Real citations + UI polish (Component 6, Contribution 2)

### ☐ B1. Fetch real citation counts via Semantic Scholar API (Component 6)

**Goal:** Replace synthetic `citations_nlp.csv` / `citations_covid-19.csv` with real per-year citation counts.

**Tasks:**
- [ ] Test if S2 API is now accessible (was blocked in earlier sessions):
  ```bash
  curl -I 'https://api.semanticscholar.org/graph/v1/paper/PMID:1234?fields=citationCount'
  ```
- [ ] If accessible:
  - Write `fetch_citations.py`
  - For each paper_id in `data_collection/data/{nlp,covid}/**/*.json`, fetch per-year citation counts (S2 endpoint: `/paper/{id}/citations?fields=year,citationCount`)
  - Aggregate: for each entity, sum citations across its associated papers per year
  - Export: `entity_id, year, citations` → `exports/citations_{nlp,covid}.csv`
- [ ] If still blocked: fallback to OpenAlex API (`api.openalex.org`) — similar schema, fewer rate limits
- [ ] Run for both domains
- [ ] Verify: velocity formula `vel(u) = (cit[2024] − cit[2022]) / 2` produces sensible values (most >0, some <0 for declining concepts)

**File to create:** `dashboard/fetch_citations.py`
**Time est.:** 2–4 hrs (or fallback path)

---

### ☐ B2. Add supporting paper abstracts to dashboard Tab 2 (UI completeness)

**Goal:** Fix the one UI gap noted in the spec — show which paper abstracts support each ranked gap.

**Tasks:**
- [ ] In `dashboard/app.py` Tab 2 ("Top Gaps"):
  - For each ranked gap, look up all papers mentioning both concepts (use `_build_paper_index_post_cutoff`-style logic)
  - Display: title, year, abstract preview (first 200 chars + "…")
  - Add an expander: "📄 Supporting papers (N)"
- [ ] Test in `streamlit run app.py`

**File to modify:** `dashboard/app.py`
**Time est.:** 1 hr

---

## Phase C — Final write-up + polish

### ☐ C1. Update COMPREHENSIVE_COMPONENT1_3_REPORT.md with final numbers

**Tasks:**
- [ ] After A4 + B1 done: add Component 5/6 results section
  - Node2Vec training stats (graph sizes per year, training time)
  - SPECTER2 encoding stats (entities encoded per year)
  - α-ablation table (from A4)
  - Citation velocity stats (mean/median per domain)
- [ ] Update "Status by Docx Component" tables to reflect Components 5/6 as ✅ DONE
- [ ] Update "What Remains" → all checked off

**File to modify:** `COMPREHENSIVE_COMPONENT1_3_REPORT.md`
**Time est.:** 1 hr

---

### ☐ C2. Update PROJECT_PLAN.md "What Remains" section

**Tasks:**
- [ ] Tick off items 1–5 in "Immediate (before final submission)"
- [ ] Update Component 5/6 status from ⚠️/❌ to ✅
- [ ] Add "Phase A/B/C completed YYYY-MM-DD" entry

**File to modify:** `PROJECT_PLAN.md`
**Time est.:** 15 min

---

### ☐ C3. (Optional) Refresh landing page

**Tasks:**
- [ ] Update `emergent_landing.html` with real Component 5/6 numbers
  - Replace any "demo" / "synthetic" copy
  - Add α-ablation table or chart
- [ ] Verify in browser

**File to modify:** `emergent_landing.html`
**Time est.:** 30 min

---

## Optional / Deferred

- [ ] **D1.** Neo4j migration (Component 4 deviation) — only if required for final submission. At current scale (~15K entities, ~30K relations), in-memory NetworkX is sufficient.
- [ ] **D2.** Extend corpus to 2025+ data — noted as future work in spec; needs citation accumulation window first.
- [ ] **D3.** Make entity IDs human-readable in Component 3 reports (10-line patch to `_write_summary()` to show `surface_form` alongside `canon_*` IDs).
- [ ] **D4.** Learned α (instead of fixed {0.3, 0.5, 0.7}) — explicitly deferred in spec to future work.

---

## Suggested execution order

```
A1 (Node2Vec) → A2 (SPECTER2) → A3 (dim harmonization, before A4) → A4 (α-sweep)
                                                                      ↓
                                              B1 (citations) → B2 (UI) → C1 (report) → C2 (plan) → C3 (landing)
```

**Total time est.:** ~12–18 hrs of work.

**Biggest blockers:**
- A2: SPECTER2 model download (network access — may need retry/vpn/mirror)
- B1: S2 API access (was blocked — may still be)
- Both have fallbacks so neither is fatal.

---

## Definition of done

All of the following must be true before considering the project complete:

- [ ] `dashboard/exports/node2vec_{nlp,covid}_{year}.csv` are real (not from `make_demo_embeddings.py`)
- [ ] `dashboard/exports/specter2_{nlp,covid}_{year}.csv` are real
- [ ] `dashboard/exports/citations_{nlp,covid}.csv` are real
- [ ] α-ablation table written and shows fusion > single-channel OR explains why not
- [ ] Dashboard Tab 2 shows supporting abstracts per gap
- [ ] `COMPREHENSIVE_COMPONENT1_3_REPORT.md` and `PROJECT_PLAN.md` updated
- [ ] Dashboard runs end-to-end with real data: `streamlit run dashboard/app.py` works, all 4 tabs functional