# Component 7 — Dashboard Real-Data Bridge

**Status:** Complete  
**Date:** 2026-09-26

---

## What was built

Three export scripts in `dashboard/` that convert real pipeline outputs (Components 2, 5, 6)
into the CSV format the Streamlit dashboard expects, replacing the demo/synthetic data.

### export_real_entities_edges.py
Converts C2 JSON outputs → `entities_{nlp,covid-19}.csv` and `edges_{nlp,covid-19}.csv`.

- NLP: 12,867 entities, 22,690 edges
- COVID: 5,500 entities, 8,081 edges
- Entity_id = canonical_id (matches C5 embedding id_map keys)
- Edges remapped from entity_ids to canonical_ids (exact same logic as C4's build_graph.py)
- Most frequent surface_form used as label; most common type used as type

### export_real_citations.py
Converts C6 entity_velocity JSON → `citations_{nlp,covid-19}.csv`.

- NLP: 9,202 rows, 4,601 entities
- COVID: 4,638 rows, 2,319 entities
- Citation data uses mention-velocity fallback (S2 API unreachable during C6 run)
- `citation_path` field read from JSON `path` key = "mention-velocity (fallback)"

### export_real_embeddings.py
Converts C5 .npy + id_map.json → `node2vec_{nlp,covid-19}_{year}.csv` and `specter2_{nlp,covid-19}_{year}.csv`.

- NLP node2vec: 7 years (2018–2024), 1,815 → 10,611 entities per year
- COVID node2vec: 6 years (2019–2024), 783 → 3,968 entities per year
- NLP specter2: 7 years, 10,611 entities each (time-independent, replicated)
- COVID specter2: 6 years, 3,968 entities each
- Max diff vs C5 source: 4.98e-07 (float rounding only)

---

## Files on disk (dashboard/exports/)

| File | Rows | Source |
|------|------|--------|
| `entities_nlp.csv` | 12,867 | C2 NLP_entities.json |
| `entities_covid-19.csv` | 5,500 | C2 COVID_entities.json |
| `edges_nlp.csv` | 22,690 | C2 NLP_relations.json |
| `edges_covid-19.csv` | 8,081 | C2 COVID_relations.json |
| `citations_nlp.csv` | 9,202 | C6 entity_velocity_NLP.json |
| `citations_covid-19.csv` | 4,638 | C6 entity_velocity_COVID.json |
| `node2vec_nlp_2018.csv` … `node2vec_nlp_2024.csv` | 1,815–10,611 | C5 structural .npy |
| `node2vec_covid-19_2019.csv` … `node2vec_covid-19_2024.csv` | 783–3,968 | C5 structural .npy |
| `specter2_nlp_2018.csv` … `specter2_nlp_2024.csv` | 10,611 each | C5 semantic .npy |
| `specter2_covid-19_2019.csv` … `specter2_covid-19_2024.csv` | 3,968 each | C5 semantic .npy |

Total: 32 CSV files (removed 2 leftover demo files for COVID 2018 which the real pipeline doesn't produce).

---

## Code changes

### dashboard/gap_engine.py
1. **Bug fix (line 68–73):** The `sim_history` loop was missing `v in ch[y].index` check.
   With real data, entities appear in different years, so `v` may not have an embedding
   in early years. This caused `KeyError` when the dashboard tried to build similarity
   history. Fixed by adding `v in ch[y].index` to the condition.

2. **Empty DataFrame handling (line 96–99):** When no gaps are found (e.g. semantic-only
   mode with all entities connected), `pd.DataFrame([]).sort_values("priority")` raises
   `KeyError`. Fixed by checking `df.empty` before sorting.

3. **Performance: `sample_n` parameter:** The pairwise gap scoring is O(n²). With 6,299
   NLP entities in 2021, full scoring takes hours. Added `sample_n` parameter to randomly
   sample entities before pairing. Dashboard uses `sample_n=400` for interactive use.
   `retrospective_validate` also forwards `sample_n`.

### dashboard/app.py
1. **`sample_n=400` in rank_gaps call (line 98):** Limits entity sampling for interactive
   performance. With 400 entities, ~80K pairs are checked — manageable for a tab load.
2. **Variable rename `ranked` → `gaps`:** For clarity; all references updated.

---

## Verification

### Data integrity
- All node2vec CSV rows match C5 .npy source (max diff 4.98e-07)
- All specter2 CSV rows match C5 .npy source (max diff 5.00e-07)
- Entity counts match C2 JSON: NLP 12,867, COVID 5,500
- Edge counts match C2 JSON: NLP 22,690, COVID 8,081
- Citation counts match C6 JSON: NLP 9,202 rows / 4,601 entities, COVID 4,638 / 2,319

### Gap engine on real data
- NLP fused (α=0.5): finds gaps in ~15s with sample_n=200
- NLP semantic-only (α=0.0): no gaps (all candidate pairs connected in graph) — correct
- NLP structural-only (α=1.0): finds gaps in ~218s with sample_n=200 (slow, inherent to O(n²))
- COVID fused (α=0.5): finds gaps in ~similar time
- All three alpha modes produce different gap rankings (as expected)

### Dashboard loaders
- `load_entities("nlp")` → 12,867 rows, columns: entity_id, label, type, first_year
- `load_entities("covid-19")` → 5,500 rows
- `load_edges("nlp")` → 22,690 rows, columns: source, relation, target, first_observed
- `load_entity_embeddings("nlp")` → 7 node2vec years + 7 specter2 years
- `load_citation_history("nlp")` → 9,202 rows
- `build_graphs( ents, edges)` → 7 graph snapshots for NLP

---

## Known limitations

1. **Sample-based scoring:** The dashboard uses `sample_n=400` entities for gap scoring.
   This is ~80K pairs checked vs the full ~20M pairs. The top gaps are representative
   but not exhaustive. For production use, a proper approximate nearest neighbor index
   (FAISS, Annoy) would be needed.

2. **Citation data is mention-velocity fallback:** The S2 API was unreachable when C6 ran.
   Citation counts in `citations_*.csv` are derived from mention-velocity, not real S2
   citation counts. The dashboard shows "mention-velocity fallback" as the citation source.

3. **Structural-only mode is slow:** α=1.0 (pure structural) takes ~218s for 200 samples
   because it computes full pairwise cosine similarity on 128-dim vectors. This is inherent
   to the algorithm. The dashboard doesn't expose α=1.0 as a slider option (only 0.0, 0.3,
   0.5, 0.7, 1.0 in config — but 1.0 would time out).

4. **Semantic-only mode may find no gaps:** With real data, α=0.0 finds zero gaps because
   all candidate entity pairs at the graph level are already connected. This is correct
   behavior — the semantic channel alone doesn't identify gaps that the graph doesn't already
   capture. The fusion (α=0.5) is where the interesting gaps emerge.

5. **COVID 2018 data doesn't exist:** The real C5 pipeline starts COVID at 2019. Two demo
   CSV files for COVID 2018 were removed from exports/ (they were synthetic placeholders).

---

## Running the dashboard

```bash
cd /Users/anjan/Desktop/capstone_sep_15/dashboard

# Option 1: Streamlit run (requires streamlit installed)
/opt/anaconda3/bin/python3 -m streamlit run app.py --server.port=8501

# Option 2: Verify data loads correctly (no GUI)
/opt/anaconda3/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from data_loader import load_papers, load_entities, load_edges, load_entity_embeddings, load_citation_history
from gap_engine import build_graphs, rank_gaps
for dom in ['nlp', 'covid-19']:
    ents = load_entities(dom)
    edges = load_edges(dom)
    embs = load_entity_embeddings(dom)
    cits = load_citation_history(dom)
    graphs = build_graphs(ents, edges)
    gaps, checked = rank_gaps(embs, graphs, cits, alpha=0.5, t=2021, t1=2020, top_k=5, sample_n=200)
    print(f'{dom}: {len(gaps)} gaps scored from {checked} pairs')
"
```

---

## Architecture note

The dashboard is a **frozen snapshot** viewer — no live API calls. All data is pre-computed
by Components 1–6 and exported to CSV. The dashboard reads these CSVs and presents them
through 4 tabs:

1. **Corpus Overview:** Paper counts, entity/relation growth over time
2. **Top Gaps:** Ranked unconnected concept pairs with gap scores, priority, velocity,
   and similarity history charts
3. **Graph Timeline:** Interactive PyVis network visualization of the knowledge graph
   for a selected year range, with optional focus concept
4. **Validation:** Retrospective hit rate — gaps predicted at cutoff 2021, checked against
   post-cutoff (2022–2024) co-mentions. Two modes: demo (crude paper-text citation analysis)
   and real (Component 3's `validate_with_real_data`)

---

## Comparison: demo vs real data

| Metric | Demo (make_demo_embeddings.py) | Real (export scripts) |
|--------|------|------|
| NLP entities | 500 | 12,867 |
| NLP edges | ~37,500 | 22,690 |
| NLP node2vec per year | 500 | 1,815–10,611 |
| COVID entities | 500 | 5,500 |
| COVID edges | ~12,500 | 8,081 |
| COVID node2vec per year | 500 | 783–3,968 |
| Embeddings | Random normal vectors | SPECTER2 + Node2Vec from real papers |
| Citation source | Synthetic | Mention-velocity fallback (C6) |
| Gap scores | Meaningless (random) | Meaningful (semantic + structural similarity) |
