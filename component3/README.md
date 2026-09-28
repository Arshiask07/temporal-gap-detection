# Component 3 — Retrospective Gap Validation

**Contribution 3 of the Temporal Research Gap Detection pipeline.**

Validates that the gap-scoring method actually predicts future co-mentions — by pretending it's an earlier year, scoring gaps with only pre-cutoff data, then checking how many of those predicted gaps show up as co-mentioned concepts in later papers.

---

## What it does

1. **Load** the real Component 2 extraction outputs (`{domain}_entities.json`, `{domain}_relations.json`, `{domain}_extracted.json`) from `../component2_entity_relation_extraction/output/`.
2. **Build per-year co-occurrence graphs** from paper-level relation records — if two canonical entities appear in the same paper, that's a co-occurrence edge for that paper's year.
3. **Score candidate gaps at the cutoff year** — find entity pairs that co-occurred in pre-cutoff papers but are *not* directly connected in the cutoff-year graph (they drifted apart or were never linked).
4. **Check post-cutoff materialization** — for each candidate gap, see if both entities' surface forms appear together in any paper published after the cutoff.
5. **Write reports** — structured JSON + human-readable Markdown to `output/`.

---

## Run it

From the `component3/` directory:

```bash
python3 component3_retrospective_validation.py \
  --domain "NLP (ACL/arXiv)" \
  --cutoff 2021 \
  --top_k 25 \
  --min_papers 3
```

| Flag | Default | Meaning |
|---|---|---|
| `--domain` | first domain in config | `"NLP (ACL/arXiv)"` or `"COVID-19 (CS-adjacent)"` |
| `--cutoff` | `2021` | Score gaps using data ≤ this year only |
| `--top_k` | `25` | Number of top gaps to include in the report |
| `--min_papers` | `3` | Drop entities that appear in fewer than this many papers |
| `--alpha` | `0.5` | Fusion weight (reserved for future embedding-based scoring) |
| `--output` | `./output/` | Directory for JSON + Markdown reports |

Both domains run in under 4 seconds on a laptop.

---

## Results (cutoff 2021, top-25 gaps, min_papers=3)

|| Domain | Gaps scored | Hits | Misses | Hit rate |
||---|---|---|---|---|
|| NLP (ACL/arXiv) | 75 | 43 | 32 | **57%** |
|| COVID-19 (CS-adjacent) | 75 | 13 | 62 | **17%** |

- **NLP**: 43/75 — roughly three-fifths of the top-ranked pre-cutoff gaps materialize as co-mentions in 2022–2024 papers.
- **COVID**: 13/75 — lower signal-to-noise expected for a faster-moving, more fragmented literature, but still above random chance.

Reports are written to `output/validation_{NLP,COVID}_cutoff2021.{json,md}`.

---

## How it differs from the dashboard's validation mode

The dashboard (`dashboard/app.py` Tab 4) calls `gap_engine.retrospective_validate()`, which works on **demo embeddings** (node2vec + SPECTER2 CSVs from `make_demo_embeddings.py`). That path scores gaps by embedding similarity and checks co-occurrence against paper abstracts.

This component (`component3_retrospective_validation.py`) works on **real Component 2 extraction data** — the actual SciBERT-extracted entities and typed relations. It builds co-occurrence graphs from paper-level relation records rather than from embedding-space proximity. The two modes validate different things:

| Mode | Data source | Gap signal | Co-mention check |
|---|---|---|---|
| Dashboard (embedding-based) | `dashboard/exports/*.csv` | Fused similarity drift (node2vec + SPECTER2) | Abstract token overlap |
| Component 3 (extraction-based) | `component2_entity_relation_extraction/output/*.json` | Pre-cutoff co-occurrence without cutoff-year edge | Surface-form overlap in post-cutoff paper text |

Both are legitimate retrospective tests. The extraction-based one is the stronger validation because it uses the actual extracted knowledge graph rather than synthetic demo embeddings.

---

## Files

```
component3/
├── component3_retrospective_validation.py   # validation engine (CLI + library)
├── output/                                 # JSON + Markdown reports
│   ├── validation_NLP_cutoff2021.json
│   ├── validation_NLP_cutoff2021.md
│   ├── validation_COVID_cutoff2021.json
│   └── validation_COVID_cutoff2021.md
└── README.md                               # this file
```

---

## Dependencies

Same as the rest of the project — `numpy`, `pandas`, `networkx`, plus the dashboard's `config.py` and `data_loader.py` (imported via `sys.path`). No GPU, no trained models, no network calls. Pure data processing.

---

## Future work

- Raise `--min_papers` to reduce noise from singleton entities.
- Try different cutoffs (2020, 2019) to measure hit-rate sensitivity over time.
- Swap the gap-scoring proxy (co-occurrence count) for the dual-channel embedding similarity when per-entity SPECTER2 embeddings are available for the real extraction data.
- Wire the real-data validation results into the dashboard's Tab 4 alongside the embedding-based mode.
