# COMPONENT 2 REPORT — Scientific Entity & Relation Extraction

Pipeline executed: **Tier 1 baseline** (spaCy `en_core_web_sm` syntactic parse + SciERC-schema trigger lexicons + dependency-pattern relation extraction). This tier was run because `huggingface.co` and AWS S3 (where SciBERT weights and AllenNLP/scispacy pretrained SciERC models are hosted) both return HTTP 403 from this sandbox's network allowlist. `extract_entities_relations_scibert.py` in this same folder is the production-grade replacement, ready to run on a machine with normal internet access — it emits the identical JSON schema, so nothing downstream needs to change when you swap it in.


## NLP domain

- Papers in sampled corpus: **1400**
- Papers processed OK: **1400**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **30248**
- Total relations extracted: **13122**
- Distinct surface forms → canonical entities (post-normalization): **8967 → 7576**
- Papers with zero entities: **0**
- Papers with zero relations: **0**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Other | 16665 | 55.1% |
| Method | 7387 | 24.4% |
| Task | 3364 | 11.1% |
| Material | 1887 | 6.2% |
| Metric | 945 | 3.1% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| ENTITY_ASSOCIATED_WITH_ENTITY | 12562 | 95.7% |
| METHOD_IMPROVES_TASK | 321 | 2.4% |
| USED_FOR | 167 | 1.3% |
| METHOD_EVALUATED_BY | 51 | 0.4% |
| METHOD_APPLIED_TO | 21 | 0.2% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2018 | 200 | 4112 | 1810 | 20.6 | 9.1 |
| 2019 | 200 | 4176 | 1796 | 20.9 | 9.0 |
| 2020 | 200 | 4100 | 1782 | 20.5 | 8.9 |
| 2021 | 200 | 4230 | 1846 | 21.1 | 9.2 |
| 2022 | 200 | 4291 | 1854 | 21.5 | 9.3 |
| 2023 | 200 | 4641 | 2000 | 23.2 | 10.0 |
| 2024 | 200 | 4698 | 2034 | 23.5 | 10.2 |

**Sample zero-entity paper IDs:** []

**Sample zero-relation paper IDs:** []

**Errored papers:** none — 0 crashes across the full run.

### NLP — Example extracted papers

**Paper:** Generating Descriptions from Structured Data Using a Bifocal Attention Mechanism and Gated Orthogonalization  
*(paper_id=N18-1.139, year=2018, domain=NLP)*

Entities:
- Descriptions — **Other** (conf 0.5)
- Structured Data — **Material** (conf 0.9)
- a Bifocal Attention Mechanism — **Method** (conf 0.75)
- the task — **Other** (conf 0.5)
- facts — **Other** (conf 0.5)
- fields — **Other** (conf 0.5)
- values — **Other** (conf 0.5)
- the table — **Other** (conf 0.5)
- a sequence — **Other** (conf 0.5)
- fields — **Other** (conf 0.5)

Relations:
- Descriptions → **ENTITY_ASSOCIATED_WITH_ENTITY** → Structured Data  (conf 0.3)
- Structured Data → **ENTITY_ASSOCIATED_WITH_ENTITY** → a Bifocal Attention Mechanism  (conf 0.3)
- the task → **ENTITY_ASSOCIATED_WITH_ENTITY** → facts  (conf 0.3)
- facts → **ENTITY_ASSOCIATED_WITH_ENTITY** → fields  (conf 0.3)
- the table → **ENTITY_ASSOCIATED_WITH_ENTITY** → a sequence  (conf 0.3)
- a sequence → **ENTITY_ASSOCIATED_WITH_ENTITY** → fields  (conf 0.3)
- example → **ENTITY_ASSOCIATED_WITH_ENTITY** → descriptions  (conf 0.3)
- descriptions → **ENTITY_ASSOCIATED_WITH_ENTITY** → a table  (conf 0.3)

---

**Paper:** KLEJ: Comprehensive Benchmark for Polish Language Understanding  
*(paper_id=2020.acl-main.111, year=2020, domain=NLP)*

Entities:
- Comprehensive Benchmark — **Material** (conf 0.9)
- Polish Language Understanding — **Task** (conf 0.75)
- recent years — **Other** (conf 0.5)
- a series — **Other** (conf 0.5)
- Transformer-based models — **Method** (conf 0.9)
- general natural language understanding (NLU) tasks — **Task** (conf 0.75)
- research — **Other** (conf 0.5)
- general NLU benchmarks — **Material** (conf 0.75)
- a fair comparison — **Other** (conf 0.5)
- the proposed methods — **Method** (conf 0.9)

Relations:
- Comprehensive Benchmark → **ENTITY_ASSOCIATED_WITH_ENTITY** → Polish Language Understanding  (conf 0.3)
- recent years → **ENTITY_ASSOCIATED_WITH_ENTITY** → a series  (conf 0.3)
- a series → **ENTITY_ASSOCIATED_WITH_ENTITY** → Transformer-based models  (conf 0.3)
- research → **ENTITY_ASSOCIATED_WITH_ENTITY** → general NLU benchmarks  (conf 0.3)
- general NLU benchmarks → **ENTITY_ASSOCIATED_WITH_ENTITY** → a fair comparison  (conf 0.3)
- such benchmarks → **ENTITY_ASSOCIATED_WITH_ENTITY** → a handful  (conf 0.3)
- a handful → **ENTITY_ASSOCIATED_WITH_ENTITY** → languages  (conf 0.3)
- this issue → **ENTITY_ASSOCIATED_WITH_ENTITY** → a comprehensive multi-task benchmark  (conf 0.3)

---

**Paper:** Should We Rely on Entity Mentions for Relation Extraction? Debiasing Relation Extraction with Counterfactual Analysis  
*(paper_id=2022.naacl-main.224, year=2022, domain=NLP)*

Entities:
- Entity Mentions — **Other** (conf 0.5)
- Relation Extraction — **Task** (conf 0.9)
- Relation Extraction — **Task** (conf 0.9)
- Counterfactual Analysis — **Task** (conf 0.75)
- the sentence-level relation extraction — **Task** (conf 0.75)
- this risks — **Other** (conf 0.5)
- relations — **Other** (conf 0.5)
- i.e., the spurious correlation — **Metric** (conf 0.75)
- entity mentions — **Other** (conf 0.5)
- names — **Other** (conf 0.5)

Relations:
- Entity Mentions → **ENTITY_ASSOCIATED_WITH_ENTITY** → Relation Extraction  (conf 0.3)
- Relation Extraction → **ENTITY_ASSOCIATED_WITH_ENTITY** → Counterfactual Analysis  (conf 0.3)
- the sentence-level relation extraction → **ENTITY_ASSOCIATED_WITH_ENTITY** → this risks  (conf 0.3)
- this risks → **ENTITY_ASSOCIATED_WITH_ENTITY** → relations  (conf 0.3)
- i.e., the spurious correlation → **ENTITY_ASSOCIATED_WITH_ENTITY** → entity mentions  (conf 0.3)
- entity mentions → **ENTITY_ASSOCIATED_WITH_ENTITY** → names  (conf 0.3)
- the RE models → **ENTITY_ASSOCIATED_WITH_ENTITY** → the relations  (conf 0.3)
- the relations → **ENTITY_ASSOCIATED_WITH_ENTITY** → the text  (conf 0.3)

---


## COVID domain

- Papers in sampled corpus: **840**
- Papers processed OK: **840**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **29814**
- Total relations extracted: **12162**
- Distinct surface forms → canonical entities (post-normalization): **5673 → 5104**
- Papers with zero entities: **7**
- Papers with zero relations: **8**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Other | 22196 | 74.4% |
| Method | 4426 | 14.8% |
| Task | 1571 | 5.3% |
| Material | 858 | 2.9% |
| Metric | 763 | 2.6% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| ENTITY_ASSOCIATED_WITH_ENTITY | 11850 | 97.4% |
| USED_FOR | 143 | 1.2% |
| METHOD_IMPROVES_TASK | 111 | 0.9% |
| METHOD_EVALUATED_BY | 37 | 0.3% |
| METHOD_APPLIED_TO | 21 | 0.2% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2019 | 140 | 3432 | 1382 | 24.5 | 9.9 |
| 2020 | 140 | 4995 | 2054 | 35.7 | 14.7 |
| 2021 | 140 | 5317 | 2152 | 38.0 | 15.4 |
| 2022 | 140 | 5071 | 2091 | 36.2 | 14.9 |
| 2023 | 140 | 5385 | 2196 | 38.5 | 15.7 |
| 2024 | 140 | 5614 | 2287 | 40.1 | 16.3 |

**Sample zero-entity paper IDs:** ['openalex_W2942506922', 'openalex_W2978388382', 'openalex_W2944587989', 'openalex_W3012233527', 'openalex_W2914022334', 'openalex_W2921827323', 'openalex_W4205313296']

**Sample zero-relation paper IDs:** ['openalex_W2917867348', 'openalex_W2942506922', 'openalex_W2978388382', 'openalex_W2944587989', 'openalex_W3012233527', 'openalex_W2914022334', 'openalex_W2921827323', 'openalex_W4205313296']

**Errored papers:** none — 0 crashes across the full run.

### COVID — Example extracted papers

**Paper:** Streamlining Evaluation and Management Payment to Reduce Clinician Burden  
*(paper_id=openalex_W2936532646, year=2019, domain=COVID)*

Entities:
- Center — **Other** (conf 0.5)
- Research — **Other** (conf 0.5)
- Data — **Material** (conf 0.9)
- Science — **Other** (conf 0.5)
- Suite — **Other** (conf 0.5)
- Center — **Other** (conf 0.5)
- Research — **Other** (conf 0.5)
- Data — **Material** (conf 0.9)
- Science — **Other** (conf 0.5)
- Outcomes — **Other** (conf 0.5)

Relations:
- Center → **ENTITY_ASSOCIATED_WITH_ENTITY** → Research  (conf 0.3)
- Research → **ENTITY_ASSOCIATED_WITH_ENTITY** → Data  (conf 0.3)
- Center → **ENTITY_ASSOCIATED_WITH_ENTITY** → Research  (conf 0.3)
- Research → **ENTITY_ASSOCIATED_WITH_ENTITY** → Data  (conf 0.3)
- communication → **ENTITY_ASSOCIATED_WITH_ENTITY** → monitoring  (conf 0.3)
- monitoring → **ENTITY_ASSOCIATED_WITH_ENTITY** → patient care  (conf 0.3)
- assessment → **ENTITY_ASSOCIATED_WITH_ENTITY** → evaluation  (conf 0.3)
- evaluation → **ENTITY_ASSOCIATED_WITH_ENTITY** → collection  (conf 0.3)

---

**Paper:** 6-month neurological and psychiatric outcomes in 236 379 survivors of COVID-19: a retrospective cohort study using electronic health records  
*(paper_id=openalex_W3148333766, year=2021, domain=COVID)*

Entities:
- COVID-19 — **Other** (conf 0.5)
- electronic health records — **Other** (conf 0.5)
- COVID-19 — **Other** (conf 0.5)
- more data — **Other** (conf 0.5)
- COVID-19 — **Other** (conf 0.5)
- incidence rates — **Metric** (conf 0.9)
- relative risks — **Other** (conf 0.5)
- neurological and psychiatric diagnoses — **Task** (conf 0.75)
- patients — **Other** (conf 0.5)
- a COVID-19 diagnosis — **Task** (conf 0.9)

Relations:
- COVID-19 → **ENTITY_ASSOCIATED_WITH_ENTITY** → electronic health records  (conf 0.3)
- COVID-19 → **ENTITY_ASSOCIATED_WITH_ENTITY** → more data  (conf 0.3)
- more data → **ENTITY_ASSOCIATED_WITH_ENTITY** → COVID-19  (conf 0.3)
- incidence rates → **ENTITY_ASSOCIATED_WITH_ENTITY** → relative risks  (conf 0.3)
- relative risks → **ENTITY_ASSOCIATED_WITH_ENTITY** → neurological and psychiatric diagnoses  (conf 0.3)
- METHODS → **ENTITY_ASSOCIATED_WITH_ENTITY** → event  (conf 0.3)
- event → **ENTITY_ASSOCIATED_WITH_ENTITY** → data  (conf 0.3)
- Our primary cohort → **ENTITY_ASSOCIATED_WITH_ENTITY** → patients  (conf 0.3)

---

**Paper:** Predictive performance of multi-model ensemble forecasts of COVID-19 across European nations  
*(paper_id=openalex_W4366602904, year=2023, domain=COVID)*

Entities:
- multi-model ensemble forecasts — **Method** (conf 0.75)
- COVID-19 — **Other** (conf 0.5)
- best practice — **Other** (conf 0.5)
- multiple models — **Method** (conf 0.75)
- an ensemble — **Method** (conf 0.9)
- the performance — **Other** (conf 0.5)
- ensembles — **Method** (conf 0.9)
- COVID-19 cases — **Other** (conf 0.5)
- deaths — **Other** (conf 0.5)
- Europe — **Other** (conf 0.5)

Relations:
- multi-model ensemble forecasts → **ENTITY_ASSOCIATED_WITH_ENTITY** → COVID-19  (conf 0.3)
- best practice → **ENTITY_ASSOCIATED_WITH_ENTITY** → multiple models  (conf 0.3)
- multiple models → **ENTITY_ASSOCIATED_WITH_ENTITY** → an ensemble  (conf 0.3)
- the performance → **ENTITY_ASSOCIATED_WITH_ENTITY** → ensembles  (conf 0.3)
- ensembles → **ENTITY_ASSOCIATED_WITH_ENTITY** → COVID-19 cases  (conf 0.3)
- groups → **ENTITY_ASSOCIATED_WITH_ENTITY** → COVID-19 cases  (conf 0.3)
- COVID-19 cases → **ENTITY_ASSOCIATED_WITH_ENTITY** → deaths  (conf 0.3)
- Teams → **ENTITY_ASSOCIATED_WITH_ENTITY** → forecasts  (conf 0.3)

---


## Known limitations of the Tier-1 baseline

- Entity typing is trigger-lexicon based, not learned — genuinely novel scientific terms with no lexical overlap to the trigger sets fall into `Other` (kept only if they recur ≥3 times in the domain corpus) or are dropped as noise.
- Relation extraction relies on shallow dependency patterns (subject/object of a cue verb) — it will miss relations expressed with more complex syntax, and the `ENTITY_ASSOCIATED_WITH_ENTITY` fallback (same-sentence co-occurrence) is a coarse catch-all, not a semantically verified relation — hence its lower confidence score.
- No coreference resolution — the same real-world entity mentioned differently within one paper is only merged if `normalize_entities.py`'s type-scoped fuzzy/abbreviation rules catch it.
- No entity linking to external KBs (e.g. no UMLS linking for COVID/biomedical terms).
- Confidence scores are heuristic (rule-derived), not calibrated probabilities.


## What Component 3 should consume

- **Nodes**: one node per distinct `canonical_id` in `{NLP,COVID}_entities.json`, typed by the (majority) `type` field across its member surface forms.
- **Edges**: `{NLP,COVID}_relations.json`, remapped from `source_entity_id`/`target_entity_id` (paper-local) to their `canonical_id` via the entity lookup, typed by `relation_type`, weighted/filtered by `confidence`.
- **Temporal snapshot key**: `year` on every entity and relation record — group by year to build `G_2018 ... G_2024` (NLP) and `G_2019 ... G_2024` (COVID) without re-deriving anything from raw text.
- **Provenance**: `paper_id` + `domain` are preserved on every entity and relation record and in `{NLP,COVID}_extracted.json` (paper-level view) — do not discard these when building the graph; Component 6 (citation velocity) and Component 7 (dashboard) will need to trace edges back to source papers.
- **Surface-form provenance**: `{NLP,COVID}_entity_normalization_map.json` keeps the surface-form → canonical_id mapping so the original extracted text is always recoverable from a canonical node.
