# COMPONENT 2 REPORT — Scientific Entity & Relation Extraction


## NLP domain

**Entity extraction method:** **Tier 2 production** — SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned on the SciERC corpus for entity recognition. Relation extraction still uses the Tier-1 dependency-pattern layer (not a trained relation classifier) — see Known Limitations below.

**Relation extraction method:** SciBERT span-pair classifier, fine-tuned on SciERC's native relation labels.

- Papers in sampled corpus: **1400**
- Papers processed OK: **1400**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **28463**
- Total relations extracted: **22690**
- Distinct surface forms → canonical entities (post-normalization): **15084 → 12867**
- Papers with zero entities: **0**
- Papers with zero relations: **8**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Method | 10473 | 36.8% |
| Other | 8997 | 31.6% |
| Task | 4855 | 17.1% |
| Material | 3028 | 10.6% |
| Metric | 1110 | 3.9% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| USED_FOR | 9548 | 42.1% |
| ENTITY_ASSOCIATED_WITH_ENTITY | 7738 | 34.1% |
| METHOD_EVALUATED_BY | 2992 | 13.2% |
| METHOD_APPLIED_TO | 2412 | 10.6% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2018 | 200 | 3662 | 2960 | 18.3 | 14.8 |
| 2019 | 200 | 3780 | 2959 | 18.9 | 14.8 |
| 2020 | 200 | 3956 | 3210 | 19.8 | 16.1 |
| 2021 | 200 | 4016 | 3108 | 20.1 | 15.5 |
| 2022 | 200 | 4200 | 3295 | 21.0 | 16.5 |
| 2023 | 200 | 4332 | 3469 | 21.7 | 17.3 |
| 2024 | 200 | 4517 | 3689 | 22.6 | 18.4 |

**Sample zero-entity paper IDs:** []

**Sample zero-relation paper IDs:** ['P18-1.33', '2020.acl-tutorials.4', '2020.acl-tutorials.2', '2021.naacl-main.290', '2022.emnlp-main.229', '2024.emnlp-main.900', '2024.acl-long.24', '2024.eacl-long.115']

**Errored papers:** none — 0 crashes across the full run.

### NLP — Example extracted papers

**Paper:** Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures  
*(paper_id=D18-1.458, year=2018, domain=NLP)*

Entities:
- Self — **Method** (conf 0.4482)
- Neural Machine Translation Architectures — **Method** (conf 0.9591)
- non-recurrent architectures — **Method** (conf 0.9676)
- convolutional, — **Method** (conf 0.6273)
- self-attentional) — **Method** (conf 0.8795)
- RNNs — **Method** (conf 0.9536)
- neural machine translation — **Task** (conf 0.9238)
- CNNs — **Method** (conf 0.9275)
- self-attentional networks — **Method** (conf 0.9566)
- RNNs — **Method** (conf 0.9412)

Relations:
- non-recurrent architectures → **ENTITY_ASSOCIATED_WITH_ENTITY** → RNNs  (conf 0.6078)
- non-recurrent architectures → **METHOD_APPLIED_TO** → neural machine translation  (conf 0.8277)
- convolutional, → **ENTITY_ASSOCIATED_WITH_ENTITY** → non-recurrent architectures  (conf 0.843)
- self-attentional) → **ENTITY_ASSOCIATED_WITH_ENTITY** → non-recurrent architectures  (conf 0.8753)
- RNNs → **METHOD_APPLIED_TO** → neural machine translation  (conf 0.8929)
- CNNs → **ENTITY_ASSOCIATED_WITH_ENTITY** → self-attentional networks  (conf 0.9323)
- CNNs → **ENTITY_ASSOCIATED_WITH_ENTITY** → RNNs  (conf 0.7619)
- CNNs → **ENTITY_ASSOCIATED_WITH_ENTITY** → self-attentional networks  (conf 0.6671)

---

**Paper:** HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training  
*(paper_id=2020.emnlp-main.161, year=2020, domain=NLP)*

Entities:
- HERO — **Method** (conf 0.7926)
- Hierarchical Encoder — **Method** (conf 0.9666)
- Video+Language Omni-representation Pre-training — **Task** (conf 0.8896)
- HERO — **Method** (conf 0.846)
- framework — **Other** (conf 0.7419)
- large-scale video+language omni-representation learning — **Task** (conf 0.8784)
- HERO — **Method** (conf 0.6181)
- multimodal inputs — **Other** (conf 0.6474)
- hierarchical structure — **Other** (conf 0.7882)
- local context — **Other** (conf 0.8069)

Relations:
- Video+Language Omni-representation Pre-training → **USED_FOR** → Hierarchical Encoder  (conf 0.6619)
- HERO → **ENTITY_ASSOCIATED_WITH_ENTITY** → framework  (conf 0.9285)
- framework → **USED_FOR** → large-scale video+language omni-representation learning  (conf 0.9581)
- HERO → **USED_FOR** → multimodal inputs  (conf 0.8924)
- hierarchical structure → **USED_FOR** → multimodal inputs  (conf 0.9388)
- local context → **ENTITY_ASSOCIATED_WITH_ENTITY** → global video context  (conf 0.6808)
- Cross → **USED_FOR** → multimodal inputs  (conf 0.8835)
- Cross → **USED_FOR** → local context  (conf 0.9533)

---

**Paper:** Causal Distillation for Language Models  
*(paper_id=2022.naacl-main.318, year=2022, domain=NLP)*

Entities:
- Causal Distillation — **Method** (conf 0.8774)
- Language Models — **Method** (conf 0.725)
- Distillation — **Method** (conf 0.7738)
- language models — **Method** (conf 0.9664)
- approach — **Other** (conf 0.6067)
- dist — **Task** (conf 0.505)
- student model — **Method** (conf 0.9563)
- language — **Task** (conf 0.5221)
- hidden states — **Other** (conf 0.7129)
- student model — **Method** (conf 0.9073)

Relations:
- Causal Distillation → **USED_FOR** → Language Models  (conf 0.9951)
- Distillation → **USED_FOR** → language models  (conf 0.9378)
- approach → **USED_FOR** → dist  (conf 0.9926)
- approach → **USED_FOR** → student model  (conf 0.8722)
- hidden states → **ENTITY_ASSOCIATED_WITH_ENTITY** → student model  (conf 0.4048)
- hidden states → **ENTITY_ASSOCIATED_WITH_ENTITY** → teacher model  (conf 0.2966)
- student model → **USED_FOR** → hidden states  (conf 0.7779)
- student model → **ENTITY_ASSOCIATED_WITH_ENTITY** → teacher model  (conf 0.6818)

---


## COVID domain

**Entity extraction method:** **Tier 2 production** — SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned on the SciERC corpus for entity recognition. Relation extraction still uses the Tier-1 dependency-pattern layer (not a trained relation classifier) — see Known Limitations below.

**Relation extraction method:** SciBERT span-pair classifier, fine-tuned on SciERC's native relation labels.

- Papers in sampled corpus: **840**
- Papers processed OK: **840**
- Papers skipped (empty/invalid abstract): **0**
- Papers errored: **0**
- Total entities extracted: **11918**
- Total relations extracted: **8081**
- Distinct surface forms → canonical entities (post-normalization): **6117 → 5500**
- Papers with zero entities: **6**
- Papers with zero relations: **92**

**Entity type distribution:**

| Type | Count | % |
|---|---|---|
| Other | 4273 | 35.9% |
| Method | 3602 | 30.2% |
| Material | 2463 | 20.7% |
| Task | 1319 | 11.1% |
| Metric | 261 | 2.2% |

**Relation type distribution:**

| Relation | Count | % |
|---|---|---|
| ENTITY_ASSOCIATED_WITH_ENTITY | 4400 | 54.4% |
| USED_FOR | 2560 | 31.7% |
| METHOD_APPLIED_TO | 801 | 9.9% |
| METHOD_EVALUATED_BY | 320 | 4.0% |

**Per-year statistics:**

| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |
|---|---|---|---|---|---|
| 2019 | 140 | 1629 | 1045 | 11.6 | 7.5 |
| 2020 | 140 | 1716 | 985 | 12.3 | 7.0 |
| 2021 | 140 | 1884 | 1101 | 13.5 | 7.9 |
| 2022 | 140 | 2279 | 1711 | 16.3 | 12.2 |
| 2023 | 140 | 2153 | 1643 | 15.4 | 11.7 |
| 2024 | 140 | 2257 | 1596 | 16.1 | 11.4 |

**Sample zero-entity paper IDs:** ['openalex_W3134128968', 'openalex_W2917867348', 'openalex_W2914022334', 'openalex_W2948006344', 'openalex_W2912761948', 'openalex_W2772116241']

**Sample zero-relation paper IDs:** ['openalex_W3134128968', 'openalex_W2909090086', 'openalex_W2982241583', 'openalex_W2917867348', 'openalex_W2985646009', 'gen_5efb882d08f9', 'openalex_W2977501160', 'openalex_W2942506922', 'openalex_W3126998785', 'openalex_W3012233527', 'openalex_W2936532646', 'openalex_W2971390764', 'openalex_W3016472752', 'openalex_W4287977093', 'openalex_W3115613875']

**Errored papers:** none — 0 crashes across the full run.

### COVID — Example extracted papers

**Paper:** Connecting omics signatures of diseases, drugs, and mechanisms of actions with iLINCS  
*(paper_id=openalex_W2982434722, year=2019, domain=COVID)*

Entities:
- omics signatures of — **Other** (conf 0.5803)
- iLINCS — **Method** (conf 0.5933)
- platforms — **Other** (conf 0.5776)
- omics data — **Material** (conf 0.6029)
- bioinformatics tools — **Method** (conf 0.6786)
- integrative analyses — **Task** (conf 0.5488)
- visualization — **Task** (conf 0.2581)
- iLI — **Method** (conf 0.8719)
- NCS — **Method** (conf 0.5032)
- iLI — **Method** (conf 0.9073)

Relations:
- platforms → **USED_FOR** → integrative analyses  (conf 0.8991)
- platforms → **USED_FOR** → visualization  (conf 0.7676)
- omics data → **USED_FOR** → platforms  (conf 0.7657)
- omics data → **ENTITY_ASSOCIATED_WITH_ENTITY** → bioinformatics tools  (conf 0.7588)
- integrative analyses → **ENTITY_ASSOCIATED_WITH_ENTITY** → visualization  (conf 0.9814)
- iLI → **ENTITY_ASSOCIATED_WITH_ENTITY** → integrative  (conf 0.9355)
- iLI → **ENTITY_ASSOCIATED_WITH_ENTITY** → web-based platform  (conf 0.9241)
- iLI → **METHOD_APPLIED_TO** → omics data  (conf 0.6524)

---

**Paper:** An Agile Digital Platform to Support Population Health—A Case Study of a Digital Platform to Support Patients with Delirium Using IoT, NLP, and AI  
*(paper_id=openalex_W3164580315, year=2021, domain=COVID)*

Entities:
- Agile Digital Platform — **Method** (conf 0.709)
- Population Health — **Task** (conf 0.5271)
- Digital Platform — **Method** (conf 0.7233)
- IoT — **Method** (conf 0.8065)
- NLP — **Method** (conf 0.8793)
- AI — **Method** (conf 0.5895)
- digital services — **Method** (conf 0.539)
- agile — **Method** (conf 0.6518)
- digital platforms — **Method** (conf 0.6563)
- dynamic software ecosystems — **Other** (conf 0.4803)

Relations:
- Agile Digital Platform → **METHOD_APPLIED_TO** → Population Health  (conf 0.9737)
- Digital Platform → **METHOD_APPLIED_TO** → Population Health  (conf 0.9878)
- IoT → **USED_FOR** → Agile Digital Platform  (conf 0.8822)
- IoT → **USED_FOR** → Digital Platform  (conf 0.9969)
- IoT → **ENTITY_ASSOCIATED_WITH_ENTITY** → NLP  (conf 0.5334)
- IoT → **ENTITY_ASSOCIATED_WITH_ENTITY** → AI  (conf 0.6483)
- NLP → **USED_FOR** → Agile Digital Platform  (conf 0.8924)
- NLP → **USED_FOR** → Digital Platform  (conf 0.9987)

---

**Paper:** A machine learning and explainable artificial intelligence triage-prediction system for COVID-19  
*(paper_id=openalex_W4372311383, year=2023, domain=COVID)*

Entities:
- machine learning and explainable artificial intelligence triage-prediction system — **Method** (conf 0.9793)
- COV — **Material** (conf 0.3138)
- COV — **Material** (conf 0.3516)
- respiratory — **Other** (conf 0.2795)
- SAR — **Material** (conf 0.3851)
- Artificial intelligence advances — **Method** (conf 0.7316)
- healthcare diagnosis — **Task** (conf 0.6587)
- hetero — **Method** (conf 0.9581)
- genous Machine Learning and — **Method** (conf 0.768)
- Deep Learning algorithms — **Method** (conf 0.9267)

Relations:
- machine learning and explainable artificial intelligence triage-prediction system → **METHOD_APPLIED_TO** → COV  (conf 0.9215)
- COV → **ENTITY_ASSOCIATED_WITH_ENTITY** → respiratory  (conf 0.9065)
- SAR → **USED_FOR** → respiratory  (conf 0.9373)
- Artificial intelligence advances → **METHOD_APPLIED_TO** → healthcare diagnosis  (conf 0.9664)
- clinical markers → **ENTITY_ASSOCIATED_WITH_ENTITY** → Deep Learning algorithms  (conf 0.6344)
- classifier architectures → **USED_FOR** → COV  (conf 0.983)
- Logistic Regression → **ENTITY_ASSOCIATED_WITH_ENTITY** → Random Forest  (conf 0.794)
- Logistic Regression → **ENTITY_ASSOCIATED_WITH_ENTITY** → Support Vector Machine  (conf 0.6801)

---


## Known limitations

- Entities: SciBERT fine-tuned on SciERC (learned, not lexicon-based). See `eval_scierc_f1.py` output for held-out entity F1.

- Relations: SciBERT span-pair classifier fine-tuned on SciERC's native relation labels (Used-for, Feature-of, Hyponym-of, Part-of, Compare, Conjunction, Evaluate-for), mapped onto this project's 5-value relation schema. `scierc_relation_type` is kept on every relation record alongside `relation_type` so the original SciERC-native label is never lost.

- No coreference resolution, no external KB linking (e.g. UMLS).


## What Component 3 should consume

- **Nodes**: one node per distinct `canonical_id` in `{NLP,COVID}_entities.json`.
- **Edges**: `{NLP,COVID}_relations.json`, remapped to `canonical_id` via the entity lookup.
- **Temporal snapshot key**: `year` on every entity/relation record.
- **Provenance**: `paper_id` + `domain` preserved everywhere.
- **Surface-form provenance**: `{NLP,COVID}_entity_normalization_map.json`.
