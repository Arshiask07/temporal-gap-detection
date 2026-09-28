# COMPONENT 2 — CROSS-MODEL COMPARISON (SciBERT vs RoBERTa vs PubMedBERT)


## SciBERT  (`allenai/scibert_scivocab_uncased`)

- SciERC held-out TEST entity F1: **0.5974** (P=0.6108, R=0.5847, n=551 sentences)

**NLP domain:**
- Papers: 1400  |  Entities: 28463  |  Relations: 22690
- Typed-relation share (non-generic): **65.9%**
- 'Other' entity-type share: **31.6%**
- Entity types: Method=10473, Other=8997, Task=4855, Material=3028, Metric=1110
- Relation types: USED_FOR=9548, ENTITY_ASSOCIATED_WITH_ENTITY=7738, METHOD_EVALUATED_BY=2992, METHOD_APPLIED_TO=2412

**COVID domain:**
- Papers: 840  |  Entities: 11918  |  Relations: 8081
- Typed-relation share (non-generic): **45.6%**
- 'Other' entity-type share: **35.9%**
- Entity types: Other=4273, Method=3602, Material=2463, Task=1319, Metric=261
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=4400, USED_FOR=2560, METHOD_APPLIED_TO=801, METHOD_EVALUATED_BY=320

## RoBERTa  (`roberta-base`)

- SciERC held-out TEST entity F1: **0.5642** (P=0.5492, R=0.5799, n=551 sentences)

**NLP domain:**
- Papers: 1400  |  Entities: 30316  |  Relations: 26236
- Typed-relation share (non-generic): **61.1%**
- 'Other' entity-type share: **24.5%**
- Entity types: Method=12127, Other=7439, Task=5506, Material=4623, Metric=621
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=10216, USED_FOR=9709, METHOD_APPLIED_TO=3348, METHOD_EVALUATED_BY=2963

**COVID domain:**
- Papers: 840  |  Entities: 13677  |  Relations: 11321
- Typed-relation share (non-generic): **39.7%**
- 'Other' entity-type share: **23.1%**
- Entity types: Method=4709, Other=3163, Material=2781, Task=2650, Metric=374
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=6821, USED_FOR=2705, METHOD_APPLIED_TO=1313, METHOD_EVALUATED_BY=482

## PubMedBERT  (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)

- SciERC held-out TEST entity F1: **0.6082** (P=0.5911, R=0.6263, n=551 sentences)

**NLP domain:**
- Papers: 1400  |  Entities: 31135  |  Relations: 27318
- Typed-relation share (non-generic): **61.6%**
- 'Other' entity-type share: **33.7%**
- Entity types: Other=10480, Method=10083, Task=6597, Material=2872, Metric=1103
- Relation types: USED_FOR=10998, ENTITY_ASSOCIATED_WITH_ENTITY=10483, METHOD_EVALUATED_BY=3129, METHOD_APPLIED_TO=2708

**COVID domain:**
- Papers: 840  |  Entities: 14308  |  Relations: 13042
- Typed-relation share (non-generic): **34.6%**
- 'Other' entity-type share: **33.2%**
- Entity types: Other=4749, Method=3354, Task=3035, Material=2641, Metric=529
- Relation types: ENTITY_ASSOCIATED_WITH_ENTITY=8534, USED_FOR=2967, METHOD_APPLIED_TO=1179, METHOD_EVALUATED_BY=362