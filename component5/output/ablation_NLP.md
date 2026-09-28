# Component 5 — Ablation Results: NLP

**Domain:** NLP  
**Gap source:** `04_fuse_and_rank.py` (FAISS-ranked, top-75 per setting)  
**Validation:** Post-cutoff co-mention check (Component 3 methodology)  
**Generated:** 2026-09-21 19:15:59  

## Hit Rate by Setting × Cutoff

| Setting | Cutoff 2021 | Cutoff 2022 | Cutoff 2023 |
|---|---|------|------|---
| α=0.0 (semantic-only) | **92.0% (69/75)** | **85.3% (64/75)** | **82.7% (62/75)** |
| α=0.25 | 76.0% (57/75) | 78.7% (59/75) | 73.3% (55/75) |
| α=0.5 (default) | 64.0% (48/75) | 68.0% (51/75) | 54.7% (41/75) |
| α=0.75 | 25.3% (19/75) | 36.0% (27/75) | 20.0% (15/75) |
| α=1.0 (structural-only) | 20.0% (15/75) | 29.3% (22/75) | 16.0% (12/75) |
| structural_only | 20.0% (15/75) | 29.3% (22/75) | 16.0% (12/75) |
| semantic_only | **92.0% (69/75)** | **85.3% (64/75)** | **82.7% (62/75)** |
| random_baseline | 48.0% (36.0/75.0) | 58.7% (44.0/75.0) | 50.2% (37.67/75.0) |

## Top Gaps — Latest Year (alpha=0.5)

| Rank | Entity A | Entity B | Structural | Semantic | Fused |
|---|---|------|---|---
| 1 | canon_Other_01743<br><small>dialogue summaries [Other]</small> | canon_Task_01208<br><small>Abstractive Dialogue Summarization [Task]</small> | 0.9937 | 0.9616 | 0.9777 |
| 2 | canon_Material_00902<br><small>Machine Learning Papers [Material]</small> | canon_Method_02705<br><small>automatic machine learning pipeline [Method]</small> | 0.9874 | 0.9645 | 0.9770 |
| 3 | canon_Method_03699<br><small>contrastive learning framework [Method]</small> | canon_Method_03701<br><small>exemplar-wise contrastive learning [Method]</small> | 0.9891 | 0.9559 | 0.9738 |
| 4 | canon_Task_01032<br><small>syntactically-controlled sentence generation [Task]</small> | canon_Task_01202<br><small>Lexically constrained generation [Task]</small> | 0.9896 | 0.9569 | 0.9729 |
| 5 | canon_Method_00654<br><small>multi-task learning framework [Method]</small> | canon_Method_01511<br><small>joint multi-task learning setting [Method]</small> | 0.9548 | 0.9884 | 0.9722 |
| 6 | canon_Material_01800<br><small>English essays [Material]</small> | canon_Material_01801<br><small>essayforum [Material]</small> | 0.9876 | 0.9581 | 0.9720 |
| 7 | canon_Other_03224<br><small>cross-modal nature [Other]</small> | canon_Task_02241<br><small>cross-modal task [Task]</small> | 0.9757 | 0.9647 | 0.9715 |
| 8 | canon_Other_00282<br><small>pla [Other]</small> | canon_Other_01498<br><small>other [Other]</small> | 0.9948 | 0.9464 | 0.9713 |
| 9 | canon_Other_02467<br><small>aspect category-sentiment pairs [Other]</small> | canon_Other_02759<br><small>user sentiment [Other]</small> | 0.9959 | 0.9461 | 0.9705 |
| 10 | canon_Task_00352<br><small>ER) [Task]</small> | canon_Task_02356<br><small>LCG) [Task]</small> | 0.9903 | 0.9467 | 0.9692 |

## Notes

- **Alpha sweep:** [0.0, 0.25, 0.5, 0.75, 1.0]  
- **Ablations:** structural_only, semantic_only, random_baseline  
- **Top-K evaluated per setting:** 75  
- **Random baseline seeds:** [42, 123, 456]  
- **Bold cells** indicate the best hit rate for that cutoff.  
- **Structural-only** (α=1.0) and **semantic-only** (α=0.0) are included in the alpha sweep.  
- **Random baseline** shuffles the top-75 α=0.5 pairs and reports mean hit rate over seeds.  

## Input Files

Gap files: `component5/output/gaps/{domain}_{cutoff}_alpha{alpha}.json` (from 04)  
Component 2 data: `{domain}_entities.json`, `{domain}_extracted.json`, `{domain}_relations.json`  
