# Component 5 — Ablation Results: COVID

**Domain:** COVID  
**Gap source:** `05_evaluate.py` (random draw of 75 from candidate pool, same set as the α=0.5 ranking)  
**Validation:** Post-cutoff co-mention check (Component 3 methodology)  
**Generated:** 2026-09-21 19:16:00  

## Hit Rate by Setting × Cutoff

| Setting | Cutoff 2020 | Cutoff 2021 | Cutoff 2022 |
|---|---|------|------|---
| α=0.0 (semantic-only) | **94.7% (71/75)** | **92.0% (69/75)** | **89.3% (67/75)** |
| α=0.25 | 69.3% (52/75) | 80.0% (60/75) | 76.0% (57/75) |
| α=0.5 (default) | 36.0% (27/75) | 42.7% (32/75) | 49.3% (37/75) |
| α=0.75 | 22.7% (17/75) | 13.3% (10/75) | 5.3% (4/75) |
| α=1.0 (structural-only) | 12.0% (9/75) | 2.7% (2/75) | 5.3% (4/75) |
| structural_only | 12.0% (9/75) | 2.7% (2/75) | 5.3% (4/75) |
| semantic_only | **94.7% (71/75)** | **92.0% (69/75)** | **89.3% (67/75)** |
| random_baseline | 17.3% (13.0/75.0) | 22.7% (17.0/75.0) | 24.9% (18.67/75.0) |

## Top Gaps — Latest Year (random draw of 75 from candidate pool)

| Rank | Entity A | Entity B | Structural | Semantic | Fused |
|---|---|------|---|---
| 1 | canon_Method_01334<br><small>crowdsource knowledge graph based approach [Method]</small> | canon_Method_01335<br><small>hierarchical crowdsource knowledge graph based framework [Method]</small> | 0.9891 | 0.9862 | 0.9888 |
| 2 | canon_Method_00928<br><small>Scientometric Review [Method]</small> | canon_Method_00929<br><small>scientometric analysis [Method]</small> | 0.9871 | 0.9873 | 0.9875 |
| 3 | canon_Metric_00024<br><small>egg quality characteristics [Metric]</small> | canon_Metric_00026<br><small>egg quality traits [Metric]</small> | 0.9937 | 0.9690 | 0.9823 |
| 4 | canon_Method_00368<br><small>Emotion Cognizance [Method]</small> | canon_Method_00369<br><small>emotion cognizant representations [Method]</small> | 0.9674 | 0.9843 | 0.9772 |
| 5 | canon_Method_01877<br><small>multiple linear regression (MLR) [Method]</small> | canon_Method_01878<br><small>partial least square regression (PLSR) [Method]</small> | 0.9861 | 0.9593 | 0.9742 |
| 6 | canon_Other_00614<br><small>Human Factors [Other]</small> | canon_Other_01814<br><small>personal factors [Other]</small> | 0.9945 | 0.9509 | 0.9739 |
| 7 | canon_Material_00581<br><small>ImageNet database [Material]</small> | canon_Method_01147<br><small>DenseNet169 [Method]</small> | 0.9975 | 0.9466 | 0.9725 |
| 8 | canon_Other_00212<br><small>personalization-privacy paradox [Other]</small> | canon_Other_00515<br><small>personalization- [Other]</small> | 0.9752 | 0.9579 | 0.9692 |
| 9 | canon_Method_01496<br><small>docking and [Method]</small> | canon_Method_01528<br><small>docking software [Method]</small> | 0.9512 | 0.9815 | 0.9666 |
| 10 | canon_Material_00401<br><small>global digital social networks [Material]</small> | canon_Material_00544<br><small>online and social networking platforms [Material]</small> | 0.9837 | 0.9394 | 0.9634 |

## Notes

- **Alpha sweep:** [0.0, 0.25, 0.5, 0.75, 1.0]  
- **Ablations:** structural_only, semantic_only, random_baseline  
- **Top-K evaluated per setting:** 75  
- **Random baseline seeds:** [42, 123, 456]  
- **Bold cells** indicate the best hit rate for that cutoff.  
- **Structural-only** (α=1.0) and **semantic-only** (α=0.0) are included in the alpha sweep.  
- **Random baseline** is a random draw of 75 pairs from the candidate pool (not a shuffle of the top-75).  

## Input Files

Gap files: `component5/output/gaps/{domain}_{cutoff}_alpha{alpha}.json` (from 04)  
Component 2 data: `{domain}_entities.json`, `{domain}_extracted.json`, `{domain}_relations.json`  
