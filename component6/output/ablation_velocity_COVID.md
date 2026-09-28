# Component 6 — Ablation Results: COVID

**Domain:** COVID  
**Citation path:** mention-velocity fallback (S2 API unreachable)  
**Gap source:** velocity re-ranked (priority = fused_score × mean(vel_u, vel_v))  
**Validation:** Post-cutoff co-mention check (Component 3 methodology)  
**Generated:** 2026-09-21 20:27:21  

## Hit Rate by Setting × Cutoff

| Setting | Cutoff 2020 | Cutoff 2021 | Cutoff 2022 |
|---|---|------|------|---
| fused_only (C5 baseline) | **36.0% (27/75)** | **42.7% (32/75)** | **49.3% (37/75)** |
| velocity (C6 re-ranked) | 28.0% (21/75) | 22.7% (17/75) | 37.3% (28/75) |
| random_baseline | 20.4% (15.33/75.0) | 24.4% (18.33/75.0) | 26.2% (19.67/75.0) |

### Top Velocity Gaps — Latest Year

| Rank | Entity A | Entity B | Fused | vel_u | vel_v | Priority |
|---|---|------|---|---
| 1 | canon_Material_00855<br><small>CRIs [Material]</small> | canon_Other_00119<br><small>Hyper [Other]</small> | 0.9337 | 0.5000 | 2.0000 | 1.1671 |
| 2 | canon_Other_00119<br><small>Hyper [Other]</small> | canon_Other_00667<br><small>stra [Other]</small> | 0.9383 | 2.0000 | 0.0000 | 0.9383 |
| 3 | canon_Task_00170<br><small>tuberculosis detection [Task]</small> | canon_Task_00781<br><small>Chest Disease Classification [Task]</small> | 0.9440 | 1.0000 | 0.5000 | 0.7080 |
| 4 | canon_Other_00359<br><small>REST [Other]</small> | canon_Task_00751<br><small>RTD [Task]</small> | 0.9386 | 1.0000 | 0.5000 | 0.7039 |
| 5 | canon_Method_00355<br><small>MPL [Method]</small> | canon_Method_01708<br><small>Python libraries [Method]</small> | 0.9325 | 1.0000 | 0.5000 | 0.6994 |
| 6 | canon_Method_01877<br><small>multiple linear regression (MLR) [Method]</small> | canon_Method_01878<br><small>partial least square regression (PLSR) [Method]</small> | 0.9742 | 0.5000 | 0.5000 | 0.4871 |
| 7 | canon_Other_00614<br><small>Human Factors [Other]</small> | canon_Other_01814<br><small>personal factors [Other]</small> | 0.9739 | 0.0000 | 1.0000 | 0.4870 |
| 8 | canon_Other_01643<br><small>Disinformation Hashtags [Other]</small> | canon_Other_01644<br><small>hashtag [Other]</small> | 0.9571 | 0.5000 | 0.5000 | 0.4786 |
| 9 | canon_Other_01659<br><small>health misinformation perceptions [Other]</small> | canon_Task_00606<br><small>Health Misinformation and Disinformation [Task]</small> | 0.9500 | 1.0000 | 0.0000 | 0.4750 |
| 10 | canon_Method_01837<br><small>Deeplabv3plus-Based CNN Model [Method]</small> | canon_Method_01841<br><small>deeplabv3plus [Method]</small> | 0.9460 | 0.5000 | 0.5000 | 0.4730 |

## Notes

- **Velocity window:** 2022–2024  
- **Velocity formula:** vel(e) = (c_e[2024] − c_e[2022]) / 2  
- **Priority formula:** priority = fused_score × mean(vel_u, vel_v)  
- **Citation path:** mention-velocity fallback (S2 API unreachable)  
- **Fused-only** uses Component 5's alpha=0.5 gaps (top-75)  
- **Velocity** uses Component 6's priority-sorted gaps (top-75)  
- **Random baseline** draws 75 pairs uniformly from the candidate pool, mean over seeds [42, 123, 456]  
- **Random baseline pool:** mention-velocity (fallback)  
- **Bold cells** indicate the best hit rate for that cutoff.  

## Input Files

Velocity gaps: `component6/output/reranked/{domain}_{cutoff}_alpha0.5.json` (from 03_rerank.py)  
Fused baseline: `component5/output/gaps/{domain}_{cutoff}_alpha0.5.json` (from 04_fuse_and_rank.py)  
Component 2 data: `{domain}_entities.json`, `{domain}_extracted.json`  
