# Component 6 — Ablation Results: NLP

**Domain:** NLP  
**Citation path:** mention-velocity fallback (S2 API unreachable)  
**Gap source:** velocity re-ranked (priority = fused_score × mean(vel_u, vel_v))  
**Validation:** Post-cutoff co-mention check (Component 3 methodology)  
**Generated:** 2026-09-21 20:27:20  

## Hit Rate by Setting × Cutoff

| Setting | Cutoff 2021 | Cutoff 2022 | Cutoff 2023 |
|---|---|------|------|---
| fused_only (C5 baseline) | **64.0% (48/75)** | 68.0% (51/75) | 54.7% (41/75) |
| velocity (C6 re-ranked) | 48.0% (36/75) | **72.0% (54/75)** | **61.3% (46/75)** |
| random_baseline | 54.7% (41.0/75.0) | 58.2% (43.67/75.0) | 54.2% (40.67/75.0) |

### Top Velocity Gaps — Latest Year

| Rank | Entity A | Entity B | Fused | vel_u | vel_v | Priority |
|---|---|------|---|---
| 1 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_01139<br><small>TKGT [Method]</small> | 0.9454 | 1.5000 | 1.0000 | 1.1817 |
| 2 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_04801<br><small>different [Method]</small> | 0.9478 | 1.5000 | 0.5000 | 0.9478 |
| 3 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_05079<br><small>X-ELMs [Method]</small> | 0.9455 | 1.5000 | 0.5000 | 0.9455 |
| 4 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_03391<br><small>UniRE [Method]</small> | 0.9486 | 1.5000 | 0.0000 | 0.7114 |
| 5 | canon_Method_00392<br><small>AxCell [Method]</small> | canon_Method_00520<br><small>XLT [Method]</small> | 0.9460 | 0.0000 | 1.5000 | 0.7095 |
| 6 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_00710<br><small>MXNet [Method]</small> | 0.9459 | 1.5000 | 0.0000 | 0.7095 |
| 7 | canon_Method_02318<br><small>distance metrics [Method]</small> | canon_Metric_00560<br><small>Fréchet Distance [Method]</small> | 0.9446 | 0.0000 | 1.5000 | 0.7085 |
| 8 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_04056<br><small>BMR) [Method]</small> | 0.9444 | 1.5000 | 0.0000 | 0.7083 |
| 9 | canon_Method_00520<br><small>XLT [Method]</small> | canon_Method_02477<br><small>XLMR [Method]</small> | 0.9442 | 1.5000 | 0.0000 | 0.7082 |
| 10 | canon_Material_01800<br><small>English essays [Material]</small> | canon_Material_01801<br><small>essayforum [Material]</small> | 0.9720 | 0.5000 | 0.5000 | 0.4860 |

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
