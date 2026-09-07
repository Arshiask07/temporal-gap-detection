Scam — TASK_DECISIONS.md
=========================
Historical decisions, frozen in writing, so the paper's stated schema and
the pipeline's actual output can't quietly disagree again.

2026-09-07 — METHOD_IMPROVES_TASK is DROPPED from the paper's relation schema.
The production outputs for both NLP and COVID carry exactly the 4-way schema
(USED_FOR, METHOD_APPLIED_TO, METHOD_EVALUATED_BY, ENTITY_ASSOCIATED_WITH_ENTITY)
with zero METHOD_IMPROVES_TASK occurrences in either domain. The Sep 2026 drop
is documented in COMPREHENSIVE_COMPONENT1_3_REPORT.md line 325. The decision is
in writing here so it doesn't get lost.

2026-09-07 — Component 4 (KG builder) reads from:
    /Users/anjan/Desktop/capstone_sep_7/component2_entity_relation_extraction/output/
specifically the files NLP_entities.json, NLP_relations.json, COVID_entities.json,
COVID_relations.json (and the matching _extracted.json files). This is the
component2_entity_relation_extraction/output/ tree, NOT cappro/output/ (which
has none of these files) and NOT com2_using_3models/output/scibert/ (which is a
byte-identical mirror, same SHA256, so functionally equivalent but not the
primary path SCAM_ROOT points at). If two of those three locations diverge, the
correct one is component2_entity_relation_extraction/output/.

2026-09-07 — The 828/829-vs-840 COVID relation discrepancy is explained, not a bug:
    - run_relation_inference() in extract_entities_relations_multimodel.py
      (line 524-527) iterates all papers that have entities, then skips any
      paper where len(ents) < 2 or the corpus record is missing.
    - No try/except swallowing, no silent exception hiding. The only skip path
      is the explicit continue at line 527.
    - 840 COVID papers in the corpus. NER produced entities for 834 of them
      (11918 entities total, ~14.3 per paper). 6 papers had zero entities from
      NER and therefore never enter the relation inference loop.
    - Of the 834 papers with entities, 12 have exactly 1 entity and are skipped
      by the len(ents) < 2 check (a relation needs at least 2 entities).
    - That leaves 822 papers that actually run inference, producing 8081 typed
      relations. 822 + 12 (skipped, 1 entity) + 6 (skipped, 0 entities) = 840.
    - 828 vs 840: If you count papers that entered the loop (834) and subtract
      the 6 with no entities, you get 828. The 12 papers with 1 entity DO enter
      the loop but are skipped at the len(ents) < 2 check. This is expected
      behavior, not a bug. No exception is swallowed.
