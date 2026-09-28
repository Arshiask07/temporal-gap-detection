"""
Component 2 — validation / QC report generator.

IMPORTANT: this script computes every number LIVE from
{domain}_extracted.json / {domain}_entities.json / {domain}_relations.json
every time it runs. It does NOT read any cached "run_stats.json" file —
that was a real bug in an earlier version: the baseline script wrote such
a cache, the SciBERT script never did, so after a SciBERT run the report
silently kept showing stale baseline totals next to fresh SciBERT per-year
numbers. Computing live from the actual output files makes that class of
bug impossible — there's nothing to go stale.

Which extraction method actually produced the current output is detected
from an `extraction_method` marker written by whichever script ran last
(see METHOD_MARKER handling below) rather than assumed/hardcoded.
"""
from __future__ import annotations
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import OUTPUT_DIR, REPORT_DIR, read_json

random.seed(42)

METHOD_DESCRIPTIONS = {
    "baseline": (
        "**Tier 1 baseline** (spaCy `en_core_web_sm` syntactic parse + "
        "SciERC-schema trigger lexicons + dependency-pattern relation "
        "extraction). This tier runs when SciBERT/AllenNLP weights are not "
        "reachable (e.g. huggingface.co / AWS S3 blocked)."
    ),
    "scibert_finetuned": (
        "**Tier 2 production** — SciBERT (`allenai/scibert_scivocab_uncased`) "
        "fine-tuned on the SciERC corpus for entity recognition. Relation "
        "extraction still uses the Tier-1 dependency-pattern layer (not a "
        "trained relation classifier) — see Known Limitations below."
    ),
    "unknown": (
        "**Unknown** — no `extraction_method` marker was found next to the "
        "output files. Re-run extraction so this report can state accurately "
        "which pipeline produced these numbers."
    ),
}


def detect_method(domain: str) -> str:
    marker_path = OUTPUT_DIR / f"{domain}_extraction_method.json"
    if marker_path.exists():
        return read_json(marker_path).get("method", "unknown")
    return "unknown"


def detect_relation_method(domain: str) -> str:
    marker_path = OUTPUT_DIR / f"{domain}_relation_method.json"
    if marker_path.exists():
        return read_json(marker_path).get("method", "heuristic")
    return "heuristic"  # no marker => still the Tier-1 dependency-pattern layer


def compute_stats_live(domain: str, extracted: list[dict], entities: list[dict],
                        relations: list[dict], mapping: dict) -> dict:
    """Every summary number, computed fresh from the actual output files —
    never from a cache that could be out of date."""
    status_counts = Counter(p["status"] for p in extracted)
    entity_type_counts = Counter(e["type"] for e in entities)
    relation_type_counts = Counter(r["relation_type"] for r in relations)
    zero_entity = [p["paper_id"] for p in extracted if len(p.get("entities", [])) == 0]
    zero_relation = [p["paper_id"] for p in extracted if len(p.get("relations", [])) == 0]
    failed = [{"paper_id": p["paper_id"], "reason": p.get("error", p["status"])}
              for p in extracted if p["status"] == "error"]

    return {
        "domain": domain,
        "papers_total": len(extracted),
        "papers_processed_ok": status_counts.get("ok", 0),
        "papers_skipped_empty_abstract": status_counts.get("skipped_empty_abstract", 0),
        "papers_errored": status_counts.get("error", 0),
        "total_entities": len(entities),
        "total_relations": len(relations),
        "entity_type_counts": dict(entity_type_counts),
        "relation_type_counts": dict(relation_type_counts),
        "papers_zero_entities": len(zero_entity),
        "papers_zero_relations": len(zero_relation),
        "zero_entity_paper_ids_sample": zero_entity[:15],
        "zero_relation_paper_ids_sample": zero_relation[:15],
        "failed_papers": failed,
        "distinct_surface_forms": len(mapping),
        "distinct_canonical_entities": len(set(mapping.values())) if mapping else 0,
    }


def per_year_domain_table(extracted, domain):
    rows = defaultdict(lambda: {"papers": 0, "entities": 0, "relations": 0})
    for p in extracted:
        y = p.get("year")
        rows[y]["papers"] += 1
        rows[y]["entities"] += len(p.get("entities", []))
        rows[y]["relations"] += len(p.get("relations", []))
    lines = ["| Year | Papers | Entities | Relations | Avg Ent/Paper | Avg Rel/Paper |",
              "|---|---|---|---|---|---|"]
    for y in sorted(rows, key=lambda v: (v is None, v)):
        r = rows[y]
        avg_e = r["entities"] / r["papers"] if r["papers"] else 0
        avg_r = r["relations"] / r["papers"] if r["papers"] else 0
        lines.append(f"| {y} | {r['papers']} | {r['entities']} | {r['relations']} | {avg_e:.1f} | {avg_r:.1f} |")
    return "\n".join(lines)


def pick_example_papers(extracted, n=3):
    good = [p for p in extracted if p["status"] == "ok" and p.get("relations")]
    years_seen = sorted(set(p["year"] for p in good))
    picks = []
    step = max(1, len(years_seen) // n)
    for y in years_seen[::step][:n]:
        candidates = [p for p in good if p["year"] == y]
        candidates.sort(key=lambda p: -len(p["relations"]))
        if candidates:
            picks.append(candidates[0])
    while len(picks) < n and good:
        c = random.choice(good)
        if c not in picks:
            picks.append(c)
    return picks


def render_example(p, entity_lookup):
    lines = [f"**Paper:** {p['title']}  ", f"*(paper_id={p['paper_id']}, year={p['year']}, domain={p['domain']})*", ""]
    lines.append("Entities:")
    for e in p["entities"][:10]:
        lines.append(f"- {e['surface_form']} — **{e['type']}** (conf {e['confidence']})")
    lines.append("")
    lines.append("Relations:")
    for r in p["relations"][:8]:
        src = entity_lookup.get(r["source_entity_id"], {}).get("surface_form", "?")
        tgt = entity_lookup.get(r["target_entity_id"], {}).get("surface_form", "?")
        lines.append(f"- {src} → **{r['relation_type']}** → {tgt}  (conf {r['confidence']})")
    return "\n".join(lines)


def build_report():
    sections = []
    sections.append("# COMPONENT 2 REPORT — Scientific Entity & Relation Extraction\n")

    methods_seen = set()
    rel_methods_seen = set()
    domain_stats = {}

    for domain in ("NLP", "COVID"):
        extracted = read_json(OUTPUT_DIR / f"{domain}_extracted.json")
        entities = read_json(OUTPUT_DIR / f"{domain}_entities.json")
        relations = read_json(OUTPUT_DIR / f"{domain}_relations.json")
        try:
            mapping = read_json(OUTPUT_DIR / f"{domain}_entity_normalization_map.json")
        except FileNotFoundError:
            mapping = {}

        method = detect_method(domain)
        rel_method = detect_relation_method(domain)
        methods_seen.add(method)
        rel_methods_seen.add(rel_method)
        st = compute_stats_live(domain, extracted, entities, relations, mapping)
        domain_stats[domain] = st

        entity_lookup = {e["entity_id"]: e for e in entities}

        sections.append(f"\n## {domain} domain\n")
        sections.append(f"**Entity extraction method:** {METHOD_DESCRIPTIONS[method]}\n")
        rel_desc = ("**Relation extraction method:** SciBERT span-pair classifier, "
                    "fine-tuned on SciERC's native relation labels.\n"
                    if rel_method == "scibert_relation_finetuned" else
                    "**Relation extraction method:** Tier-1 dependency-pattern heuristic "
                    "(not a trained classifier) — see Known Limitations below.\n")
        sections.append(rel_desc)
        sections.append(f"- Papers in sampled corpus: **{st['papers_total']}**")
        sections.append(f"- Papers processed OK: **{st['papers_processed_ok']}**")
        sections.append(f"- Papers skipped (empty/invalid abstract): **{st['papers_skipped_empty_abstract']}**")
        sections.append(f"- Papers errored: **{st['papers_errored']}**")
        sections.append(f"- Total entities extracted: **{st['total_entities']}**")
        sections.append(f"- Total relations extracted: **{st['total_relations']}**")
        sections.append(f"- Distinct surface forms → canonical entities (post-normalization): "
                         f"**{st['distinct_surface_forms']} → {st['distinct_canonical_entities']}**")
        sections.append(f"- Papers with zero entities: **{st['papers_zero_entities']}**")
        sections.append(f"- Papers with zero relations: **{st['papers_zero_relations']}**\n")

        sections.append("**Entity type distribution:**\n")
        sections.append("| Type | Count | % |")
        sections.append("|---|---|---|")
        total_e = st["total_entities"] or 1
        for etype, cnt in sorted(st["entity_type_counts"].items(), key=lambda x: -x[1]):
            sections.append(f"| {etype} | {cnt} | {100*cnt/total_e:.1f}% |")

        sections.append("\n**Relation type distribution:**\n")
        sections.append("| Relation | Count | % |")
        sections.append("|---|---|---|")
        total_r = st["total_relations"] or 1
        for rtype, cnt in sorted(st["relation_type_counts"].items(), key=lambda x: -x[1]):
            sections.append(f"| {rtype} | {cnt} | {100*cnt/total_r:.1f}% |")

        sections.append("\n**Per-year statistics:**\n")
        sections.append(per_year_domain_table(extracted, domain))

        sections.append(f"\n**Sample zero-entity paper IDs:** {st['zero_entity_paper_ids_sample']}")
        sections.append(f"\n**Sample zero-relation paper IDs:** {st['zero_relation_paper_ids_sample']}")
        if st["failed_papers"]:
            sections.append(f"\n**Errored papers:** {st['failed_papers'][:10]}")
        else:
            sections.append("\n**Errored papers:** none — 0 crashes across the full run.")

        sections.append(f"\n### {domain} — Example extracted papers\n")
        for p in pick_example_papers(extracted, n=3):
            sections.append(render_example(p, entity_lookup))
            sections.append("\n---\n")

    sections.append("\n## Known limitations\n")
    entity_line = (
        "- Entities: SciBERT fine-tuned on SciERC (learned, not lexicon-based). "
        "See `eval_scierc_f1.py` output for held-out entity F1.\n"
        if "scibert_finetuned" in methods_seen else
        "- Entities: trigger-lexicon based, not learned.\n"
    )
    relation_line = (
        "- Relations: SciBERT span-pair classifier fine-tuned on SciERC's native "
        "relation labels (Used-for, Feature-of, Hyponym-of, Part-of, Compare, "
        "Conjunction, Evaluate-for), mapped onto this project's 5-value relation "
        "schema. `scierc_relation_type` is kept on every relation record alongside "
        "`relation_type` so the original SciERC-native label is never lost.\n"
        if "scibert_relation_finetuned" in rel_methods_seen else
        "- Relations: STILL the Tier-1 dependency-pattern heuristic, not a trained "
        "classifier — same-sentence co-occurrence dominates "
        "(`ENTITY_ASSOCIATED_WITH_ENTITY`), which is why typed relations are a "
        "small minority. Run `extract_relations_scibert.py --train --run` to "
        "replace this with a trained relation classifier.\n"
    )
    sections.append(entity_line)
    sections.append(relation_line)
    sections.append("- No coreference resolution, no external KB linking (e.g. UMLS).\n")

    sections.append("\n## What Component 3 should consume\n")
    sections.append(
        "- **Nodes**: one node per distinct `canonical_id` in `{NLP,COVID}_entities.json`.\n"
        "- **Edges**: `{NLP,COVID}_relations.json`, remapped to `canonical_id` via the "
        "entity lookup.\n"
        "- **Temporal snapshot key**: `year` on every entity/relation record.\n"
        "- **Provenance**: `paper_id` + `domain` preserved everywhere.\n"
        "- **Surface-form provenance**: `{NLP,COVID}_entity_normalization_map.json`.\n"
    )

    report_path = REPORT_DIR / "COMPONENT2_REPORT.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {report_path} ({report_path.stat().st_size} bytes)")
    print(f"Detected methods: {methods_seen}")


if __name__ == "__main__":
    build_report()
