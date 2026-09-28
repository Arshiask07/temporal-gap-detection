"""
Component 2 — Tier 1 (BASELINE, executed in this environment)
Scientific entity + relation extraction using spaCy syntax (noun chunks +
dependency parse) driven by domain trigger lexicons, following the SciERC
entity/relation schema as closely as a rule/statistical system reasonably can.

WHY THIS TIER EXISTS: this sandbox cannot reach huggingface.co or AWS S3
(both return HTTP 403 through the network allowlist), so no SciBERT weights
or AllenNLP/scispacy pretrained SciERC models can be downloaded here. This
script uses only PyPI-installable, no-external-weights components (spaCy's
small English pipeline, downloaded once from a GitHub release, which IS
reachable) so that Component 2 has a real, fully executed, non-mock output
today. See scripts/extract_entities_relations_scibert.py for the intended
production-grade replacement (SciBERT backbone, fine-tuned on the SciERC
corpus) to run on a machine with normal internet access — swapping it in
requires no changes to Component 3+, since both scripts emit the identical
JSON schema defined below.

Entity types:   Task, Method, Material, Metric, Other
Relation types: USED_FOR, METHOD_APPLIED_TO, METHOD_IMPROVES_TASK,
                METHOD_EVALUATED_BY, ENTITY_ASSOCIATED_WITH_ENTITY

Run:
    python3 extract_entities_relations_baseline.py
"""
from __future__ import annotations
import re
import sys
import time
import logging
from collections import Counter
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import load_domain_corpus, write_json, OUTPUT_DIR, LOG_DIR
from lexicons import (
    ALL_TYPE_LEXICONS, USED_FOR_VERBS, IMPROVES_VERBS, EVALUATED_BY_VERBS,
    ASSOCIATION_VERBS, STOPWORD_HEADS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "component2_baseline.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("component2")

MIN_ABSTRACT_CHARS = 20
MIN_CHUNK_LEN = 3          # chars
MAX_CHUNK_WORDS = 6
DOC_FREQ_KEEP_THRESHOLD = 3  # corpus-level salience floor for untyped ('Other') chunks
MAX_FALLBACK_PAIRS_PER_SENT = 2


# --------------------------------------------------------------------------- #
# Text normalization helpers
# --------------------------------------------------------------------------- #
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an|this|these|those|our|its|their)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    # crude singularization (avoid over-stripping short/irregular words)
    if len(s) > 4 and s.endswith("s") and not s.endswith(("ss", "us", "is", "ies")):
        s = s[:-1]
    elif s.endswith("ies") and len(s) > 5:
        s = s[:-3] + "y"
    return s


def classify_type(chunk_text_norm: str, chunk_lemmas: set[str]) -> str | None:
    for etype in ("Metric", "Material", "Method", "Task"):
        triggers = ALL_TYPE_LEXICONS[etype]
        if chunk_text_norm in triggers:
            return etype
        if chunk_lemmas & triggers:
            return etype
        for trig in triggers:
            if " " in trig and trig in chunk_text_norm:
                return etype
    return None


GENERIC_HEAD_WORDS = {
    "which", "that", "this", "these", "those", "it", "they", "who", "whom",
    "something", "anything", "everything", "nothing", "one", "ones",
    "background", "methods", "results", "conclusion", "conclusions",
    "objective", "objectives", "purpose", "update", "effect", "effects",
}


def is_junk_chunk(chunk, text_norm: str, root_lemma: str) -> bool:
    if len(text_norm) < MIN_CHUNK_LEN:
        return True
    if root_lemma in STOPWORD_HEADS:
        return True
    if text_norm.isdigit():
        return True
    if len(text_norm.split()) > MAX_CHUNK_WORDS:
        return True
    if not re.search(r"[a-zA-Z]", text_norm):
        return True
    if chunk.root.pos_ == "PRON":
        return True
    if text_norm in GENERIC_HEAD_WORDS:
        return True
    # single common-word chunk with no content beyond a bare pronoun/determiner head
    if len(chunk) == 1 and chunk[0].pos_ in ("PRON", "DET"):
        return True
    return False


# --------------------------------------------------------------------------- #
# Relation helpers
# --------------------------------------------------------------------------- #
def subtree_char_span(token):
    subtree = list(token.subtree)
    start = min(t.idx for t in subtree)
    end = max(t.idx + len(t.text) for t in subtree)
    return start, end


def entity_for_span(span, entities):
    s, e = span
    best, best_ov = None, 0
    for ent in entities:
        if e <= ent["char_start"] or s >= ent["char_end"]:
            continue
        ov = min(e, ent["char_end"]) - max(s, ent["char_start"])
        if ov > best_ov:
            best, best_ov = ent, ov
    return best


def extract_sentence_relations(sent, sent_entities, rel_counter):
    relations = []
    if len(sent_entities) < 2:
        return relations

    connected_pairs = set()

    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        lemma = tok.lemma_.lower()
        subj_children = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
        dobj_children = [c for c in tok.children if c.dep_ == "dobj"]
        prep_children = [c for c in tok.children if c.dep_ == "prep"]
        pobj_children = []
        for p in prep_children:
            pobj_children.extend([c for c in p.children if c.dep_ == "pobj"])

        candidate_objects = dobj_children + pobj_children
        anchor_children = subj_children if subj_children else dobj_children[:1]

        for a in anchor_children:
            a_span = subtree_char_span(a)
            src = entity_for_span(a_span, sent_entities)
            if not src:
                continue
            for b in candidate_objects:
                if b is a:
                    continue
                b_span = subtree_char_span(b)
                tgt = entity_for_span(b_span, sent_entities)
                if not tgt or tgt is src:
                    continue

                if lemma in EVALUATED_BY_VERBS:
                    rel_type, conf = "METHOD_EVALUATED_BY", 0.65
                elif lemma in IMPROVES_VERBS:
                    rel_type, conf = "METHOD_IMPROVES_TASK", 0.65
                elif lemma in USED_FOR_VERBS:
                    if src["type"] == "Method" and tgt["type"] in ("Task", "Material"):
                        rel_type, conf = "METHOD_APPLIED_TO", 0.6
                    else:
                        rel_type, conf = "USED_FOR", 0.55
                elif lemma in ASSOCIATION_VERBS:
                    rel_type, conf = "ENTITY_ASSOCIATED_WITH_ENTITY", 0.5
                else:
                    continue

                pair_key = (src["entity_id"], tgt["entity_id"])
                if pair_key in connected_pairs:
                    continue
                connected_pairs.add(pair_key)
                relations.append((src, tgt, rel_type, conf))
                rel_counter[rel_type] += 1

    # Fallback co-occurrence relation for sentences with entities but no
    # verb-pattern match, so genuinely related terms aren't silently dropped.
    if not relations:
        ents_sorted = sorted(sent_entities, key=lambda e: e["char_start"])
        n_pairs = min(len(ents_sorted) - 1, MAX_FALLBACK_PAIRS_PER_SENT)
        for i in range(n_pairs):
            src, tgt = ents_sorted[i], ents_sorted[i + 1]
            relations.append((src, tgt, "ENTITY_ASSOCIATED_WITH_ENTITY", 0.3))
            rel_counter["ENTITY_ASSOCIATED_WITH_ENTITY"] += 1

    return relations


# --------------------------------------------------------------------------- #
# Main per-domain pipeline
# --------------------------------------------------------------------------- #
def process_domain(nlp, domain: str):
    records = load_domain_corpus(domain)
    log.info(f"[{domain}] loaded {len(records)} sampled papers")

    texts, meta = [], []
    for rec in records:
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        text = (title + ". " + abstract).strip() if abstract else ""
        texts.append(text if len(text) >= MIN_ABSTRACT_CHARS else "")
        meta.append(rec)

    log.info(f"[{domain}] running spaCy pipeline over {len(texts)} documents ...")
    t0 = time.time()
    docs = list(nlp.pipe(texts, batch_size=64))
    log.info(f"[{domain}] spaCy parse done in {time.time()-t0:.1f}s")

    # ---- Pass 1: corpus-level chunk document-frequency (salience signal) ----
    doc_freq = Counter()
    for doc in docs:
        seen = set()
        for chunk in doc.noun_chunks:
            norm = normalize_text(chunk.text)
            if norm and norm not in seen:
                seen.add(norm)
        doc_freq.update(seen)

    # ---- Pass 2: entity + relation extraction ----
    paper_records, all_entities, all_relations = [], [], []
    failed_papers = []
    zero_entity_papers, zero_relation_papers = [], []
    entity_type_counts = Counter()
    relation_type_counts = Counter()
    year_stats = Counter()

    for rec, doc, text in zip(meta, docs, texts):
        paper_id = rec["paper_id"]
        year = rec.get("year")
        try:
            if not text:
                failed_papers.append({"paper_id": paper_id, "reason": "empty_or_invalid_abstract"})
                paper_records.append({
                    "paper_id": paper_id, "year": year, "domain": domain,
                    "title": rec.get("title", ""), "venue": rec.get("venue") or rec.get("journal", ""),
                    "source": rec.get("source", ""), "status": "skipped_empty_abstract",
                    "entities": [], "relations": [],
                })
                zero_entity_papers.append(paper_id)
                zero_relation_papers.append(paper_id)
                continue

            paper_entities = []
            for idx, chunk in enumerate(doc.noun_chunks):
                norm = normalize_text(chunk.text)
                root_lemma = chunk.root.lemma_.lower()
                if is_junk_chunk(chunk, norm, root_lemma):
                    continue
                lemmas = {t.lemma_.lower() for t in chunk}
                etype = classify_type(norm, lemmas)
                if etype is None:
                    if doc_freq[norm] < DOC_FREQ_KEEP_THRESHOLD:
                        continue
                    etype = "Other"
                    conf = 0.5
                else:
                    conf = 0.75 + (0.15 if doc_freq[norm] >= DOC_FREQ_KEEP_THRESHOLD else 0.0)
                    conf = min(conf, 0.95)

                entity = {
                    "entity_id": f"e_{paper_id}_{idx}",
                    "canonical_id": None,
                    "surface_form": chunk.text,
                    "normalized_form": norm,
                    "type": etype,
                    "char_start": chunk.start_char,
                    "char_end": chunk.end_char,
                    "confidence": round(conf, 3),
                    "paper_id": paper_id,
                    "year": year,
                    "domain": domain,
                }
                paper_entities.append(entity)
                entity_type_counts[etype] += 1

            paper_relations = []
            rel_idx = 0
            for sent in doc.sents:
                sent_entities = [e for e in paper_entities
                                  if e["char_start"] >= sent.start_char and e["char_end"] <= sent.end_char]
                sent_rels = extract_sentence_relations(sent, sent_entities, relation_type_counts)
                for src, tgt, rtype, conf in sent_rels:
                    relation = {
                        "relation_id": f"r_{paper_id}_{rel_idx}",
                        "source_entity_id": src["entity_id"],
                        "target_entity_id": tgt["entity_id"],
                        "relation_type": rtype,
                        "confidence": round(conf, 3),
                        "paper_id": paper_id,
                        "year": year,
                        "domain": domain,
                    }
                    paper_relations.append(relation)
                    rel_idx += 1

            all_entities.extend(paper_entities)
            all_relations.extend(paper_relations)
            year_stats[year] += 1

            if not paper_entities:
                zero_entity_papers.append(paper_id)
            if not paper_relations:
                zero_relation_papers.append(paper_id)

            paper_records.append({
                "paper_id": paper_id, "year": year, "domain": domain,
                "title": rec.get("title", ""), "venue": rec.get("venue") or rec.get("journal", ""),
                "source": rec.get("source", ""), "status": "ok",
                "n_entities": len(paper_entities), "n_relations": len(paper_relations),
                "entities": paper_entities, "relations": paper_relations,
            })

        except Exception as exc:  # never let one bad paper kill the run
            log.error(f"[{domain}] FAILED paper_id={paper_id}: {exc}")
            failed_papers.append({"paper_id": paper_id, "reason": str(exc)})
            paper_records.append({
                "paper_id": paper_id, "year": year, "domain": domain,
                "title": rec.get("title", ""), "status": "error", "error": str(exc),
                "entities": [], "relations": [],
            })

    stats = {
        "domain": domain,
        "papers_total": len(records),
        "papers_processed_ok": sum(1 for p in paper_records if p["status"] == "ok"),
        "papers_skipped_empty_abstract": sum(1 for p in paper_records if p["status"] == "skipped_empty_abstract"),
        "papers_errored": sum(1 for p in paper_records if p["status"] == "error"),
        "total_entities": len(all_entities),
        "total_relations": len(all_relations),
        "entity_type_counts": dict(entity_type_counts),
        "relation_type_counts": dict(relation_type_counts),
        "papers_by_year": dict(sorted(year_stats.items())),
        "papers_zero_entities": len(zero_entity_papers),
        "papers_zero_relations": len(zero_relation_papers),
        "zero_entity_paper_ids_sample": zero_entity_papers[:15],
        "zero_relation_paper_ids_sample": zero_relation_papers[:15],
        "failed_papers": failed_papers,
    }
    return paper_records, all_entities, all_relations, stats


def main():
    log.info("Loading spaCy pipeline (en_core_web_sm) ...")
    nlp = spacy.load("en_core_web_sm")

    overall_stats = {}
    for domain in ("NLP", "COVID"):
        paper_records, entities, relations, stats = process_domain(nlp, domain)
        write_json(OUTPUT_DIR / f"{domain}_extracted.json", paper_records)
        write_json(OUTPUT_DIR / f"{domain}_entities.json", entities)
        write_json(OUTPUT_DIR / f"{domain}_relations.json", relations)
        write_json(OUTPUT_DIR / f"{domain}_stats.json", stats)
        write_json(OUTPUT_DIR / f"{domain}_extraction_method.json", {
            "method": "baseline",
            "model": "spacy en_core_web_sm + trigger lexicons",
        })
        # This run also overwrites {domain}_relations.json with heuristic
        # output, so reset the relation-method marker too — otherwise a
        # trained relation classifier's marker from an earlier run would
        # stay behind and make the report claim SciBERT relations that
        # this baseline run just replaced with heuristic ones.
        write_json(OUTPUT_DIR / f"{domain}_relation_method.json", {
            "method": "heuristic",
            "model": "dependency-pattern rules (Tier-1 baseline)",
        })
        overall_stats[domain] = stats
        log.info(f"[{domain}] DONE — {stats['total_entities']} entities, "
                  f"{stats['total_relations']} relations, "
                  f"{stats['papers_processed_ok']}/{stats['papers_total']} papers OK")

    write_json(OUTPUT_DIR / "component2_run_stats.json", overall_stats)
    log.info("Component 2 baseline extraction complete.")


if __name__ == "__main__":
    main()
