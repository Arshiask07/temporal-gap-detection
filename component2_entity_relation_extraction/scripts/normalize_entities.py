"""
Component 2 — entity-level normalization / deduplication.

Uses RapidFuzz (threshold ~0.85, matching the project's established
paper-level dedup threshold from SAMPLING_REPORT.md) to cluster surface
forms into canonical entities, WITHOUT losing the original extracted text.

Guardrails against over-merging (per the task spec — string similarity
alone is not sufficient):
  - Only entities of the SAME type are ever merged (a Method never merges
    with a Task, even if the strings are close).
  - Only entities within the SAME domain are merged (NLP canonical space
    is kept separate from COVID's — a temporal KG per domain in Component 3
    should not accidentally bridge unrelated corpora).
  - Abbreviation matching is handled separately from fuzzy string matching:
    an all-caps token of 2-6 letters is linked to a longer phrase only if
    its letters match that phrase's initials (e.g. "NMT" <-> "neural
    machine translation"), not by edit-distance.
  - A fuzzy match additionally requires the two normalized strings to
    share a token, to avoid merging short unrelated strings that happen to
    score high on RapidFuzz's ratio.

Output: for each domain, a canonical_id is written back onto every entity
record, plus a standalone surface_form -> canonical_id mapping file so the
original text is always recoverable.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path
from collections import defaultdict

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import OUTPUT_DIR, read_json, write_json

FUZZY_THRESHOLD = 85.0  # RapidFuzz 0-100 scale == the project's 0.85 ratio


def is_abbrev(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{2,6}", text)) and text == text.upper()


def initials(phrase: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z]+", phrase) if w.lower() not in
             {"of", "the", "a", "an", "for", "and", "in", "on", "to"}]
    return "".join(w[0] for w in words).upper()


def cluster_domain(entities: list[dict]) -> tuple[list[dict], dict]:
    by_type = defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)

    surface_to_canonical: dict[str, str] = {}
    next_id_counter = defaultdict(int)

    for etype, ents in by_type.items():
        # frequency-ranked so common surface forms become the canonical anchor
        freq = defaultdict(int)
        surface_examples: dict[str, str] = {}
        for e in ents:
            freq[e["normalized_form"]] += 1
            surface_examples.setdefault(e["normalized_form"], e["surface_form"])

        norms_by_freq = sorted(freq.keys(), key=lambda n: -freq[n])
        assigned: dict[str, str] = {}   # normalized_form -> canonical_id

        for norm in norms_by_freq:
            if norm in assigned:
                continue

            # try abbreviation match against already-canonicalized long forms
            matched_canonical = None
            if is_abbrev(surface_examples[norm]):
                for other_norm, canon in list(assigned.items()):
                    if len(other_norm.split()) >= 2 and initials(other_norm) == norm.upper():
                        matched_canonical = canon
                        break

            # fuzzy match against existing canonical anchors of same type
            if matched_canonical is None:
                for other_norm, canon in list(assigned.items()):
                    if not (set(norm.split()) & set(other_norm.split())):
                        continue  # require shared token — guards against spurious merges
                    score = fuzz.token_sort_ratio(norm, other_norm)
                    if score >= FUZZY_THRESHOLD:
                        matched_canonical = canon
                        break

            if matched_canonical is None:
                next_id_counter[etype] += 1
                matched_canonical = f"canon_{etype}_{next_id_counter[etype]:05d}"

            assigned[norm] = matched_canonical

        for norm, canon in assigned.items():
            surface_to_canonical[norm] = canon

    # write canonical_id back onto every entity
    for e in entities:
        e["canonical_id"] = surface_to_canonical.get(e["normalized_form"])

    mapping = {
        norm: canon for norm, canon in surface_to_canonical.items()
    }
    return entities, mapping


def main():
    for domain in ("NLP", "COVID"):
        entities = read_json(OUTPUT_DIR / f"{domain}_entities.json")
        entities, mapping = cluster_domain(entities)
        write_json(OUTPUT_DIR / f"{domain}_entities.json", entities)
        write_json(OUTPUT_DIR / f"{domain}_entity_normalization_map.json", mapping)

        n_surface = len(mapping)
        n_canonical = len(set(mapping.values()))
        print(f"[{domain}] {n_surface} distinct surface forms -> "
              f"{n_canonical} canonical entities "
              f"({n_surface - n_canonical} merges)")

        # propagate canonical_id back into the *_extracted.json paper records too
        extracted = read_json(OUTPUT_DIR / f"{domain}_extracted.json")
        id_to_canon = {e["entity_id"]: e["canonical_id"] for e in entities}
        for paper in extracted:
            for e in paper.get("entities", []):
                e["canonical_id"] = id_to_canon.get(e["entity_id"])
        write_json(OUTPUT_DIR / f"{domain}_extracted.json", extracted)


if __name__ == "__main__":
    main()
