#!/usr/bin/env python3
"""
Component 5 — Step 3: Semantic embeddings (SPECTER / MiniLM fallback).

Computes dense semantic embeddings for each canonical entity (per domain)
using a sentence transformer.  The input text for each entity is its
representative surface_form plus the entity type in brackets, e.g.:

    "Word Analogy Testing [Method]"

This gives the model both the lexical content and the type signal.

Model priority (try in order, first success wins):
    1. allenai/specter          — 768-dim, SPECTER v1 (SciBERT-based)
    2. sentence-transformers/all-MiniLM-L6-v2  — 384-dim, 11M params (lightweight fallback)

Output (per domain, time-independent — same embeddings for all years):
    component5/output/embeddings/semantic/{domain}.npy
        shape [n_entities, dim], L2-normalized (cosine == dot product)
    component5/output/embeddings/semantic/{domain}_semantic_id_map.json
        {canonical_id: row_index}
    component5/output/embeddings/semantic/semantic_build_log.json
        per-domain: model_name, dim, device, entity_count, time, peak_mem

Input (read-only, from Component 2 — active tree only):
    component2_entity_relation_extraction/output/
    ├── NLP_entities.json
    └── COVID_entities.json

Runtime: CPU or CUDA.  Batch size is configurable (default 8) to control
memory.  Seeded: torch.manual_seed(42), numpy.random.seed(42).

NOTE: If the machine has < 8GB free RAM, even SPECTER may OOM at load time.
MiniLM (11M params) fits comfortably on 8GB.  The script catches load errors
and falls back automatically.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ── paths ──────────────────────────────────────────────────────────────────

# Paths are derived from this script's own location so the pipeline runs
# unchanged on any machine (Windows/macOS/Linux) with no hand-editing.
# Layout assumed:  <project_root>/component5/<this script>
#                  <project_root>/component2_entity_relation_extraction/output
PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"

COMP5_ROOT = PROJECT_ROOT / "component5"
COMP5_OUT = COMP5_ROOT / "output"
SEM_DIR = COMP5_OUT / "embeddings" / "semantic"

DOMAIN_SPECS = {
    "NLP": COMP2_OUT / "NLP_entities.json",
    "COVID": COMP2_OUT / "COVID_entities.json",
}

# Model candidates: (hf_name, expected_dim)
MODEL_CANDIDATES = [
    ("allenai/specter", 768),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
]

# ── helpers ────────────────────────────────────────────────────────────────

def load_entities(entities_path: Path) -> list[dict]:
    with open(entities_path, encoding="utf-8") as f:
        return json.load(f)


def compute_representative_surface_forms(entities: list[dict]) -> dict[str, str]:
    """canonical_id → most frequent surface_form (tie-break: first occurrence)."""
    # Track frequency and first-seen order per canonical_id
    surf_freq: dict[str, Counter] = defaultdict(Counter)
    surf_first: dict[str, dict[str, int]] = defaultdict(dict)
    order = 0
    for e in entities:
        cid = e.get("canonical_id", "")
        surf = e.get("surface_form", "")
        typ = e.get("type", "Other")
        if not cid or not surf:
            continue
        surf_freq[cid][surf] += 1
        if surf not in surf_first[cid]:
            surf_first[cid][surf] = order
            order += 1
    result: dict[str, str] = {}
    for cid, freq in surf_freq.items():
        # most common surface form; tie-break by earliest first-seen
        best_surf = max(freq.keys(),
                        key=lambda s: (freq[s], -surf_first[cid][s]))
        result[cid] = best_surf
    return result


def get_entity_type(canonical_id: str, entities: list[dict]) -> str:
    """Return the most common type for a canonical_id."""
    type_counts: Counter = Counter()
    for e in entities:
        if e.get("canonical_id") == canonical_id:
            type_counts[e.get("type", "Other")] += 1
    return type_counts.most_common(1)[0][0] if type_counts else "Other"


def get_entities_in_relations(relations_path: Path,
                               entities: list[dict] | None = None) -> set[str]:
    """Return set of canonical_ids that appear in at least one relation.

    IMPORTANT: relation records carry *paper-local* entity ids
    (`source_entity_id` / `target_entity_id`, e.g. "e_P18-1.114_3"), NOT
    canonical ids. They must be remapped through the entity list's
    entity_id -> canonical_id lookup before they can be intersected with
    anything keyed by canonical_id. Comparing the two namespaces directly
    always yields an empty set (this was a real bug: the filter silently
    produced 0 candidates and wrote empty embeddings).

    This is the same remapping Component 2's report specifies for
    Component 3+: "remapped from source_entity_id/target_entity_id
    (paper-local) to their canonical_id via the entity lookup".
    """
    with open(relations_path, encoding="utf-8") as f:
        relations = json.load(f)

    eid_to_canon: dict[str, str] = {}
    if entities:
        for e in entities:
            eid = e.get("entity_id")
            cid = e.get("canonical_id")
            if eid and cid:
                eid_to_canon[eid] = cid

    canon_ids: set[str] = set()
    unmapped = 0
    for r in relations:
        for key in ("source_entity_id", "target_entity_id"):
            raw = r.get(key)
            if not raw:
                continue
            if eid_to_canon:
                mapped = eid_to_canon.get(raw)
                if mapped is None:
                    unmapped += 1
                    continue
                canon_ids.add(mapped)
            else:
                # No entity list supplied — fall back to the raw id so the
                # caller at least gets a non-empty set, but this is only
                # correct if relations already store canonical ids.
                canon_ids.add(raw)
    if unmapped:
        print(f"  [warn] {unmapped} relation endpoints had no matching "
              f"entity_id in the entities file (skipped)")
    return canon_ids


def load_model_and_tokenizer(model_name: str):
    """Try to load a model + tokenizer.  Raises on failure."""
    print(f"    Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"    Loading model: {model_name} (low_cpu_mem_usage=True)", flush=True)
    model = AutoModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
        # allenai/specter's repo only ships model.safetensors on some
        # revisions alongside the older pytorch_model.bin; recent
        # transformers refuses to load the .bin via torch.load on
        # torch<2.6 (CVE-2025-32434 guard — same issue hit in Component 2's
        # SciBERT loading). Forcing safetensors avoids that path entirely.
        # If a candidate model has no safetensors file at all, this raises
        # and the existing try/except in main() falls through to the next
        # candidate in MODEL_CANDIDATES, so behavior is still safe.
    )
    return tokenizer, model


def encode_batch(tokenizer, model, texts: list[str], device: torch.device,
                 batch_size: int) -> np.ndarray:
    """Encode a list of texts, return L2-normalized numpy array [n, dim]."""
    model.eval()
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Use CLS pooling (pooler_output) if available, else mean-pool
            if outputs.pooler_output is not None:
                embs = outputs.pooler_output.cpu().numpy()
            else:
                last = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).cpu().numpy()
                embs = (last.numpy() * mask).sum(axis=1) / mask.sum(axis=1)
            # L2-normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            all_embs.append(embs)
        # Free per-batch memory
        del inputs, outputs, embs
        gc.collect()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def validate_entities_json(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Entities JSON missing: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Entities JSON is empty or not a list: {path}")
    # Check required fields in first record
    first = data[0]
    for field in ("entity_id", "canonical_id", "surface_form", "type"):
        if field not in first:
            raise ValueError(
                f"Entities JSON missing required field '{field}' in first record: {path}"
            )
    print(f"  [validate] {path.name}: {len(data)} records OK")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 5 Step 3 — semantic embeddings (SPECTER / MiniLM)")
    parser.add_argument("--force", action="store_true",
                        help="redo embeddings even if .npy exists")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="texts per forward pass (default 8; increase if RAM allows)")
    parser.add_argument("--model", type=str, default=None,
                        help="explicit model name to use (overrides auto-detection)")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    SEM_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    for domain, path in DOMAIN_SPECS.items():
        validate_entities_json(path)

    # Load relations to find which entities participate in relations
    rel_paths = {
        "NLP": COMP2_OUT / "NLP_relations.json",
        "COVID": COMP2_OUT / "COVID_relations.json",
    }
    entities_in_rels: dict[str, set[str]] = {}
    for domain, rpath in rel_paths.items():
        if rpath.exists():
            # Pass the entity list so paper-local entity_ids in the relations
            # file get remapped to canonical_ids before intersection.
            domain_entities = load_entities(DOMAIN_SPECS[domain])
            entities_in_rels[domain] = get_entities_in_relations(
                rpath, domain_entities)
            print(f"  [info] {domain}: {len(entities_in_rels[domain])} canonical_ids "
                  f"appear in relations")
        else:
            print(f"  [warning] {domain} relations JSON not found: {rpath}")

    used_model_name: str | None = None
    used_dim: int = 0
    # The loaded model/tokenizer/device must persist ACROSS the domain loop:
    # the loading block below only runs on the first domain (it's guarded by
    # `if used_model_name is None`), so if these were assigned only inside
    # that block they'd be unbound on the second domain's encode_batch call
    # (real bug: UnboundLocalError on COVID after NLP succeeded).
    tokenizer = None
    model = None
    device = None

    # Determine model candidates
    if args.model:
        candidates = [(args.model, 0)]  # dim unknown until loaded
    else:
        candidates = MODEL_CANDIDATES

    for domain, entities_path in DOMAIN_SPECS.items():
        print(f"\n{'='*60}")
        print(f"  Domain: {domain}")
        print(f"{'='*60}")

        entities = load_entities(entities_path)
        surf_forms = compute_representative_surface_forms(entities)

        # Filter to entities that appear in relations (if relations file exists)
        rel_cid_set = entities_in_rels.get(domain, set())
        if rel_cid_set:
            candidate_cids = [cid for cid in surf_forms if cid in rel_cid_set]
            print(f"  Entities with surface forms: {len(surf_forms)}")
            print(f"  Entities in relations: {len(candidate_cids)}")
            if len(candidate_cids) == 0:
                print(f"  [WARNING] No entities appear in both surface forms and relations "
                      f"— writing empty semantic embedding")
                empty_arr = np.zeros((0, 768 if used_dim == 0 else used_dim),
                                     dtype=np.float32)
                np.save(SEM_DIR / f"{domain}.npy", empty_arr)
                (SEM_DIR / f"{domain}_semantic_id_map.json").write_text(
                    json.dumps({"_warning": "no entities in relations"}, indent=2)
                )
                continue
        else:
            candidate_cids = list(surf_forms.keys())
            print(f"  Entities with surface forms: {len(candidate_cids)} "
                  f"(no relations filter applied)")

        # Build input texts: surface_form [type]
        texts: list[str] = []
        cid_to_idx: dict[str, int] = {}
        for idx, cid in enumerate(sorted(candidate_cids)):
            surf = surf_forms.get(cid, "")
            typ = get_entity_type(cid, entities)
            text = f"{surf} [{typ}]"
            texts.append(text)
            cid_to_idx[cid] = idx

        print(f"  Encoding {len(texts)} entities as 'surface_form [type]'")

        # Load model if not yet loaded
        if used_model_name is None:
            loaded = False
            for model_name, expected_dim in candidates:
                try:
                    print(f"\n  Trying model: {model_name}", flush=True)
                    tokenizer, model = load_model_and_tokenizer(model_name)
                    device = torch.device("cuda" if torch.cuda.is_available()
                                          else "cpu")
                    model = model.to(device)
                    model.eval()

                    # Determine actual dim from a test encoding
                    test_texts = [texts[0]] if texts else ["test [Other]"]
                    with torch.no_grad():
                        t_inputs = tokenizer(
                            test_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        )
                        t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
                        t_out = model(**t_inputs)
                        if t_out.pooler_output is not None:
                            test_emb = t_out.pooler_output[0].cpu().numpy()
                        else:
                            last = t_out.last_hidden_state
                            mask = t_inputs["attention_mask"].unsqueeze(-1).cpu().numpy()
                            test_emb = (last.cpu().numpy() * mask).sum(axis=1) / mask.sum(axis=1)
                        actual_dim = test_emb.shape[0]

                    # Free test artifacts
                    del t_inputs, t_out, test_emb
                    gc.collect()

                    used_model_name = model_name
                    used_dim = actual_dim
                    loaded = True
                    print(f"  ✅ Model loaded: {model_name}, dim={actual_dim}, "
                          f"device={device}")
                    break
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {type(e).__name__}: {e}",
                          flush=True)
                    continue

            if not loaded:
                raise RuntimeError(
                    "Could not load any semantic embedding model.  "
                    "Tried: " + ", ".join(m for m, _ in candidates) +
                    ".  Install transformers and try again, or pass --model to "
                    "specify a different model."
                )

        # Encode all texts
        if tokenizer is None or model is None or device is None:
            raise RuntimeError(
                f"Model not loaded before encoding domain '{domain}'. "
                f"used_model_name={used_model_name!r}. This indicates the "
                f"model-loading block was skipped without leaving a usable "
                f"model behind."
            )
        t_enc = time.time()
        embs = encode_batch(tokenizer, model, texts, device,
                            batch_size=args.batch_size)
        enc_elapsed = time.time() - t_enc
        print(f"  Encoding done: {len(texts)} vectors, shape={embs.shape}, "
              f"time={enc_elapsed:.1f}s")

        # Save
        npy_path = SEM_DIR / f"{domain}.npy"
        np.save(npy_path, embs)
        idmap_path = SEM_DIR / f"{domain}_semantic_id_map.json"
        idmap_path.write_text(json.dumps(cid_to_idx, indent=2))

        print(f"  Saved: {npy_path.name} ({embs.shape[0]} entities, {embs.shape[1]} dim)")
        print(f"  Saved: {idmap_path.name}")

        # NOTE: previously freed model/tokenizer here between domains to save
        # memory, but that deleted the bindings without resetting
        # used_model_name, so the reuse-check above (`if used_model_name is
        # None`) never re-triggered a reload on the next domain — it just
        # tried to reuse now-deleted variables (UnboundLocalError). The
        # models used here (SPECTER ~440MB, MiniLM ~90MB) aren't large enough
        # to justify the complexity of a proper reload-on-next-domain path,
        # so the model is now simply kept loaded for the whole run instead.

    # Write build log
    log_entry = {
        "model_name": used_model_name,
        "embedding_dim": used_dim,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "domains": {},
        "total_time_s": round(time.time() - t_start, 2),
    }
    for domain in DOMAIN_SPECS:
        npy_path = SEM_DIR / f"{domain}.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            log_entry["domains"][domain] = {
                "entities_embedded": arr.shape[0],
                "embedding_dim": arr.shape[1],
            }

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log_entry["peak_memory_mb"] = round(peak / 1024 / 1024, 1)

    log_path = SEM_DIR / "semantic_build_log.json"
    log_path.write_text(json.dumps(log_entry, indent=2))
    print(f"\n  Build log: {log_path}")
    print(f"\n{'='*60}")
    print(f"  Done.  Total time: {log_entry['total_time_s']}s")
    print(f"  Model: {used_model_name} ({used_dim}-dim)")
    print(f"  Peak memory: {log_entry['peak_memory_mb']} MiB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
