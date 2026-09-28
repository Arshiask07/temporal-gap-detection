#!/usr/bin/env python3
"""
Component 6 — Step 1: Fetch per-paper citation counts from Semantic Scholar API.

For each domain, extracts unique paper_ids from Component 2's entities.json,
looks up each paper's citation count via the S2 API (search by title), and
caches the result to component6/output/citations/{domain}_paper_citations.json.

If the API is unreachable or blocked, writes empty caches + a fallback flag and
exits 0 — the rest of the pipeline uses mention-velocity instead.

Usage:
    python3 component6/01_fetch_citations.py [--force] [--domain NLP|COVID]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import requests

# ── paths (derived from this script's location, like Component 5) ──────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP2_OUT = PROJECT_ROOT / "component2_entity_relation_extraction" / "output"
COMP6_ROOT = PROJECT_ROOT / "component6"
COMP6_OUT = COMP6_ROOT / "output"
CITATIONS_DIR = COMP6_OUT / "citations"

# ── S2 API constants (from data_collection/scripts/02_collect_semantic_scholar.py) ──

API_KEY = os.environ.get("S2_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_DELAY = 1.0 if API_KEY else 3.5  # seconds between requests

# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_unique_paper_ids(entities: list[dict]) -> set[str]:
    """Return the set of unique paper_ids mentioned in entities.json."""
    pids: set[str] = set()
    for e in entities:
        pid = e.get("paper_id", "")
        if pid:
            pids.add(pid)
    return pids


def _build_title_lookup(extracted: list[dict], paper_ids: set[str]) -> dict[str, str]:
    """Return {paper_id: title} for the given paper_ids from extracted.json."""
    lookup: dict[str, str] = {}
    for rec in extracted:
        pid = rec.get("paper_id", "")
        if pid in paper_ids and pid not in lookup:
            lookup[pid] = str(rec.get("title", "")).strip()
    return lookup


def _s2_search_citation_count(title: str) -> int | None:
    """Search S2 by title, return citationCount or None on failure."""
    try:
        resp = requests.get(
            SEARCH_URL,
            params={
                "query": title,
                "fields": "paperId,citationCount",
                "limit": 1,
            },
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                cnt = data[0].get("citationCount")
                if cnt is not None:
                    return int(cnt)
        elif resp.status_code == 429:
            print("  [RATE] Rate limited — backing off 30s")
            time.sleep(30)
    except requests.RequestException as exc:
        raise RuntimeError(f"S2 API request failed: {exc}") from exc
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 6 Step 1 — fetch per-paper citation counts from S2 API"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch even if citation cache already exists",
    )
    parser.add_argument(
        "--domain",
        choices=["NLP", "COVID"],
        help="Single domain (default: both NLP and COVID)",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.time()

    domains = [args.domain] if args.domain else ["NLP", "COVID"]

    # Probe API reachability once before doing any real work
    api_reachable = _probe_api()

    CITATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not api_reachable:
        _write_fallback(domains)
        _report_done(t_start, "FALLBACK: S2 API unreachable — mention velocity will be used")
        return

    for domain in domains:
        _fetch_domain(domain, args.force)

    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDone.  Total time: {elapsed:.1f}s   Peak memory: {peak / 1024 / 1024:.1f} MiB")


def _probe_api() -> bool:
    """Try a trivial S2 search to check reachability.  Returns True if 200."""
    try:
        resp = requests.get(
            SEARCH_URL,
            params={"query": "the", "fields": "paperId", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        ok = resp.status_code == 200
        print(f"API probe: {'reachable' if ok else 'UNREACHABLE'} "
              f"(status={resp.status_code})")
        return ok
    except requests.RequestException as exc:
        print(f"API probe: UNREACHABLE ({exc})")
        return False


def _fetch_domain(domain: str, force: bool) -> None:
    """Fetch citation counts for one domain, cache to disk."""
    ent_path = COMP2_OUT / f"{domain}_entities.json"
    ext_path = COMP2_OUT / f"{domain}_extracted.json"
    cache_path = CITATIONS_DIR / f"{domain}_paper_citations.json"
    fallback_flag = CITATIONS_DIR / f"{domain}_FALLBACK.txt"

    # Input validation
    for p in [ent_path, ext_path]:
        if not p.exists():
            raise FileNotFoundError(f"Component 2 input missing: {p}")

    # Skip if cache exists and not forcing
    if cache_path.exists() and not force and not fallback_flag.exists():
        print(f"[SKIP] {domain}: citation cache exists ({cache_path}) — use --force to refetch")
        return

    # If a previous run set the fallback flag for this domain, honour it
    if fallback_flag.exists():
        print(f"[SKIP] {domain}: fallback flag present — API was unreachable before. "
              f"Delete {fallback_flag} to retry.")
        return

    print(f"\n{'='*50}")
    print(f"  Domain: {domain}  —  fetching citation counts from S2 API")
    print(f"{'='*50}")

    entities = json.loads(ent_path.read_text(encoding="utf-8"))
    extracted = json.loads(ext_path.read_text(encoding="utf-8"))

    paper_ids = _extract_unique_paper_ids(entities)
    print(f"  Unique paper_ids in entities.json: {len(paper_ids)}")

    title_lookup = _build_title_lookup(extracted, paper_ids)
    print(f"  Papers with titles in extracted.json: {len(title_lookup)}")

    if not title_lookup:
        print(f"  [WARNING] No titles found for {domain} — cannot fetch citations. "
              f"Switching to mention-velocity fallback.")
        _write_fallback([domain])
        return

    # Fetch citation counts
    citations: dict[str, int] = {}
    total = len(title_lookup)
    failed: list[str] = []

    for i, (pid, title) in enumerate(sorted(title_lookup.items()), 1):
        try:
            cnt = _s2_search_citation_count(title)
            if cnt is not None:
                citations[pid] = cnt
            else:
                failed.append(pid)
        except RuntimeError as exc:
            # API became unreachable mid-fetch — write partial cache + fallback
            print(f"  [ERROR] API unreachable mid-fetch: {exc}")
            print(f"  Writing partial cache ({len(citations)} papers) + fallback flag")
            cache_path.write_text(json.dumps(citations, indent=2), encoding="utf-8")
            fallback_flag.write_text(
                f"API unreachable during fetch at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Partial cache: {len(citations)} papers fetched, {len(failed)} failed\n"
                f"Original error: {exc}"
            )
            return

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} done  "
                  f"(cached: {len(citations)}, failed: {len(failed)}, "
                  f"remaining: {total - i})")

        time.sleep(REQUEST_DELAY)

    # Write cache
    cache_path.write_text(json.dumps(citations, indent=2), encoding="utf-8")
    print(f"\n  Saved {len(citations)} citation counts to {cache_path}")
    if failed:
        print(f"  [WARNING] {len(failed)} papers returned no citation count (new papers, etc.)")


def _write_fallback(domains: list[str]) -> None:
    """Write empty citation caches + fallback flags for the given domains."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for domain in domains:
        cache_path = CITATIONS_DIR / f"{domain}_paper_citations.json"
        fallback_path = CITATIONS_DIR / f"{domain}_FALLBACK.txt"
        cache_path.write_text("{}", encoding="utf-8")
        fallback_path.write_text(
            f"S2 API unreachable at {ts}\n"
            f"Using mention-velocity fallback for {domain}.\n"
            f"To retry API fetch: delete {fallback_path} and re-run with --force."
        )
        print(f"  [FALLBACK] {domain}: wrote empty cache + fallback flag")


def _report_done(t_start: float, message: str) -> None:
    """Print final timing and exit."""
    elapsed = time.time() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n{message}")
    print(f"Done.  Total time: {elapsed:.1f}s   Peak memory: {peak / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
