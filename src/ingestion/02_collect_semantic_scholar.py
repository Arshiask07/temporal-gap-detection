"""
Collector 2/4: Semantic Scholar API
======================================
Two jobs, both handled by this script:

  (a) NLP domain: citation-count + gap-filling for papers already pulled from
      ACL Anthology (adds citationCount, influentialCitationCount, s2 paperId)
  (b) COVID domain: primary search for CS-adjacent COVID papers, 2023-2024
      (CORD-19 was frozen mid-2022, so this is how you fill 2023-2024)

NOTE: This could NOT be run inside the assistant's sandbox (api.semanticscholar.org
is not on the sandbox's network allowlist -> HTTP 403 there). Run this on your own
machine, where it will work normally.

Usage:
    # (a) enrich existing ACL Anthology papers with citation counts
    python 02_collect_semantic_scholar.py enrich --year 2023

    # (b) search for COVID CS-adjacent papers for 2023-2024 gap
    python 02_collect_semantic_scholar.py covid-search --years 2023 2024

Get a free API key (optional but strongly recommended - raises your rate limit
from 100 req/5min to 1 req/sec sustained): https://www.semanticscholar.org/product/api
Set it as an env var: export S2_API_KEY=your_key_here
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

API_KEY = os.environ.get("S2_API_KEY")  # optional but recommended
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# CS-adjacent COVID subtopics per your Domain 2 spec
COVID_CS_KEYWORDS = [
    "COVID-19 misinformation detection",
    "COVID-19 contact tracing app",
    "COVID-19 chatbot",
    "COVID-19 epidemiological modeling machine learning",
    "COVID-19 health informatics NLP",
    "COVID-19 NLP text mining",
    "COVID-19 social media analysis",
]

REQUEST_DELAY = 1.1 if API_KEY else 3.5  # seconds between requests, be conservative


def enrich_with_citations(year: int):
    """Add citation counts to an existing ACL Anthology snapshot for `year`."""
    in_path = DATA_DIR / "nlp" / f"acl_anthology_{year}.json"
    if not in_path.exists():
        print(f"Missing {in_path} - run 01_collect_acl_anthology.py first.")
        return

    papers = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"Enriching {len(papers)} papers from {year} with citation data...")

    # Semantic Scholar batch endpoint takes up to 500 title-matched IDs at once,
    # but since we don't have S2 paper IDs yet, we search by title one at a time.
    # (Slower, but reliable - no ambiguous ID mapping.)
    for i, paper in enumerate(papers):
        title = paper["title"]
        try:
            resp = requests.get(
                SEARCH_URL,
                params={"query": title, "fields": "paperId,citationCount,influentialCitationCount", "limit": 1},
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    match = data[0]
                    paper["s2_paper_id"] = match.get("paperId")
                    paper["citation_count"] = match.get("citationCount")
                    paper["influential_citation_count"] = match.get("influentialCitationCount")
            elif resp.status_code == 429:
                print("  Rate limited, backing off 30s...")
                time.sleep(30)
                continue
        except requests.RequestException as e:
            print(f"  [warn] {title[:50]}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(papers)} done")
        time.sleep(REQUEST_DELAY)

    out_path = DATA_DIR / "nlp" / f"acl_anthology_{year}_enriched.json"
    out_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved enriched snapshot -> {out_path}")


def covid_search(years: list[int]):
    """Search Semantic Scholar directly for CS-adjacent COVID papers (2023-2024 gap-fill)."""
    out_dir = DATA_DIR / "covid"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        all_papers = []
        seen_ids = set()
        for keyword in COVID_CS_KEYWORDS:
            print(f"Searching '{keyword}' for {year}...")
            try:
                resp = requests.get(
                    SEARCH_URL,
                    params={
                        "query": keyword,
                        "year": str(year),
                        "fields": "paperId,title,abstract,authors,year,citationCount,externalIds",
                        "limit": 50,
                    },
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200:
                    for p in resp.json().get("data", []):
                        if p["paperId"] not in seen_ids:
                            seen_ids.add(p["paperId"])
                            all_papers.append({
                                "s2_paper_id": p["paperId"],
                                "title": p.get("title"),
                                "abstract": p.get("abstract"),
                                "authors": [a.get("name") for a in p.get("authors", [])],
                                "year": p.get("year"),
                                "citation_count": p.get("citationCount"),
                                "matched_keyword": keyword,
                                "source": "semantic_scholar",
                            })
                elif resp.status_code == 429:
                    print("  Rate limited, backing off 30s...")
                    time.sleep(30)
            except requests.RequestException as e:
                print(f"  [warn] {e}")
            time.sleep(REQUEST_DELAY)

        out_path = out_dir / f"covid_semantic_scholar_{year}.json"
        out_path.write_text(json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(all_papers)} papers -> {out_path}\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    enrich_p = sub.add_parser("enrich", help="add citation counts to existing ACL Anthology snapshot")
    enrich_p.add_argument("--year", type=int, required=True)

    covid_p = sub.add_parser("covid-search", help="search for CS-adjacent COVID papers")
    covid_p.add_argument("--years", nargs="+", type=int, default=[2023, 2024])

    args = parser.parse_args()
    if args.mode == "enrich":
        enrich_with_citations(args.year)
    elif args.mode == "covid-search":
        covid_search(args.years)


if __name__ == "__main__":
    main()
