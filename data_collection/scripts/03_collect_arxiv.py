"""
Collector 3/4: arXiv API (Domain 1 - NLP, preprint fallback for early years)
================================================================================
Pulls cs.CL preprints per year. Useful for years/topics where ACL Anthology
coverage is thin, or to capture work that never got a formal venue.

NOTE: like the Semantic Scholar script, this must be run on your own machine -
export.arxiv.org is not reachable from the assistant's sandbox.

Usage:
    python 03_collect_arxiv.py --years 2018 2019 2020 2021 2022 2023 2024 --max-per-year 300
"""

import argparse
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

BASE_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nlp"


def fetch_year(year: int, max_results: int = 300) -> list[dict]:
    papers = []
    start = 0
    page_size = 100
    query = f"cat:cs.CL AND submittedDate:[{year}01010000 TO {year}12312359]"

    while start < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page_size, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  [warn] network error: {e}")
            break

        if resp.status_code != 200:
            print(f"  [warn] HTTP {resp.status_code}")
            break

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for entry in entries:
            arxiv_id = entry.findtext("atom:id", default="", namespaces=NS).split("/abs/")[-1]
            title = entry.findtext("atom:title", default="", namespaces=NS).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", default="", namespaces=NS).strip().replace("\n", " ")
            authors = [a.findtext("atom:name", default="", namespaces=NS)
                       for a in entry.findall("atom:author", NS)]
            published = entry.findtext("atom:published", default="", namespaces=NS)

            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published,
                "year": year,
                "source": "arxiv",
            })

        start += page_size
        time.sleep(3)  # arXiv asks for 3s between requests

    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2025)))
    parser.add_argument("--max-per-year", type=int, default=300)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        print(f"Fetching arXiv cs.CL for {year}...")
        papers = fetch_year(year, args.max_per_year)
        out_path = OUT_DIR / f"arxiv_cscl_{year}.json"
        out_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(papers)} papers -> {out_path}\n")


if __name__ == "__main__":
    main()
