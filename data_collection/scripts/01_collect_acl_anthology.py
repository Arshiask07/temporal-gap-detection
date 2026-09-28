"""
Collector 1/4: ACL Anthology (Domain 1 - NLP, primary corpus)
================================================================
Source: ACL Anthology's own GitHub repo (raw XML), NOT the website itself.
This is the standard, documented way researchers pull ACL Anthology data in bulk.

Run ONCE. Output is a static snapshot -> data/nlp/acl_anthology_<year>.json

Usage:
    python 01_collect_acl_anthology.py --years 2018 2019 2020 2021 2022 2023 2024

Notes:
- The Anthology stores one XML file per "collection" (roughly: venue+year), e.g.
  2023.acl.xml, 2023.emnlp-main.xml, 2023.findings.xml, 2023.naacl-main.xml
- We target the major general NLP venues per year. Add more IDs to VENUE_IDS
  if you want deeper coverage (workshops, *SEM, CoNLL, etc.)
- Some venue IDs don't exist in every year (e.g. NAACL is not held every year) -
  the script skips those gracefully (404 -> skip, logged, not fatal).
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{collection_id}.xml"

# Major general-NLP venues to pull per year (used for years >= 2020, which use
# the "YYYY.venue" collection-id scheme).
VENUE_IDS = [
    "acl",              # ACL main conference
    "emnlp",            # EMNLP main
    "naacl",            # NAACL main (not held every year)
    "findings",         # Findings of ACL (catch-all volume, varies by year)
    "eacl",             # EACL main (not held every year)
    "coling",           # COLING (biennial)
    "conll",            # CoNLL
]

# Pre-2020 Anthology IDs use single-letter prefixes instead of "YYYY.venue".
# P=ACL, D=EMNLP, N=NAACL, C=COLING, K=CoNLL, E=EACL.
OLD_SCHEME_PREFIXES = {
    "acl": "P",
    "emnlp": "D",
    "naacl": "N",
    "coling": "C",
    "conll": "K",
    "eacl": "E",
}


def collection_ids_for(year: int, venue: str) -> list[str]:
    """Return the Anthology collection id(s) to try for a given year+venue."""
    if year >= 2020:
        return [f"{year}.{venue}"]
    prefix = OLD_SCHEME_PREFIXES.get(venue)
    if prefix is None:
        return []  # "findings" didn't exist pre-2020, e.g.
    yy = str(year)[2:]  # e.g. 2018 -> "18"
    return [f"{prefix}{yy}"]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nlp"


def fetch_collection_xml(collection_id: str) -> str | None:
    url = BASE_URL.format(collection_id=collection_id)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return resp.text
        return None  # 404 etc -> doesn't exist, skip silently
    except requests.RequestException as e:
        print(f"  [warn] network error for {collection_id}: {e}")
        return None


def parse_papers(xml_text: str, target_year: int, venue: str, collection_id: str) -> list[dict]:
    papers = []
    root = ET.fromstring(xml_text)
    for volume in root.findall("volume"):
        meta = volume.find("meta")
        vol_year = target_year
        if meta is not None:
            vol_year = int(meta.findtext("year", default=str(target_year)))
        if vol_year != target_year:
            continue  # a stub/co-located pointer for another year, skip

        for paper in volume.findall("paper"):
            title_el = paper.find("title")
            if title_el is None:
                continue
            title = "".join(title_el.itertext()).strip()

            authors = []
            for author in paper.findall("author"):
                first = author.findtext("first", default="")
                last = author.findtext("last", default="")
                authors.append(f"{first} {last}".strip())

            abstract_el = paper.find("abstract")
            abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

            paper_id = paper.get("id", "")
            volume_id = volume.get("id", "")
            anthology_id = f"{collection_id}-{volume_id}.{paper_id}"

            papers.append({
                "anthology_id": anthology_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": vol_year,
                "venue": venue,
                "source": "acl_anthology",
            })
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2025)))
    parser.add_argument("--venues", nargs="+", default=VENUE_IDS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        year_papers = []
        seen_collections = set()
        for venue in args.venues:
            for collection_id in collection_ids_for(year, venue):
                if collection_id in seen_collections:
                    continue
                seen_collections.add(collection_id)
                print(f"Fetching {collection_id} ({venue} {year}) ...")
                xml_text = fetch_collection_xml(collection_id)
                if xml_text is None:
                    print(f"  -> not found, skipping")
                    continue
                try:
                    papers = parse_papers(xml_text, year, venue, collection_id)
                    print(f"  -> {len(papers)} papers")
                    year_papers.extend(papers)
                except ET.ParseError as e:
                    print(f"  [warn] parse error: {e}")
                time.sleep(0.5)  # be polite to GitHub's raw file server

        out_path = OUT_DIR / f"acl_anthology_{year}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(year_papers, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(year_papers)} papers -> {out_path}\n")


if __name__ == "__main__":
    main()
