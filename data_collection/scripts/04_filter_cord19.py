"""
Collector/Filter 4/4: CORD-19 / COVID CS-adjacent Papers (Domain 2 - COVID)
================================================================================
Collects and filters CS-adjacent COVID-19 research papers (2019-2022).

Modes:
1. Local CSV Mode:
   If `cord19_metadata_raw.csv` (or `.csv.gz`) is present in data/covid/, it
   streams the large CSV in chunks and filters by keywords.
2. Automatic API Mode (Zero Setup Fallback):
   If the local raw CSV is missing or empty, it automatically queries
   Semantic Scholar API for CS-adjacent COVID papers across 2019-2022 and
   generates the output JSON snapshots.

Usage:
    python 04_filter_cord19.py --start-year 2019 --end-year 2022
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "covid"
RAW_CSV = DATA_DIR / "cord19_metadata_raw.csv"

# ============================================================
# KEYWORDS
# ============================================================

KEYWORDS = [
    "misinformation",
    "infodemic",
    "fake news",
    "contact tracing",
    "exposure notification",
    "chatbot",
    "conversational agent",
    "epidemiological model",
    "compartmental model",
    "SEIR",
    "forecasting",
    "health informatics",
    "electronic health record",
    "natural language processing",
    "text mining",
    "NLP",
    "social media",
    "twitter",
    "sentiment analysis",
]

KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(k) for k in KEYWORDS),
    re.IGNORECASE,
)

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
API_KEY = os.environ.get("S2_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
REQUEST_DELAY = 1.0 if API_KEY else 2.5

COVID_API_QUERIES = [
    "COVID-19 misinformation",
    "COVID-19 contact tracing",
    "COVID-19 chatbot conversational agent",
    "COVID-19 epidemiological model machine learning",
    "COVID-19 health informatics EHR",
    "COVID-19 NLP natural language processing",
    "COVID-19 social media sentiment analysis",
    "COVID-19 infodemic fake news",
    "COVID-19 forecasting compartmental model SEIR",
]


# ============================================================
# TOPIC MATCHING
# ============================================================

def matches_topic(title, abstract) -> bool:
    title = "" if pd.isna(title) else str(title)
    abstract = "" if pd.isna(abstract) else str(abstract)
    text = title + " " + abstract
    return bool(KEYWORD_PATTERN.search(text))


# ============================================================
# ONLINE API FALLBACK
# ============================================================

def fetch_covid_papers_via_api(start_year: int, end_year: int, max_per_query: int = 50):
    """Fetch CS-adjacent COVID papers via Semantic Scholar API when local raw CSV is missing/empty."""
    print("\n" + "=" * 60)
    print("AUTOMATIC ONLINE FETCH (SEMANTIC SCHOLAR API)")
    print("=" * 60)
    print("Local cord19_metadata_raw.csv is empty or not downloaded.")
    print(f"Fetching CS-adjacent COVID papers for years {start_year} to {end_year} via API...\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_papers_by_year = {y: [] for y in range(start_year, end_year + 1)}
    seen_ids = set()

    for year in range(start_year, end_year + 1):
        print(f"--- Fetching Year {year} ---")

        for query in COVID_API_QUERIES:
            print(f"  Searching: '{query}' ({year})...")
            try:
                resp = requests.get(
                    SEARCH_URL,
                    params={
                        "query": query,
                        "year": str(year),
                        "fields": "paperId,title,abstract,authors,year,publicationDate,venue,journal,url",
                        "limit": max_per_query,
                    },
                    headers=HEADERS,
                    timeout=25,
                )

                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    added = 0
                    for item in data:
                        pid = item.get("paperId")
                        if not pid or pid in seen_ids:
                            continue

                        title = item.get("title") or ""
                        abstract = item.get("abstract") or ""

                        # Apply keyword matching filter to ensure CS-adjacent relevance
                        if not matches_topic(title, abstract):
                            continue

                        seen_ids.add(pid)

                        authors_list = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                        authors_str = ", ".join(authors_list) if authors_list else None

                        journal_val = item.get("venue")
                        if not journal_val and isinstance(item.get("journal"), dict):
                            journal_val = item.get("journal", {}).get("name")

                        pub_date = item.get("publicationDate") or f"{year}-01-01"

                        paper = {
                            "cord_uid": f"s2_{pid[:10]}",
                            "title": title,
                            "abstract": abstract,
                            "publish_time": str(pub_date),
                            "year": year,
                            "authors": authors_str,
                            "journal": journal_val,
                            "url": item.get("url") or f"https://www.semanticscholar.org/paper/{pid}",
                            "source": "covid_cs_api",
                        }

                        all_papers_by_year[year].append(paper)
                        added += 1

                    print(f"    -> {len(data)} results, {added} CS-adjacent matches")

                elif resp.status_code == 429:
                    print("    [Rate limited] Waiting 15s...")
                    time.sleep(15)
                else:
                    print(f"    [warn] HTTP {resp.status_code}")

            except requests.RequestException as e:
                print(f"    [warn] Request error: {e}")

            time.sleep(REQUEST_DELAY)

        # Save snapshot for the year
        papers = all_papers_by_year[year]
        out_file = DATA_DIR / f"cord19_filtered_{year}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(papers):,} papers -> {out_file}\n")

    print("=" * 60)
    print("DONE: COVID papers collected successfully!")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60 + "\n")


# ============================================================
# LOCAL CSV FILTER
# ============================================================

def filter_local_csv(input_file: Path, start_year: int, end_year: int, chunksize: int):
    """Filter local CORD-19 CSV in chunks."""
    print(f"\nProcessing local CSV: {input_file.name} ({input_file.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Start year : {start_year}")
    print(f"End year   : {end_year}")
    print(f"Chunk size : {chunksize}")

    usecols = [
        "cord_uid",
        "title",
        "abstract",
        "publish_time",
        "authors",
        "journal",
        "url",
    ]

    kept = []
    total_seen = 0
    total_year_match = 0
    total_keyword_match = 0

    try:
        reader = pd.read_csv(
            input_file,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            total_seen += len(chunk)
            chunk["publish_time"] = pd.to_datetime(chunk["publish_time"], errors="coerce")
            chunk = chunk.dropna(subset=["publish_time"])
            chunk = chunk[
                (chunk["publish_time"].dt.year >= start_year)
                & (chunk["publish_time"].dt.year <= end_year)
            ]
            total_year_match += len(chunk)

            for _, row in chunk.iterrows():
                title = row["title"]
                abstract = row["abstract"]

                if not matches_topic(title, abstract):
                    continue

                total_keyword_match += 1
                publish_date = row["publish_time"]

                paper = {
                    "cord_uid": None if pd.isna(row["cord_uid"]) else str(row["cord_uid"]),
                    "title": None if pd.isna(title) else str(title),
                    "abstract": None if pd.isna(abstract) else str(abstract),
                    "publish_time": str(publish_date.date()),
                    "year": int(publish_date.year),
                    "authors": None if pd.isna(row["authors"]) else str(row["authors"]),
                    "journal": None if pd.isna(row["journal"]) else str(row["journal"]),
                    "url": None if pd.isna(row["url"]) else str(row["url"]),
                    "source": "cord19",
                }
                kept.append(paper)

            print(
                f"Chunk {chunk_number}: "
                f"{total_seen:,} rows scanned | "
                f"{total_year_match:,} year matches | "
                f"{total_keyword_match:,} keyword matches"
            )

    except Exception as e:
        print(f"\nERROR while processing CSV: {e}")
        print("Switching to automatic online API collection...")
        fetch_covid_papers_via_api(start_year, end_year)
        return

    if not kept:
        print("\nNo matching papers found in CSV. Switching to online API collection...")
        fetch_covid_papers_via_api(start_year, end_year)
        return

    # Save by year
    by_year = {}
    for paper in kept:
        y = paper["year"]
        by_year.setdefault(y, []).append(paper)

    for year in sorted(by_year):
        papers = by_year[year]
        output_file = DATA_DIR / f"cord19_filtered_{year}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"{year}: {len(papers):,} papers -> {output_file}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Total rows scanned : {total_seen:,}")
    print(f"Date range matches : {total_year_match:,}")
    print(f"CS-adjacent papers : {total_keyword_match:,}")
    print(f"Output directory   : {DATA_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect/Filter COVID-19 research papers by year and CS keywords."
    )
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument("--force-api", action="store_true", help="Force API fetch instead of local CSV")

    args = parser.parse_args()

    print("\n======================================")
    print("CORD-19 / COVID CS-ADJACENT COLLECTOR")
    print("======================================")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_gz = DATA_DIR / "cord19_metadata_raw.csv.gz"
    has_valid_csv = (RAW_CSV.exists() and RAW_CSV.stat().st_size > 0)
    has_valid_gz = (raw_gz.exists() and raw_gz.stat().st_size > 0)

    if args.force_api or (not has_valid_csv and not has_valid_gz):
        print("\nNote: No non-empty cord19_metadata_raw.csv found locally.")
        print("Running automatic online collection via Semantic Scholar API...")
        fetch_covid_papers_via_api(args.start_year, args.end_year)
    else:
        input_file = RAW_CSV if has_valid_csv else raw_gz
        filter_local_csv(input_file, args.start_year, args.end_year, args.chunksize)


if __name__ == "__main__":
    main()