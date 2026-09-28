"""
Collector 5: Multi-Source COVID-19 Paper Harvester
================================================================================
Collects comprehensive COVID-19 research papers across multiple open academic
sources (PubMed / NCBI, Europe PMC, arXiv, OpenAlex, Semantic Scholar).

Supports broad COVID research as well as domain-specific focus areas:
  - NLP / Text Mining & Misinformation Detection
  - Machine Learning, Computer Vision & Diagnostic AI
  - Epidemiological & Forecasting Models (SEIR / Deep Learning)
  - Health Informatics, Electronic Health Records (EHR) & Clinical Chatbots
  - Public Health, Contact Tracing & Social Media Surveillance

Usage:
    # 1. Collect papers for all years (2019-2024) across all CS/Health subtopics:
    python scripts/05_collect_covid_papers.py --years 2019 2020 2021 2022 2023 2024

    # 2. Collect specifically from PubMed and Europe PMC (fastest, no rate limits):
    python scripts/05_collect_covid_papers.py --sources pubmed europepmc --years 2020 2021 2022

    # 3. Focus on NLP and Misinformation:
    python scripts/05_collect_covid_papers.py --subtopics nlp misinformation --max-per-query 100

    # 4. Custom query:
    python scripts/05_collect_covid_papers.py --custom-query "COVID-19 deep learning CT scan" --years 2021 2022
"""

import argparse
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "covid"

# ============================================================
# SUBTOPICS & SEARCH QUERIES
# ============================================================

SUBTOPIC_QUERIES = {
    "nlp": [
        "COVID-19 natural language processing",
        "COVID-19 text mining literature",
        "COVID-19 clinical NLP entity extraction",
    ],
    "misinformation": [
        "COVID-19 misinformation detection",
        "COVID-19 infodemic social media fake news",
        "COVID-19 vaccine hesitancy sentiment analysis twitter",
    ],
    "machine_learning": [
        "COVID-19 machine learning diagnosis",
        "COVID-19 deep learning classification",
        "COVID-19 computer vision chest X-ray CT",
    ],
    "epidemiology": [
        "COVID-19 epidemiological forecasting model",
        "COVID-19 SEIR compartmental transmission model",
        "COVID-19 contact tracing exposure notification",
    ],
    "health_informatics": [
        "COVID-19 electronic health records EHR informatics",
        "COVID-19 conversational agent chatbot triage",
        "COVID-19 clinical decision support system",
    ],
    "genomics_bio": [
        "SARS-CoV-2 genome sequencing mutation tracking",
        "COVID-19 drug repurposing molecular docking",
        "COVID-19 immune response neutralizing antibodies",
    ],
}


def normalize_title(title: str) -> str:
    """Normalize paper title for deduplication."""
    if not title:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", title.lower())
    return cleaned[:120]


# ============================================================
# SOURCE 1: PUBMED (NCBI E-Utilities)
# ============================================================

def fetch_from_pubmed(query: str, year: int, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch COVID-19 papers from PubMed NCBI E-Utilities API."""
    papers = []
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    term = f"({query}) AND ({year}[Date - Publication])"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": min(max_results, 100),
        "sort": "pub_date",
    }

    try:
        resp = requests.get(esearch_url, params=params, timeout=20)
        if resp.status_code != 200:
            return papers

        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return papers

        # Fetch paper summaries
        sum_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        }
        sum_resp = requests.get(esummary_url, params=sum_params, timeout=25)
        if sum_resp.status_code != 200:
            return papers

        result = sum_resp.json().get("result", {})
        for pmid in id_list:
            pdata = result.get(pmid)
            if not pdata or not isinstance(pdata, dict):
                continue

            title = pdata.get("title", "").rstrip(".")
            pub_date = pdata.get("pubdate", str(year))
            journal = pdata.get("source", "")
            authors = [a.get("name") for a in pdata.get("authors", []) if a.get("name")]

            # Extract DOI if available
            article_ids = pdata.get("articleids", [])
            doi = None
            for aid in article_ids:
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            papers.append({
                "paper_id": f"pmid_{pmid}",
                "title": title,
                "abstract": "",  # PubMed esummary does not include abstract, DOI/PMID link provided
                "authors": authors,
                "year": year,
                "publish_time": pub_date,
                "journal": journal,
                "doi": doi,
                "url": url,
                "source": "pubmed",
            })

    except Exception as e:
        print(f"    [PubMed error]: {e}")

    return papers


# ============================================================
# SOURCE 2: EUROPE PMC REST API
# ============================================================

def fetch_from_europepmc(query: str, year: int, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch COVID-19 papers from Europe PMC (includes abstracts)."""
    papers = []
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    search_term = f'("{query}") AND PUB_YEAR:{year} AND (SRC:MED OR SRC:PMC OR SRC:PPR)'
    params = {
        "query": search_term,
        "format": "json",
        "pageSize": min(max_results, 100),
        "resultType": "core",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        if resp.status_code != 200:
            return papers

        data = resp.json().get("resultList", {}).get("result", [])
        for item in data:
            title = item.get("title", "").rstrip(".")
            abstract = item.get("abstractText", "")
            author_str = item.get("authorString", "")
            authors = [a.strip() for a in author_str.split(",") if a.strip()] if author_str else []
            journal = item.get("journalInfo", {}).get("journal", {}).get("title", "") if isinstance(item.get("journalInfo"), dict) else ""
            doi = item.get("doi")
            epid = item.get("id", "")
            url = f"https://europepmc.org/article/{item.get('source', 'MED')}/{epid}" if epid else (f"https://doi.org/{doi}" if doi else "")

            pub_date = item.get("firstPublicationDate") or f"{year}-01-01"

            papers.append({
                "paper_id": f"epmc_{epid}" if epid else f"epmc_{normalize_title(title)[:20]}",
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "publish_time": pub_date,
                "journal": journal,
                "doi": doi,
                "url": url,
                "source": "europe_pmc",
            })

    except Exception as e:
        print(f"    [Europe PMC error]: {e}")

    return papers


# ============================================================
# SOURCE 3: ARXIV API
# ============================================================

def fetch_from_arxiv(query: str, year: int, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch preprints from arXiv API."""
    papers = []
    base_url = "http://export.arxiv.org/api/query"
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    clean_q = query.replace("COVID-19", "").strip()
    arxiv_query = f'all:"COVID-19" AND all:"{clean_q}" AND submittedDate:[{year}01010000 TO {year}12312359]'

    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": min(max_results, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=25)
        if resp.status_code != 200:
            return papers

        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.findtext("atom:id", default="", namespaces=ns).split("/abs/")[-1]
            title = entry.findtext("atom:title", default="", namespaces=ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip().replace("\n", " ")
            authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
            published = entry.findtext("atom:published", default="", namespaces=ns)

            papers.append({
                "paper_id": f"arxiv_{arxiv_id}",
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": year,
                "publish_time": published[:10] if published else f"{year}-01-01",
                "journal": "arXiv preprint",
                "doi": None,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "source": "arxiv",
            })

    except Exception as e:
        print(f"    [arXiv error]: {e}")

    return papers


# ============================================================
# SOURCE 4: OPENALEX API
# ============================================================

def fetch_from_openalex(query: str, year: int, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch COVID-19 papers from OpenAlex API."""
    papers = []
    base_url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "filter": f"publication_year:{year},is_paratext:false",
        "per_page": min(max_results, 50),
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        if resp.status_code != 200:
            return papers

        data = resp.json().get("results", [])
        for item in data:
            title = item.get("display_name") or item.get("title") or ""
            if not title:
                continue

            # Inverted index abstract to text reconstruction
            abstract = ""
            inv_abstract = item.get("abstract_inverted_index")
            if inv_abstract and isinstance(inv_abstract, dict):
                word_positions = []
                for word, pos_list in inv_abstract.items():
                    for pos in pos_list:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)

            authors = []
            for authorship in item.get("authorships", []):
                aname = authorship.get("author", {}).get("display_name")
                if aname:
                    authors.append(aname)

            venue_info = item.get("primary_location", {}).get("source", {}) or {}
            journal = venue_info.get("display_name", "")

            doi = item.get("doi")
            openalex_id = item.get("id", "").split("/")[-1]

            papers.append({
                "paper_id": f"openalex_{openalex_id}",
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "publish_time": item.get("publication_date") or f"{year}-01-01",
                "journal": journal,
                "doi": doi,
                "url": doi or (f"https://openalex.org/{openalex_id}" if openalex_id else ""),
                "source": "openalex",
            })

    except Exception as e:
        print(f"    [OpenAlex error]: {e}")

    return papers


# ============================================================
# MAIN HARVESTER ORCHESTRATOR
# ============================================================

def harvest_papers(
    years: List[int],
    subtopics: List[str],
    sources: List[str],
    max_per_query: int,
    custom_query: str = None,
    min_abstract_len: int = 0,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("COVID-19 MULTI-SOURCE PAPER HARVESTER")
    print("=" * 70)
    print(f"Years to harvest     : {years}")
    print(f"Active sources       : {sources}")
    print(f"Target subtopics     : {subtopics if not custom_query else ['custom: ' + custom_query]}")
    print(f"Max results/query    : {max_per_query}")
    print(f"Output directory     : {DATA_DIR}")
    print("=" * 70 + "\n")

    # Build list of queries
    queries_to_run = []
    if custom_query:
        queries_to_run.append((custom_query, "custom"))
    else:
        for st in subtopics:
            for q in SUBTOPIC_QUERIES.get(st, []):
                queries_to_run.append((q, st))

    for year in sorted(years):
        print(f"\n==========================================")
        print(f"COLLECTING FOR YEAR: {year}")
        print(f"==========================================")

        year_papers: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        seen_ids: Set[str] = set()

        # Load existing file if present to append/merge new papers
        existing_file = DATA_DIR / f"covid_harvested_{year}.json"
        if existing_file.exists():
            try:
                with open(existing_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    for p in old_data:
                        t_norm = normalize_title(p.get("title", ""))
                        if t_norm:
                            seen_titles.add(t_norm)
                        if p.get("paper_id"):
                            seen_ids.add(p["paper_id"])
                        year_papers.append(p)
                print(f"Loaded {len(year_papers):,} existing papers from {existing_file.name}")
            except Exception:
                pass

        initial_count = len(year_papers)

        for query_idx, (query_text, subtopic_name) in enumerate(queries_to_run, start=1):
            print(f"\n[{query_idx}/{len(queries_to_run)}] Query: '{query_text}' (Category: {subtopic_name})")

            # 1. PubMed
            if "pubmed" in sources or "all" in sources:
                print("  -> Querying PubMed...", end=" ", flush=True)
                p_list = fetch_from_pubmed(query_text, year, max_results=max_per_query)
                added = 0
                for p in p_list:
                    t_norm = normalize_title(p["title"])
                    if t_norm and t_norm not in seen_titles and p["paper_id"] not in seen_ids:
                        seen_titles.add(t_norm)
                        seen_ids.add(p["paper_id"])
                        p["matched_subtopic"] = subtopic_name
                        year_papers.append(p)
                        added += 1
                print(f"found {len(p_list)}, added {added} new")
                time.sleep(0.4)

            # 2. Europe PMC
            if "europepmc" in sources or "all" in sources:
                print("  -> Querying Europe PMC...", end=" ", flush=True)
                e_list = fetch_from_europepmc(query_text, year, max_results=max_per_query)
                added = 0
                for p in e_list:
                    t_norm = normalize_title(p["title"])
                    if t_norm and t_norm not in seen_titles and p["paper_id"] not in seen_ids:
                        if len(p.get("abstract", "")) >= min_abstract_len:
                            seen_titles.add(t_norm)
                            seen_ids.add(p["paper_id"])
                            p["matched_subtopic"] = subtopic_name
                            year_papers.append(p)
                            added += 1
                print(f"found {len(e_list)}, added {added} new")
                time.sleep(0.4)

            # 3. arXiv
            if "arxiv" in sources or "all" in sources:
                print("  -> Querying arXiv...", end=" ", flush=True)
                a_list = fetch_from_arxiv(query_text, year, max_results=max_per_query)
                added = 0
                for p in a_list:
                    t_norm = normalize_title(p["title"])
                    if t_norm and t_norm not in seen_titles and p["paper_id"] not in seen_ids:
                        if len(p.get("abstract", "")) >= min_abstract_len:
                            seen_titles.add(t_norm)
                            seen_ids.add(p["paper_id"])
                            p["matched_subtopic"] = subtopic_name
                            year_papers.append(p)
                            added += 1
                print(f"found {len(a_list)}, added {added} new")
                time.sleep(0.5)

            # 4. OpenAlex
            if "openalex" in sources or "all" in sources:
                print("  -> Querying OpenAlex...", end=" ", flush=True)
                o_list = fetch_from_openalex(query_text, year, max_results=max_per_query)
                added = 0
                for p in o_list:
                    t_norm = normalize_title(p["title"])
                    if t_norm and t_norm not in seen_titles and p["paper_id"] not in seen_ids:
                        if len(p.get("abstract", "")) >= min_abstract_len:
                            seen_titles.add(t_norm)
                            seen_ids.add(p["paper_id"])
                            p["matched_subtopic"] = subtopic_name
                            year_papers.append(p)
                            added += 1
                print(f"found {len(o_list)}, added {added} new")
                time.sleep(0.4)

        # Save output for this year
        out_path = DATA_DIR / f"covid_harvested_{year}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(year_papers, f, ensure_ascii=False, indent=2)

        newly_added = len(year_papers) - initial_count
        print(f"\n>>> Saved {len(year_papers):,} total papers (+{newly_added:,} new) -> {out_path}")

    print("\n" + "=" * 70)
    print("ALL HARVESTING COMPLETE!")
    print(f"Files saved in: {DATA_DIR}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Harvest large-scale COVID-19 research papers across multiple open scientific APIs."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023, 2024],
        help="Publication years to collect (default: 2019 to 2024)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        choices=["all", "pubmed", "europepmc", "arxiv", "openalex"],
        help="Data sources to query (default: all)",
    )
    parser.add_argument(
        "--subtopics",
        nargs="+",
        default=list(SUBTOPIC_QUERIES.keys()),
        choices=list(SUBTOPIC_QUERIES.keys()),
        help="Subtopic categories to harvest (default: all)",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=50,
        help="Max results fetched per query per source (default: 50)",
    )
    parser.add_argument(
        "--custom-query",
        type=str,
        default=None,
        help="Provide a custom COVID search query instead of predefined subtopics",
    )
    parser.add_argument(
        "--min-abstract-len",
        type=int,
        default=30,
        help="Minimum abstract character length to retain (default: 30)",
    )

    args = parser.parse_args()

    harvest_papers(
        years=args.years,
        subtopics=args.subtopics,
        sources=args.sources,
        max_per_query=args.max_per_query,
        custom_query=args.custom_query,
        min_abstract_len=args.min_abstract_len,
    )


if __name__ == "__main__":
    main()
