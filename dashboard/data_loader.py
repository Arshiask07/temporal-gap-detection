"""Reads the actual data_collection JSON outputs.

Expected paper-record schema (tolerant of missing keys):
{
  "paper_id": "...",        # or "id" / falls back to title hash
  "title": "...",
  "abstract": "...",
  "year": 2021,
  "authors": [...],         # optional
  "citation_count": 42      # optional; else fetched from exports/citations.csv
}
"""
import hashlib
import json
import pandas as pd

import config


def _norm_record(rec: dict, default_year: int) -> dict | None:
    abstract = rec.get("abstract") or rec.get("abstractText") or ""
    if len(str(abstract).strip()) < 40:
        return None                      # skip records without usable abstracts
    pid = rec.get("paper_id") or rec.get("id") or hashlib.md5(
        str(rec.get("title", "")).encode()).hexdigest()[:16]
    return {
        "paper_id": pid,
        "title": rec.get("title", ""),
        "abstract": abstract,
        "year": int(rec.get("year") or default_year),
        "citation_count": int(rec.get("citation_count")
                              or rec.get("citationCount") or 0),
    }


def load_papers(domain_key: str) -> pd.DataFrame:
    """Merge all matching per-year JSON files for a domain into one DataFrame."""
    cfg = config.DOMAINS[domain_key]
    rows = []
    for year in cfg["years"]:
        for pattern in cfg["patterns"]:
            path = cfg["dir"] / pattern.format(year=year)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):                       # {"papers": [...]} wrapper
                data = data.get("papers", [])
            for rec in data:
                r = _norm_record(rec, year)
                if r:
                    rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="paper_id").reset_index(drop=True)
    return df


def load_citation_history(domain_key: str) -> pd.DataFrame:
    """entity_id, year, citations — produced by Component 6 / S2 dump.
    Falls back to paper-level citation counts mapped through entity links."""
    p = config.EXPORT_DIR / f"citations_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_entity_embeddings(domain_key: str):
    """Returns {channel: {year: DataFrame(entity_id → EMB_DIM)}}.
    Files written by Component 5 (or make_demo_embeddings.py):
        exports/{channel}_{domain}_{year}.csv
    """
    short = domain_key.split()[0].lower()
    out = {}
    for channel in ("node2vec", "specter2"):
        frames = {}
        for y in config.YEARS_ALL:
            p = config.EXPORT_DIR / f"{channel}_{short}_{y}.csv"
            if p.exists():
                frames[y] = pd.read_csv(p).set_index("entity_id")
        out[channel] = frames
    return out


def load_entities(domain_key: str) -> pd.DataFrame | None:
    """Component 4 output: entity_id, label, type, first_year."""
    p = config.EXPORT_DIR / f"entities_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_edges(domain_key: str) -> pd.DataFrame | None:
    """Component 4 output triples: source, relation, target, first_observed."""
    p = config.EXPORT_DIR / f"edges_{domain_key.split()[0].lower()}.csv"
    return pd.read_csv(p) if p.exists() else None
