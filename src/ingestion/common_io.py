"""
Component 2 — shared I/O utilities.

Reads ONLY the already-sampled corpora produced by Component 1
(data_collection/data/{nlp,covid}/*_sampled.json). Never touches raw,
un-sampled files. Never writes back into data_collection/.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]           # .../cappro
DATA_ROOT = PROJECT_ROOT / "data_collection" / "data"
COMPONENT2_ROOT = Path(__file__).resolve().parents[1]         # .../component2_entity_relation_extraction
OUTPUT_DIR = COMPONENT2_ROOT / "output"
REPORT_DIR = COMPONENT2_ROOT / "reports"
LOG_DIR = COMPONENT2_ROOT / "logs"

for d in (OUTPUT_DIR, REPORT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# The two consolidated, already-sampled corpus files (sums of the per-year
# _sampled.json files — verified identical counts: 1400 NLP / 840 COVID).
CORPUS_FILES = {
    "NLP": DATA_ROOT / "nlp" / "nlp_corpus_sampled.json",
    "COVID": DATA_ROOT / "covid" / "covid_corpus_sampled.json",
}


def stable_id_from_url_or_title(rec: dict) -> str:
    """Deterministic fallback ID when no native id field exists."""
    basis = rec.get("doi") or rec.get("url") or (rec.get("title", "") + str(rec.get("year", "")))
    return "gen_" + hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]


def resolve_paper_id(rec: dict) -> str:
    """
    The two source pipelines used different native ID fields
    (ACL Anthology records use 'anthology_id'; COVID harvester records use
    'paper_id'). Normalize to a single 'paper_id' without discarding the
    original field.
    """
    return (
        rec.get("anthology_id")
        or rec.get("paper_id")
        or stable_id_from_url_or_title(rec)
    )


def load_domain_corpus(domain: str) -> list[dict]:
    path = CORPUS_FILES[domain]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected sampled corpus not found: {path}. "
            f"This script only reads existing Component 1 outputs — it does not "
            f"regenerate sampling."
        )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        rec["paper_id"] = resolve_paper_id(rec)
        rec["domain"] = domain
    return records


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
