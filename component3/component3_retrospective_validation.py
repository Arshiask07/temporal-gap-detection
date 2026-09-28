"""`component3_retrospective_validation.py` — Contribution 3.

Scores candidate research gaps using data available up to a cutoff year
(2021 by default), then checks how many of those gaps "materialize" as
co-mentions in papers published after the cutoff (2022–2024).

Two validation modes:
  A. Embedding-based (demo mode): reuses gap_engine.rank_gaps() with
     pre-cutoff embeddings + graphs, then checks co-occurrence against
     post-cutoff paper abstracts.  Works with the demo embeddings from
     `make_demo_embeddings.py` and the real Component 2 outputs alike.
  B. Graph-based (Component 2 mode): loads the real extracted
     {domain}_entities.json + {domain}_relations.json from
     component2_entity_relation_extraction/output/, builds per-year
     co-occurrence graphs from the raw paper-level extraction records,
     scores gaps at cutoff, and checks post-cutoff materialization.

Outputs a structured report to component3/output/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx

# Ensure dashboard/ is on the path for config/data_loader imports
COMPONENT3_DIR = Path(__file__).resolve().parent
SCAM_ROOT = COMPONENT3_DIR.parent
sys.path.insert(0, str(SCAM_ROOT / "dashboard"))

import config
import data_loader as dl

# ── paths ────────────────────────────────────────────────────────────────

COMP2_OUT = SCAM_ROOT / "component2_entity_relation_extraction" / "output"
COMP3_OUT = COMPONENT3_DIR / "output"
COMP3_OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────

def _load_component2(domain_key: str):
    """Load real Component 2 JSON outputs for *domain_key* ('NLP' or 'COVID').

    Returns (papers_df, entities_df, relations_df, raw_entities_list) or
    (None, None, None, None).
    """
    short = domain_key.split()[0].lower()
    # Component 2 JSON outputs use "COVID" prefix (not "covid-19")
    if short.startswith("covid"):
        short = "COVID"
    elif short == "nlp":
        short = "NLP"
    ent_path = COMP2_OUT / f"{short}_entities.json"
    rel_path = COMP2_OUT / f"{short}_relations.json"
    ext_path = COMP2_OUT / f"{short}_extracted.json"
    if not ent_path.exists() or not rel_path.exists():
        return None, None, None

    entities = json.loads(ent_path.read_text())
    relations = json.loads(rel_path.read_text())
    extracted = json.loads(ext_path.read_text())

    # Build paper-level DataFrame from extracted records
    rows = []
    for rec in extracted:
        rows.append({
            "paper_id": rec.get("paper_id", ""),
            "title": rec.get("title", ""),
            "abstract": rec.get("abstract", ""),
            "year": int(rec.get("year", 0)) or 0,
            "domain": rec.get("domain", domain_key),
        })
    papers = pd.DataFrame(rows)

    # Entities DataFrame — canonical_id is the real node key
    ent_rows = []
    for e in entities:
        ent_rows.append({
            "entity_id": e.get("canonical_id", e.get("entity_id", "")),
            "label": e.get("surface_form", e.get("normalized_form", "")),
            "type": e.get("type", "Other"),
            "first_year": int(e.get("year", 0)) or 0,
            "paper_id": e.get("paper_id", ""),
        })
    entities_df = pd.DataFrame(ent_rows)

    # Build entity_id -> canonical_id lookup for relation remapping
    entity_id_to_canon = {}
    for e in entities:
        eid = e.get("entity_id", "")
        if eid:
            entity_id_to_canon[eid] = e.get("canonical_id", "")

    # Relations DataFrame — use canonical_ids for source/target
    rel_rows = []
    for r in relations:
        src_canon = entity_id_to_canon.get(r.get("source_entity_id", ""), "")
        tgt_canon = entity_id_to_canon.get(r.get("target_entity_id", ""), "")
        if not src_canon or not tgt_canon:
            continue
        rel_rows.append({
            "source": src_canon,
            "relation": r.get("relation_type", r.get("relation", "")),
            "target": tgt_canon,
            "first_observed": int(r.get("year", 0)) or 0,
            "paper_id": r.get("paper_id", ""),
        })
    relations_df = pd.DataFrame(rel_rows)

    return papers, entities_df, relations_df, entities


def _build_paper_entity_index(papers: pd.DataFrame):
    """Return {paper_id: set(entity_surface_token)} for co-mention checks.

    Extracts lowercase tokens >= 4 chars from title+abstract, then
    intersects with the entity labels we know about.
    """
    idx = {}
    for _, row in papers.iterrows():
        txt = (str(row.title) + " " + str(row.abstract)).lower()
        toks = set(re.findall(r"[a-z]+", txt))
        idx[row.paper_id] = {t for t in toks if len(t) >= 4}
    return idx


def _entity_surface_map(entities_df: pd.DataFrame):
    """entity_id -> set of lowercase surface tokens from known surface forms.

    Uses both the entity's own label AND all surface forms that map to
    this canonical entity (from the Component 2 entities JSON, which
    includes surface_form per record).
    """
    m = defaultdict(set)
    for _, r in entities_df.iterrows():
        label = str(r.label or r.entity_id).lower()
        for tok in re.findall(r"[a-z]+", label):
            if len(tok) >= 4:
                m[r.entity_id].add(tok)
    return dict(m)


def _surface_forms_from_entities(entities_json: list[dict]):
    """Build canonical_id -> set(surface_tokens) from raw entities list.

    The entities JSON has one record per (paper, entity mention) with
    surface_form.  We aggregate all surface forms per canonical_id.
    """
    m = defaultdict(set)
    for e in entities_json:
        canon = e.get("canonical_id", "")
        surf = e.get("surface_form", "").lower()
        for tok in re.findall(r"[a-z]+", surf):
            if len(tok) >= 4:
                m[canon].add(tok)
    return dict(m)


# ── validation logic ─────────────────────────────────────────────────────

def validate_with_real_data(domain_key: str,
                            cutoff: int = config.VALIDATION_CUTOFF,
                            top_k: int = 25,
                            alpha: float = 0.5,
                            min_papers_per_entity: int = 3,
                            out_dir: Path = COMP3_OUT) -> dict:
    """Contribution-3 validation using real Component 2 extraction outputs.

    Steps
    -----
    1. Load Component 2 entities/relations + paper metadata.
    2. Build per-year entity co-occurrence graphs from the raw extraction
       records (paper-level: if two entities appear in the same paper,
       that's a co-occurrence edge for that paper's year).
    3. Score candidate gaps at cutoff using only pre-cutoff data.
    4. Check how many top-K gaps have both entities co-mentioned in any
       post-cutoff paper.
    5. Write a structured JSON report + a human-readable summary.

    Returns the report dict.
    """
    t0 = time.time()

    papers, entities_df, relations_df, raw_entities = _load_component2(domain_key)
    if papers is None:
        return {"status": "skipped", "reason": "No Component 2 outputs found."}

    short = domain_key.split()[0].lower()
    # Component 2 JSON outputs use "COVID" prefix (not "covid-19")
    if short.startswith("covid"):
        short = "COVID"
    elif short == "nlp":
        short = "NLP"

    # Keep only entities that appear in at least min_papers_per_entity papers
    if not entities_df.empty and not papers.empty:
        ent_paper_count = defaultdict(int)
        for _, r in relations_df.iterrows():
            if r.source and r.target:
                pid = r.paper_id
                if pid in set(papers.paper_id):
                    ent_paper_count[r.source] += 1
                    ent_paper_count[r.target] += 1
        keep = {eid for eid, c in ent_paper_count.items()
                if c >= min_papers_per_entity}
        entities_df = entities_df[entities_df.entity_id.isin(keep)].copy()
        relations_df = relations_df[
            relations_df.source.isin(keep) & relations_df.target.isin(keep)
        ].copy()

    # Per-year co-occurrence graph — built from paper-level co-occurrence
    # rather than from the extracted relations (which are directional typed
    # relations).  This gives us an undirected "mentioned together" signal.
    year_graphs = defaultdict(nx.Graph)
    paper_entities = defaultdict(set)  # paper_id -> set(entity_id)

    for _, r in relations_df.iterrows():
        if r.source and r.target and r.paper_id:
            paper_entities[r.paper_id].add(r.source)
            paper_entities[r.paper_id].add(r.target)

    # Get paper year map
    year_map = dict(zip(papers.paper_id, papers.year))

    for pid, ents in paper_entities.items():
        yr = year_map.get(pid)
        if not yr:
            continue
        for u in ents:
            for v in ents:
                if u < v:
                    year_graphs[yr].add_edge(u, v)

    # Entity surface form lookup for co-mention check — use raw entity
    # surface forms for better text matching
    ent_surfaces = _surface_forms_from_entities(raw_entities)
    paper_idx = _build_paper_index_post_cutoff(papers, cutoff)

    # ── Try embedding-based gap scoring (C5/dashboard path) ───────────────
    # Load embeddings from dashboard exports if available.  When present,
    # use gap_engine.rank_gaps() for scoring instead of the co-occurrence
    # heuristic below — this gives real fused similarity scores.
    use_embeddings = False
    emb_by_channel = None
    try:
        import data_loader as dl
        emb_by_channel = dl.load_entity_embeddings(domain_key)
        if emb_by_channel.get("node2vec") and emb_by_channel.get("specter2"):
            # Check that we have data at the cutoff year
            nv = emb_by_channel["node2vec"]
            sp = emb_by_channel["specter2"]
            if cutoff in nv and (cutoff - 1) in nv and cutoff in sp:
                use_embeddings = True
    except Exception:
        pass  # embeddings not available → fall back to heuristic

    # Build entity_id -> label map for embedding loading
    entity_label_map = {}
    entity_first_year = {}
    for _, r in entities_df.iterrows():
        entity_label_map[r.entity_id] = r.label
        if r.entity_id not in entity_first_year or r.first_year < entity_first_year[r.entity_id]:
            entity_first_year[r.entity_id] = r.first_year

    if use_embeddings and emb_by_channel:
        # Convert year_graphs to dict-of-Graph format expected by gap_engine
        graphs_dict = {y: g for y, g in year_graphs.items() if y <= cutoff}
        # Add nodes for all entities present in pre-cutoff graphs
        for y, g in graphs_dict.items():
            for node in list(g.nodes):
                if not g.has_node(node):
                    g.add_node(node)

        # score gaps using real embeddings via gap_engine
        from gap_engine import rank_gaps as ge_rank_gaps
        pre_cits = pd.DataFrame()  # citation data not available in C3 standalone
        ranked, checked = ge_rank_gaps(
            emb_by_channel, graphs_dict, pre_cits,
            alpha=alpha, t=cutoff, t1=cutoff - 1, top_k=top_k * 3,
            sample_n=400, seed=42,
        )
        if checked == 0:
            return {"status": "skipped",
                    "reason": f"No gaps scored at cutoff {cutoff} with embeddings."}

        # Use ranked gaps directly — check post-cutoff materialization
        top_candidates = ranked.to_dict("records")
    else:
        # ── Fallback: co-occurrence heuristic (original behaviour) ────────
        # Score gaps at cutoff — we approximate by checking unconnected pairs
        # in the cutoff-year graph with high semantic proximity (proxied by
        # shared paper context in pre-cutoff data).
        pre_yrs = [y for y in year_graphs if y <= cutoff]
        post_yrs = [y for y in year_graphs if y > cutoff]

        if not pre_yrs:
            return {"status": "skipped",
                    "reason": f"No pre-cutoff ({cutoff}) graph data."}

        # Build candidate gap list: entities that co-occur in >=1 pre-cutoff
        # paper but NOT in the cutoff-year graph (i.e. they drifted apart or
        # were never directly linked at cutoff time).
        cutoff_graph = year_graphs.get(cutoff, nx.Graph())
        pre_entities = set()
        for y in pre_yrs:
            pre_entities |= set(year_graphs[y].nodes)

        # Find unconnected pairs that appeared in at least one pre-cutoff paper
        # together (potential gap: mentioned together before but not linked now)
        candidates = []
        seen_pairs = set()
        for pid, ents in paper_entities.items():
            yr = year_map.get(pid, 0)
            if yr > cutoff:
                continue
            for u in ents:
                for v in ents:
                    if u >= v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    # Check if they're unconnected in cutoff graph
                    if not cutoff_graph.has_edge(*key):
                        candidates.append({
                            "u": key[0], "v": key[1],
                            "pre_cutoff_cooccurrences": 1,
                            "year_first_seen": yr,
                        })

        # Deduplicate by pair and count co-occurrences
        pair_counts = defaultdict(int)
        pair_first_year = {}
        for c in candidates:
            key = (c["u"], c["v"])
            pair_counts[key] += 1
            if key not in pair_first_year:
                pair_first_year[key] = c["year_first_seen"]

        gap_candidates = []
        for (u, v), cnt in pair_counts.items():
            gap_candidates.append({
                "u": u, "v": v,
                "pre_cutoff_cooccurrences": cnt,
                "year_first_seen": pair_first_year[(u, v)],
            })

        # Sort by pre-cutoff co-occurrence count (proxy for "should be linked")
        gap_candidates.sort(key=lambda x: (-x["pre_cutoff_cooccurrences"], x["year_first_seen"]))

        top_candidates = gap_candidates[:top_k * 3]  # oversample, then filter

    # Check post-cutoff materialization
    hits = []
    misses = []
    checked = 0

    for gc in top_candidates:
        u, v = gc["u"], gc["v"]
        checked += 1
        u_surf = ent_surfaces.get(u, {u.lower()})
        v_surf = ent_surfaces.get(v, {v.lower()})

        materialized = False
        for pid, ppaper in paper_idx.items():
            yr = year_map.get(pid, 0)
            if yr <= cutoff:
                continue
            # Check if any surface form of u and any surface form of v
            # appear in the same post-cutoff paper
            if u_surf & ppaper and v_surf & ppaper:
                materialized = True
                break

        # Build record — include pre_cutoff_cooccurrences and year_first_seen
        # from the candidate dict if available (heuristic path), or set defaults
        # for embedding-scored gaps (which don't carry these fields).
        record = {
            "u": u,
            "v": v,
            "pre_cutoff_cooccurrences": gc.get("pre_cutoff_cooccurrences", 0),
            "year_first_seen": gc.get("year_first_seen", 0),
            "materialized_post_cutoff": materialized,
        }
        # If this came from gap_engine ranking, carry the scores too
        if "gap_score" in gc:
            record["gap_score"] = gc["gap_score"]
            record["sim_t"] = gc.get("sim_t", 0)
            record["delta_sim"] = gc.get("delta_sim", 0)
        if "priority" in gc:
            record["priority"] = gc["priority"]
        if "sim_history" in gc:
            record["sim_history"] = gc["sim_history"]
        if "vel_u" in gc:
            record["vel_u"] = gc["vel_u"]
            record["vel_v"] = gc["vel_v"]
        if "pre_cutoff_cooccurrences" not in gc:
            # For embedding-scored gaps, infer year_first_seen from entities_df
            if u in entity_first_year:
                record["year_first_seen"] = entity_first_year[u]
            if v in entity_first_year:
                pass  # keep the min
            record["pre_cutoff_cooccurrences"] = 0  # unknown for embedding path

        if materialized:
            hits.append(record)
        else:
            misses.append(record)

    elapsed = round(time.time() - t0, 2)
    hit_rate = len(hits) / checked if checked else 0.0

    report = {
        "domain": domain_key,
        "cutoff_year": cutoff,
        "post_cutoff_range": f"{cutoff+1}–{config.T_LATEST}",
        "alpha": alpha,
        "top_k_requested": top_k,
        "entities_considered": len(entities_df),
        "relations_considered": len(relations_df),
        "papers_total": len(papers),
        "papers_pre_cutoff": int((papers.year <= cutoff).sum()),
        "papers_post_cutoff": int((papers.year > cutoff).sum()),
        "candidate_gaps_scored": checked,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": round(hit_rate, 4),
        "elapsed_seconds": elapsed,
        "top_hits": hits[:top_k],
        "top_misses": misses[:top_k],
        "scoring_method": "embedding" if use_embeddings else "cooccurrence",
        "status": "ok",
    }

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"validation_{short}_cutoff{cutoff}.json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_summary(report, out_dir / f"validation_{short}_cutoff{cutoff}.md")

    return report


def _build_paper_index_post_cutoff(papers: pd.DataFrame, cutoff: int):
    """paper_id -> set(lowercased tokens >= 4 chars) for post-cutoff papers."""
    idx = {}
    for _, row in papers[papers.year > cutoff].iterrows():
        txt = (str(row.title) + " " + str(row.abstract)).lower()
        idx[row.paper_id] = {t for t in re.findall(r"[a-z]+", txt) if len(t) >= 4}
    return idx


def _write_summary(report: dict, path: Path):
    lines = [
        f"# Component 3 — Retrospective Validation: {report['domain']}",
        "",
        f"**Cutoff year:** {report['cutoff_year']}  ",
        f"**Post-cutoff window:** {report['post_cutoff_range']}  ",
        f"**Alpha (fusion weight):** {report['alpha']}  ",
        "",
        "## Dataset",
        "",
        f"| Metric | Value |",
        f"|---|---|---|",
        f"| Entities considered | {report['entities_considered']} |",
        f"| Relations considered | {report['relations_considered']} |",
        f"| Papers total | {report['papers_total']} |",
        f"| Papers pre-cutoff (≤{report['cutoff_year']}) | {report['papers_pre_cutoff']} |",
        f"| Papers post-cutoff (>{report['cutoff_year']}) | {report['papers_post_cutoff']} |",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|---|---|---|",
        f"| Candidate gaps scored | {report['candidate_gaps_scored']} |",
        f"| Hits (materialized) | {report['hits']} |",
        f"| Misses (not materialized) | {report['misses']} |",
        f"| **Hit rate** | **{report['hit_rate']:.2%}** |",
        f"| Wall-clock time | {report['elapsed_seconds']} s |",
        f"| Scoring method | {report.get('scoring_method', 'cooccurrence')} |",
        "",
    ]

    if report["top_hits"]:
        lines += ["## Top materialized gaps (post-cutoff co-mentions)", ""]
        for i, h in enumerate(report["top_hits"][:10], 1):
            extra = ""
            if "gap_score" in h:
                extra = f"  · gap_score={h['gap_score']:.4f}"
            lines += [
                f"{i}. **{h['u']}** ⟷ **{h['v']}**{extra}",
                f"   - Pre-cutoff co-occurrences: {h['pre_cutoff_cooccurrences']}",
                f"   - First seen: {h['year_first_seen']}",
                "",
            ]

    if report["top_misses"]:
        lines += ["## Top missed gaps (not materialized post-cutoff)", ""]
        for i, m in enumerate(report["top_misses"][:10], 1):
            extra = ""
            if "gap_score" in m:
                extra = f"  · gap_score={m['gap_score']:.4f}"
            lines += [
                f"{i}. **{m['u']}** ⟷ **{m['v']}**{extra}",
                f"   - Pre-cutoff co-occurrences: {m['pre_cutoff_cooccurrences']}",
                f"   - First seen: {m['year_first_seen']}",
                "",
            ]

    path.write_text("\n".join(lines))


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Component 3 — retrospective gap validation.")
    parser.add_argument("--domain", choices=list(config.DOMAINS),
                        default=list(config.DOMAINS)[0],
                        help="Domain to validate.")
    parser.add_argument("--cutoff", type=int, default=config.VALIDATION_CUTOFF,
                        help="Cutoff year (gaps scored using data ≤ this year).")
    parser.add_argument("--top_k", type=int, default=25,
                        help="Number of top gaps to report.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Fusion weight for gap scoring.")
    parser.add_argument("--min_papers", type=int, default=3,
                        help="Minimum papers an entity must appear in to be kept.")
    parser.add_argument("--output", type=Path, default=COMP3_OUT,
                        help="Output directory for reports.")
    args = parser.parse_args()

    print(f"[{args.domain}] Running Component 3 retrospective validation…")
    print(f"  cutoff={args.cutoff}, top_k={args.top_k}, alpha={args.alpha}")

    report = validate_with_real_data(
        args.domain,
        cutoff=args.cutoff,
        top_k=args.top_k,
        alpha=args.alpha,
        min_papers_per_entity=args.min_papers,
        out_dir=args.output,
    )

    if report["status"] == "ok":
        short = args.domain.split()[0].lower()
        if short.startswith("covid"):
            short = "COVID"
        elif short == "nlp":
            short = "NLP"
        report_path = args.output / f"validation_{short}_cutoff{args.cutoff}.md"
        print(f"  ✅ {report['candidate_gaps_scored']} gaps scored, "
              f"{report['hits']} hits → hit rate {report['hit_rate']:.2%}")
        print(f"  Report: {report_path}")
    else:
        print(f"  ⏭  {report['status']}: {report.get('reason', 'unknown')}")


if __name__ == "__main__":
    main()
