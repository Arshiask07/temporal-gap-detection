import numpy as np
import pandas as pd
import networkx as nx
import config
import re


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def fused_sim(un, vn, us, vs, alpha):
    return alpha * cosine(un, vn) + (1 - alpha) * cosine(us, vs)


def build_graphs(entities_df, edges_df):
    """One nx.Graph snapshot per year from Component 4 CSV outputs."""
    graphs = {}
    for y in config.YEARS_ALL:
        g = nx.Graph()
        if entities_df is not None:
            sub = entities_df[entities_df.first_year <= y]
            g.add_nodes_from(sub.entity_id)
        if edges_df is not None:
            for _, r in edges_df[edges_df.first_observed <= y].iterrows():
                if g.has_node(r.source) and g.has_node(r.target):
                    g.add_edge(r.source, r.target)
        graphs[y] = g
    return graphs


def rank_gaps(embeddings, graphs, cit_df, alpha=config.ALPHA_CHOICES[1],
              t=config.T_LATEST, t1=config.T_PREV, min_vel=0.0, top_k=None):
    rows, checked = [], 0
    ch = embeddings["node2vec"]
    if t not in ch or t1 not in ch:
        return pd.DataFrame(), checked
    common = sorted(set(ch[t].index) & set(ch[t1].index))

    for i, u in enumerate(common):
        for v in common[i + 1:]:
            if graphs.get(t) is not None and graphs[t].has_edge(u, v):
                continue                                    # connected → not a gap
            checked += 1
            try:
                sim_t = fused_sim(ch[t].loc[u].values, ch[t].loc[v].values,
                                  embeddings["specter2"][t].loc[u].values,
                                  embeddings["specter2"][t].loc[v].values, alpha)
                sim_t1 = fused_sim(ch[t1].loc[u].values, ch[t1].loc[v].values,
                                   embeddings["specter2"][t1].loc[u].values,
                                   embeddings["specter2"][t1].loc[v].values, alpha)
            except KeyError:
                continue
            gs = sim_t * max(0.0, sim_t - sim_t1)           # Contribution 1
            if gs <= 0:
                continue
            vel = {}
            for e in (u, v):
                s = cit_df[cit_df.entity_id == e].set_index("year")["citations"] \
                    if not cit_df.empty else pd.Series(dtype=float)
                vel[e] = (s.get(config.VEL_WINDOW[1], 0)
                          - s.get(config.VEL_WINDOW[0], 0)) / (
                          config.VEL_WINDOW[1] - config.VEL_WINDOW[0])
            if max(vel.values()) < min_vel:
                continue
            hist = {}
            for y in ch:
                if u in ch[y].index and v in embeddings["specter2"].get(y, pd.DataFrame()).index:
                    hist[y] = round(fused_sim(
                        ch[y].loc[u].values, ch[y].loc[v].values,
                        embeddings["specter2"][y].loc[u].values,
                        embeddings["specter2"][y].loc[v].values, alpha), 4)
            rows.append({
                "u": u, "v": v, "sim_t": round(sim_t, 4),
                "delta_sim": round(sim_t - sim_t1, 4),
                "gap_score": round(gs, 5),
                "vel_u": round(vel[u], 1), "vel_v": round(vel[v], 1),
                "priority": round(gs * (vel[u] + vel[v]) / 2, 5),
                "sim_history": hist,
            })
    df = pd.DataFrame(rows).sort_values("priority", ascending=False)
    return (df.head(top_k) if top_k else df).reset_index(drop=True), checked


def retrospective_validate(embeddings, graphs, papers, cutoff=config.VALIDATION_CUTOFF,
                           top_k=None, alpha=config.ALPHA_CHOICES[1]):
    """Contribution 3: score gaps using only data <= cutoff, check post-cutoff.

    Returns (hits, total): how many of the top-K predicted gaps have both
    concepts co-mentioned in at least one paper published after the cutoff.
    """
    # 1. Score gaps at cutoff time using pre-cutoff embeddings & graphs
    pre_embs = _trim_embeddings(embeddings, cutoff)
    pre_graphs = {y: g for y, g in graphs.items() if y <= cutoff}
    pre_cits = _trim_citations(papers, cutoff) if not papers.empty else pd.DataFrame()
    ranked, checked = rank_gaps(pre_embs, pre_graphs, pre_cits, alpha=alpha,
                                t=cutoff, t1=cutoff - 1, top_k=top_k)
    if ranked.empty or checked == 0:
        return 0, 0

    # 2. Build post-cutoff paper text index for co-mention check
    post = papers[papers.year > cutoff]
    if post.empty:
        return 0, checked

    post_texts = (post.title.fillna("") + " " + post.abstract.fillna("")).str.lower().tolist()

    # 3. For each predicted gap, check if both concepts co-appear in any post-cutoff paper
    hits = 0
    for _, row in ranked.iterrows():
        u, v = str(row.u).lower(), str(row.v).lower()
        for txt in post_texts:
            if u in txt and v in txt:
                hits += 1
                break

    return hits, checked


def _trim_embeddings(embeddings, cutoff):
    """Keep only embedding years <= cutoff."""
    out = {}
    for channel, frames in embeddings.items():
        out[channel] = {y: f for y, f in frames.items() if y <= cutoff}
    return out


def _trim_citations(papers, cutoff):
    """Build entity→citation history from papers <= cutoff only."""
    if papers.empty:
        return pd.DataFrame()
    # Map paper-year to entities via keyword overlap (demo-friendly)
    from collections import defaultdict
    ent_cites = defaultdict(lambda: defaultdict(int))
    for _, row in papers[papers.year <= cutoff].iterrows():
        txt = (str(row.title) + " " + str(row.abstract)).lower()
        # crude: count each unique word as an "entity" citation
        for tok in set(re.findall(r"[a-z]+", txt)):
            if len(tok) > 3:
                ent_cites[tok][int(row.year)] += 1
    rows = []
    for ent, yr_map in ent_cites.items():
        for yr, cnt in yr_map.items():
            rows.append({"entity_id": ent, "year": yr, "citations": cnt})
    return pd.DataFrame(rows)
