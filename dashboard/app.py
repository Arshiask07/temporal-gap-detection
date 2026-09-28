import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
from pathlib import Path
import sys

# Make component3/ discoverable for real-data validation import
SCAM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCAM_ROOT / "component3"))

import config
import data_loader as dl
import gap_engine as ge  # type: ignore[import-not-found]

st.set_page_config(page_title="Emerging Research Gap Explorer",
                   page_icon="🔬", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
domain = st.sidebar.selectbox("Domain", list(config.DOMAINS))
alpha = st.sidebar.select_slider(
    "Fusion weight α",
    options=[0.0, 0.3, 0.5, 0.7, 1.0], value=0.5,
    help="Matches ablation sweep {0.3, 0.5, 0.7}. "
         "1.0 = pure structural, 0.0 = pure citation-semantic.")
min_vel = st.sidebar.number_input("Min citation velocity", 0, 500, 5)
top_k = st.sidebar.slider("Show top-K gaps", 5, 50, 15)

# Cache version key — increment when user explicitly requests recompute
if "version" not in st.session_state:
    st.session_state["version"] = 0

@st.cache_resource(show_spinner="Loading frozen corpus snapshot…")
def load_domain(dom, version):
    papers = dl.load_papers(dom)
    ents = dl.load_entities(dom)
    edges = dl.load_edges(dom)
    embs = dl.load_entity_embeddings(dom)
    cits = dl.load_citation_history(dom)
    graphs = ge.build_graphs(ents, edges)
    return papers, ents, edges, embs, cits, graphs

papers, ents, edges, embs, cits, graphs = load_domain(domain, st.session_state["version"])

if papers.empty:
    st.error(f"No papers found under `{config.DATA_ROOT}` — check "
             "`data_collection/scripts/06_preprocess_and_sample.py` output.")
    st.stop()

years_avail = [y for y, g in graphs.items() if len(g.nodes)]
if not years_avail:
    st.info("No graph snapshots yet — run the extraction pipeline "
            "(`make_demo_embeddings.py` fills placeholders).")
    st.stop()
YR_MIN, YR_MAX = min(years_avail), max(years_avail)

# ── Header ────────────────────────────────────────────────────────────
st.title("🔬 Temporal Research Gap Explorer")
st.caption(f"{domain} · {len(papers):,} papers · snapshots {YR_MIN}–{YR_MAX} · α={alpha}")

tab_overview, tab_gaps, tab_graph, tab_validate = st.tabs(
    ["📊 Corpus Overview", "🏆 Top Gaps", "🕸️ Graph Timeline", "✅ Validation"])

# ── Tab 1: Corpus overview ────────────────────────────────────────────
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    n_ents = len(ents) if ents is not None else 0
    n_edges = len(edges) if edges is not None else 0
    years_present = sorted(papers.year.unique())
    c1.metric("Papers indexed", f"{len(papers):,}")
    c2.metric("Concepts extracted", f"{n_ents:,}")
    c3.metric("Typed relations", f"{n_edges:,}")
    c4.metric("Coverage", f"{years_present[0]}–{years_present[-1]}")

    per_year = papers.groupby("year").size().reindex(years_present, fill_value=0)
    fig = go.Figure(go.Bar(x=per_year.index.astype(str), y=per_year.values,
                           marker_color="#6366f1"))
    fig.update_layout(title="Papers per year (frozen snapshot)", height=320,
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Graph growth overlay (entities/relations appearing per snapshot year)
    if ents is not None and not ents.empty and edges is not None and not edges.empty:
        ent_growth = ents.groupby("first_year").size().reindex(
            range(YR_MIN, YR_MAX + 1), fill_value=0)
        edge_growth = edges.groupby("first_observed").size().reindex(
            range(YR_MIN, YR_MAX + 1), fill_value=0)
        figg = go.Figure()
        figg.add_bar(name="New concepts", x=list(ent_growth.index.astype(str)),
                     y=ent_growth.values, marker_color="#34d399", opacity=0.75)
        figg.add_bar(name="New relations", x=list(edge_growth.index.astype(str)),
                     y=edge_growth.values, marker_color="#60a5fa", opacity=0.75)
        figg.update_layout(barmode="overlay",
                           title="Knowledge graph growth per year", height=300,
                           margin=dict(t=40, b=10))
        st.plotly_chart(figg, use_container_width=True)

# ── Tab 2: Ranked gaps ────────────────────────────────────────────────
with tab_gaps:
    # Only recompute when user explicitly asks — slider changes don't trigger it
    compute_key = f"gaps_{domain.replace(' ', '_')}_{alpha}_{min_vel}"
    if compute_key not in st.session_state:
        st.session_state[compute_key] = None

    col_btn, col_info = st.columns([1.5, 4])
    with col_btn:
        if st.button("🔄 Score gaps", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {top_k} gaps (α={alpha}, sample_n=400)…"):
                try:
                    gaps, checked = ge.rank_gaps(
                        embs, graphs, cits, alpha=alpha,
                        min_vel=min_vel, top_k=top_k,
                        sample_n=400,
                    )
                    st.session_state[compute_key] = (gaps, checked)
                except Exception as e:
                    st.error(f"Gap scoring failed: {e}")
                    st.session_state[compute_key] = (None, 0)

    gaps_data = st.session_state.get(compute_key)
    if gaps_data is None:
        st.warning("Click **Score gaps** to compute the top-ranked research gaps.")
    else:
        gaps, checked = gaps_data
        if gaps is None or gaps.empty:
            st.warning("No gaps found. Try a different α or domain, or check "
                       "that embedding CSVs exist in `dashboard/exports/`.")
        else:
            with col_info:
                st.info(f"Scored **{checked:,}** unconnected concept pairs · "
                        f"priority(u,v) = [α·sim_n2v + (1−α)·sim_sp2](t) × Δsim × mean(vel)")
            for i, row in gaps.iterrows():
                with st.container(border=True):
                    col_r, col_pair, col_m = st.columns([0.07, 0.38, 0.55])
                    col_r.markdown(f"### {i+1}")
                    col_pair.markdown(
                        f"**{row.u}** ⟷ **{row.v}**\n\n"
                        f"`sim_t={row.sim_t}` · `Δsim={row.delta_sim}`\n\n"
                        f"`vel(u)={row.vel_u}` · `vel(v)={row.vel_v}`")
                    m1, m2, m3 = col_m.columns(3)
                    m1.metric("Gap score", row.gap_score)
                    m2.metric("Priority", row.priority)
                    m3.metric("Mean velocity", round((row.vel_u + row.vel_v) / 2, 1))
                    if row.sim_history:
                        tf = go.Figure(go.Scatter(
                            x=list(row.sim_history.keys()),
                            y=list(row.sim_history.values()),
                            mode="lines+markers", line=dict(color="#f59e0b")))
                        tf.update_layout(height=140,
                                         margin=dict(l=0, r=0, t=5, b=0),
                                         yaxis_range=[-0.05, 1.05],
                                         yaxis_title="fused sim")
                        col_m.plotly_chart(tf, use_container_width=True)

# ── Tab 3: Graph timeline ─────────────────────────────────────────────
with tab_graph:
    yr_lo, yr_hi = st.slider("Snapshot year range", YR_MIN, YR_MAX, (YR_MIN, YR_MAX))
    focus = st.text_input("Focus concept (optional)", "")
    show_edges = st.checkbox("Show typed relations", True)
    max_nodes = st.slider("Max nodes to render", 50, 2000, 500, step=50,
                          help="Larger graphs render slower. Lower this to keep the "
                               "visualization responsive. Nodes beyond the limit are omitted.")

    # Cache the merged graph by year range so slider changes don't rebuild from scratch
    # Version bumped on each server restart so stale caches don't survive
    if "graph_cache_version" not in st.session_state:
        st.session_state["graph_cache_version"] = hash(Path(__file__).resolve().stat().st_mtime)
    graph_cache_key = (f"merged_graph_{domain.replace(' ', '_')}_{yr_lo}_{yr_hi}"
                       f"_{focus}_{max_nodes}_{st.session_state['graph_cache_version']}")
    if graph_cache_key not in st.session_state:
        st.session_state[graph_cache_key] = None

    merged = None
    if st.session_state[graph_cache_key] is None and ents is not None and edges is not None:
        # Build debounced — only when slider settles
        merged = nx.Graph()
        for row in ents[(ents.first_year >= yr_lo) & (ents.first_year <= yr_hi)].itertuples():
            merged.add_node(row.entity_id, type=row.type, label=row.label)
        for row in edges[(edges.first_observed >= yr_lo)
                          & (edges.first_observed <= yr_hi)].itertuples():
            if merged.has_node(row.source) and merged.has_node(row.target):
                merged.add_edge(row.source, row.target, rel=row.relation)
        if focus and focus in merged:
            merged = nx.ego_graph(merged, focus, radius=2)
        # Truncate to max_nodes if needed — prefer nodes with highest degree centrality
        if len(merged.nodes) > max_nodes:
            deg = nx.degree_centrality(merged)
            top_nodes = sorted(deg, key=lambda n: deg[n], reverse=True)[:max_nodes]
            merged = merged.subgraph(top_nodes).copy()
        st.session_state[graph_cache_key] = merged

    merged = st.session_state.get(graph_cache_key)

    if merged is None:
        st.info("Enter a focus concept or adjust the year range to build the graph.")
    elif len(merged.nodes) == 0:
        st.info("No nodes in this selection — widen the year range or clear the focus.")
    elif len(merged.nodes) > 400:
        st.warning(f"{len(merged.nodes)} nodes selected — rendering may be slow. "
                   "Narrow the year range or set a focus concept.")
        _render = st.checkbox("Render anyway", value=False)
    else:
        _render = True

    if len(merged.nodes) > 0 and _render:
        # ── Render the merged graph with PyVis ──────────────────────────────
        # Human-readable node labels from entity data (not canonical IDs)
        # Human-readable edge labels from relation-type mapping
        net = Network(height="600px", bgcolor="#0e1117", font_color="white",
                      heading="", notebook=False)
        palette = {"Method": "#60a5fa", "Task": "#34d399", "Metric": "#fbbf24",
                   "Material": "#f472b6", "Dataset": "#a78bfa", "Model": "#fb923c"}

        def _good_label(node_id, attr):
            """Return a display label: prefer the entity label field, with
            quality checks. Short/empty labels fall back to the canonical ID;
            very long labels are truncated."""
            raw = attr.get("label", "")
            if not raw or len(raw) <= 2:
                return node_id       # e.g. "LL", "ja" → show "canon_Other_00129"
            if len(raw) > 30:
                return raw[:27] + "…"  # e.g. "German↔English and Chinese→English translation tasks" → truncated
            return raw

        for n, d in merged.nodes(data=True):
            net.add_node(n,
                         label=_good_label(n, d),
                         color=palette.get(d.get("type"), "#94a3b8"),
                         size=12)

        # ── Human-readable edge labels ──────────────────────────────────────
        REL_LABELS = {
            "METHOD_APPLIED_TO":      "Method applied to",
            "METHOD_EVALUATED_BY":    "Method evaluated by",
            "USED_FOR":               "Used for",
            "ENTITY_ASSOCIATED_WITH_ENTITY": "Associated with",
        }
        if show_edges:
            for s, t_, d in merged.edges(data=True):
                net.add_edge(s, t_,
                             title=REL_LABELS.get(d.get("rel", ""),
                                                  d.get("rel", "")),
                             color="#475569")

        html = net.generate_html(notebook=False)
        # Disable long stabilization — 1000 default iterations = 30-60s of
        # browser-side settling for 500 nodes. 10 iterations settles in <1s.
        html = html.replace('"iterations": 1000', '"iterations": 10')
        html = html.replace('"fit": true', '"fit": false')
        components.html(html, height=620, scrolling=False)

# ── Tab 4: Retrospective validation ───────────────────────────────────
with tab_validate:
    st.subheader("Retrospective hit rate (Contribution 3)")
    st.caption("Gaps predicted using data ≤2021 only, then checked against "
               "real co-mentions in publications from 2022–2024.")

    val_mode = st.radio(
        "Validation mode",
        ["Demo embeddings (from make_demo_embeddings.py)",
         "Real extraction data (Component 2 JSON)"],
        horizontal=True,
        help="Demo mode scores gaps from synthetic embeddings. "
             "Real mode loads the actual SciBERT-extracted entities/relations "
             "and checks surface-form co-mentions in post-cutoff papers.",
    )

    if val_mode.startswith("Demo"):
        st.info("Using frozen demo embeddings from `dashboard/exports/`.")
        if st.button("▶ Run demo validation", type="primary"):
            with st.spinner(f"Re-scoring gaps at t={config.VALIDATION_CUTOFF}…"):
                try:
                    hits, total = ge.retrospective_validate(
                        embs, graphs, papers, cutoff=config.VALIDATION_CUTOFF,
                        top_k=top_k, alpha=alpha)
                    if total == 0:
                        st.error("No candidate gaps could be scored at the cutoff year — "
                                 "check that pre-cutoff embeddings exist.")
                    elif hits == 0:
                        st.warning(f"0/{total} predicted gaps realized post-cutoff. "
                                   "(Expected with demo embeddings.)")
                    else:
                        rate = hits / total
                        st.success(f"Demo hit rate @top-{top_k}: **{rate:.1%}** "
                                   f"({hits}/{total})")
                except Exception as e:
                    st.error(f"Demo validation failed: {e}")

    else:
        st.info("Loading real Component 2 extraction outputs from "
                "`component2_entity_relation_extraction/output/`.")
        if st.button("▶ Run real-data validation", type="primary"):
            try:
                import component3_retrospective_validation as c3
            except ImportError:
                st.error("Cannot import `component3_retrospective_validation` — "
                         "ensure `component3/` is on the Python path.")
            else:
                with st.spinner("Loading extraction data & scoring gaps…"):
                    try:
                        report = c3.validate_with_real_data(
                            domain, cutoff=config.VALIDATION_CUTOFF,
                            top_k=top_k, alpha=alpha,
                            min_papers_per_entity=3,
                            out_dir=Path(__file__).resolve().parent
                            / "exports" / "component3_real_validation",
                        )
                        if report["status"] == "ok":
                            rate = report["hit_rate"]
                            st.success(
                                f"Real-data hit rate @top-{top_k} "
                                f"(cutoff {config.VALIDATION_CUTOFF}): "
                                f"**{rate:.1%}** ({report['hits']}/{report['candidate_gaps_scored']})"
                            )
                            st.info(
                                f"{report['entities_considered']:,} entities · "
                                f"{report['relations_considered']:,} relations · "
                                f"{report['papers_pre_cutoff']:,} pre-cutoff papers · "
                                f"{report['papers_post_cutoff']:,} post-cutoff papers · "
                                f"{report['elapsed_seconds']}s"
                            )
                            if report["top_hits"]:
                                st.subheader("Top materialized gaps")
                                for i, h in enumerate(report["top_hits"][:10], 1):
                                    st.write(f"{i}. **{h['u']}** ⟷ **{h['v']}** "
                                             f"— {h['pre_cutoff_cooccurrences']} pre-cutoff "
                                             f"co-occurrences, first seen {h['year_first_seen']}")
                        else:
                            st.warning(f"Validation skipped: {report.get('reason', 'unknown')}")
                    except Exception as e:
                        st.error(f"Real-data validation failed: {e}")

st.divider()
st.caption("Frozen snapshot — no live API calls. Data source: "
           f"`{config.DATA_ROOT.relative_to(config.CAPPRO_ROOT)}`")
