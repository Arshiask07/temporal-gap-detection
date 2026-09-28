"""Temporary stand-in for Components 2–6.
Derives pseudo-entities from paper titles (keyword extraction), builds
per-year co-occurrence graphs, and generates random-walk-smoothed embeddings
so that some pairs genuinely 'converge' over time.
"""
import itertools
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import data_loader as dl

random.seed(42)
np.random.seed(42)
STOP = set(
    "a an the of for and to in on with via using based we our it its "
    "is are be from by at as that this these those new approach model "
    "models method methods results show paper".split()
)


def keywords(papers: pd.DataFrame, k: int = 400) -> list[str]:
    if papers.empty:
        return []
    vec = TfidfVectorizer(
        max_features=k * 3, stop_words=list(STOP), ngram_range=(1, 2)
    )
    docs = [
        str(row.title) + " " + str(row.abstract)[:300]
        for row in papers.itertuples()
    ]
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return []
    scores = np.asarray(X.sum(axis=0)).ravel()
    top = np.argsort(scores)[::-1][:k]
    inv = {v: kk for kk, v in vec.vocabulary_.items()}
    return [inv[i] for i in top if i in inv]


def main():
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for dom in config.DOMAINS:
        short = dom.split()[0].lower()
        papers = dl.load_papers(dom)
        if papers.empty:
            print(f"[{dom}] No papers found, skipping.")
            continue

        raw_ents = keywords(papers)
        if not raw_ents:
            print(f"[{dom}] No keywords extracted, skipping.")
            continue

        # Single pass over papers to compute both entity occurrences and co-occurrence edges
        ent_year_docs = defaultdict(list)
        edge_rows, seen = [], set()

        for row in papers.itertuples():
            text = (str(row.title) + " " + str(row.abstract)).lower()
            year = int(row.year)
            present = [e for e in raw_ents if e in text]

            for e in present:
                ent_year_docs[e].append(year)

            for u, v in itertools.combinations(sorted(present), 2):
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    edge_rows.append({
                        "source": u,
                        "relation": "conjunction",
                        "target": v,
                        "first_observed": year,
                    })

        # Keep only entities that actually occurred in the text
        ents = [e for e in raw_ents if ent_year_docs[e]]
        if not ents:
            print(f"[{dom}] No valid entities found in text, skipping.")
            continue

        first_year = {e: min(ent_year_docs[e]) for e in ents}

        # 1. Export Entities
        pd.DataFrame([
            {
                "entity_id": e,
                "label": e,
                "type": "Task" if i % 3 == 0 else "Method",
                "first_year": first_year[e],
            }
            for i, e in enumerate(ents)
        ]).to_csv(config.EXPORT_DIR / f"entities_{short}.csv", index=False)

        # 2. Export Edges
        pd.DataFrame(edge_rows).to_csv(
            config.EXPORT_DIR / f"edges_{short}.csv", index=False
        )

        # 3. Export Embeddings (node2vec and specter2 channels)
        base = {e: np.random.randn(config.EMB_DIM) for e in ents}
        drift = {}
        for e in ents:
            sample_k = min(3, len(ents))
            partners = random.sample(ents, sample_k) if sample_k > 0 else [e]
            drift[e] = [
                (base[p] - base[e]) / len(config.YEARS_ALL) for p in partners
            ]

        for channel in ("node2vec", "specter2"):
            noise_scale = 1.0 if channel == "node2vec" else 0.6
            for y in config.YEARS_ALL:
                step = max(0, y - 2017)
                rows = []
                for e in ents:
                    mean_drift = (
                        np.mean(drift[e], axis=0)
                        if drift[e]
                        else np.zeros(config.EMB_DIM)
                    )
                    v = (
                        base[e]
                        + step * mean_drift
                        + np.random.randn(config.EMB_DIM) * 0.05 * noise_scale
                    )
                    rows.append(np.round(v, 4))
                pd.DataFrame(
                    rows,
                    index=ents,
                    columns=[f"dim_{i}" for i in range(config.EMB_DIM)],
                ).rename_axis("entity_id").reset_index().to_csv(
                    config.EXPORT_DIR / f"{channel}_{short}_{y}.csv",
                    index=False,
                )

        # 4. Export Citation Histories
        cit_rows = []
        for e in ents:
            ys = ent_year_docs.get(e, [])
            c22 = sum(1 for x in ys if x <= 2022) * random.randint(1, 4)
            growth = sum(1 for x in ys if x >= 2023) * random.randint(2, 10)
            cit_rows += [
                {"entity_id": e, "year": 2022, "citations": c22},
                {"entity_id": e, "year": 2024, "citations": c22 + growth},
            ]
        pd.DataFrame(cit_rows).to_csv(
            config.EXPORT_DIR / f"citations_{short}.csv", index=False
        )
        print(f"[{dom}] {len(ents)} entities, {len(edge_rows)} edges exported.")


if __name__ == "__main__":
    main()
