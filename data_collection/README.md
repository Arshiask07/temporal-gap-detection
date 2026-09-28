# Data Collection Pipeline — NLP + COVID (CS-adjacent)

One-time snapshot collection, per your plan: run each source once, save locally,
then everything downstream reads only from these local files. This is standard
practice — document the collection date in your paper.

## What's already done for you

`scripts/01_collect_acl_anthology.py` was run in this environment and produced
real, verified data (in `data/nlp/acl_anthology_2018.json` through `_2024.json`,
1,891–5,921 papers/year). This works because ACL Anthology publishes its data as
XML in a public GitHub repo, which is reachable from most sandboxed environments.

**Semantic Scholar, arXiv, and CORD-19 are NOT reachable from this sandbox**
(network allowlist blocks them here — you'll see HTTP 403). The scripts for
those three are complete and tested for correctness, but you need to run them
on your own machine, where they'll work normally.

## Setup (on your machine)

```bash
cd data_collection
pip install -r requirements.txt
```

## Run order

```bash
# 1. Already done (included in this package) — re-run only if you want more years/venues
python scripts/01_collect_acl_anthology.py --years 2018 2019 2020 2021 2022 2023 2024

# 2a. Add citation counts to ACL Anthology papers (optional but useful for filtering)
python scripts/02_collect_semantic_scholar.py enrich --year 2023
python scripts/02_collect_semantic_scholar.py enrich --year 2024
# (repeat per year you care about — this is slow, ~1-4s/paper depending on API key)

# 2b. COVID domain: fill the 2023-2024 gap (CORD-19 is frozen at June 2022)
python scripts/02_collect_semantic_scholar.py covid-search --years 2023 2024

# 3. arXiv preprint fallback (optional, for early-year / non-venue coverage)
python scripts/03_collect_arxiv.py --years 2018 2019 2020 2021 2022 2023 2024

# 4. CORD-19: filter local CSV or use automatic online API collection
python scripts/04_filter_cord19.py --start-year 2019 --end-year 2022

# 5. Multi-Source COVID Harvester (PubMed, Europe PMC, arXiv, OpenAlex)
python scripts/05_collect_covid_papers.py --years 2019 2020 2021 2022 2023 2024
```

Get a free Semantic Scholar API key to avoid aggressive rate limiting:
https://www.semanticscholar.org/product/api
```bash
export S2_API_KEY=your_key_here
```

## Output layout

```
data/
  nlp/
    acl_anthology_2018.json ... acl_anthology_2024.json   <- done, verified
    acl_anthology_2023_enriched.json                       <- after step 2a
    arxiv_cscl_2018.json ... arxiv_cscl_2024.json           <- after step 3
  covid/
    cord19_metadata_raw.csv                                <- optional local CSV
    cord19_filtered_2019.json ... cord19_filtered_2022.json <- after step 4
    covid_semantic_scholar_2023.json, _2024.json            <- after step 2b
    covid_harvested_2019.json ... covid_harvested_2024.json <- after step 5
```

## Important: your raw pool is bigger than your target

Your plan targets 150–250 NLP papers/year and 100–200 COVID papers/year. The
ACL Anthology pull alone already returns 1,900–5,900 papers/year (it includes
every accepted paper at ACL/EMNLP/NAACL/Findings/COLING/CoNLL — most of that is
outside your specific subtopics of interest).

**This is intentional at this stage.** Filter down to your target size and
subtopics as a *separate* step (this becomes your Component 2+ in the pipeline
you described) — e.g. keyword/topic filtering, or feeding titles+abstracts to
SciBERT/SciERC for relevance scoring. Don't bake topic filtering into the raw
collection scripts, or you'll need to re-download every time you tweak the
keyword list. Keep the raw snapshot as your immutable base, and filter from it
as many times as you want.

## Reproducibility note for your paper

State something like:
> "NLP corpus data was collected from the ACL Anthology (GitHub XML export,
> commit as of August 2026), Semantic Scholar API, and arXiv API on August 2026.
> COVID corpus data uses the CORD-19 dataset (frozen June 2022) supplemented
> with Semantic Scholar API results for 2023–2024, collected August 2026."

Consider also saving the exact git commit hash of the ACL Anthology repo you
pulled from, for full reproducibility:
```bash
curl -s https://api.github.com/repos/acl-org/acl-anthology/commits/master | grep '"sha"' | head -1
```
