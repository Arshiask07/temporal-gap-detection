\# Predicting Emerging Research Gaps in Scientific Literature via Dual-Channel Temporal Knowledge Graphs



A time-aware pipeline that builds annual knowledge-graph snapshots from scientific literature and retrospectively validates whether structurally predicted research gaps go on to materialize as future publications.



\## Motivation



Existing literature discovery tools are Connected Papers, ResearchRabbit, Semantic Scholar which give surface citation relationships between papers. They do not indicate which research concepts have never been studied together, or which of these missing connections are becoming urgent. This project addresses that gap by constructing concept-level knowledge graphs from paper abstracts, scoring entity pairs by convergence velocity across annual snapshots, and validating the resulting predictions against literature published after the prediction cutoff.



\## Results



| Metric | NLP | COVID-19 (CS-adjacent) |

|---|---|---|

| Retrospective gap hit rate (cutoff 2021, 75 candidates) | 54.67% | 24.00% |

| Best NER F1 (SciBERT, SciERC test) | 0.5974 | — |

| Typed relation share (SciBERT) | 65.9% | 45.6% |

| Corpus size | 1,400 papers (2018–2024) | 840 papers (2019–2024) |

| Canonical entity nodes | 15,084 | 6,117 |



Full results and methodology: \[`reports/COMPREHENSIVE\_COMPONENT1\_3\_REPORT.md`](reports/COMPREHENSIVE\_COMPONENT1\_3\_REPORT.md)



\## Project status



Components 1 through 3 are complete.



\*\*Implemented\*\*

\- Multi-source temporal corpus collection (ACL Anthology, arXiv, Semantic Scholar, CORD-19), with composite quality scoring and fuzzy deduplication

\- Entity and relation extraction benchmarked across four models — spaCy baseline, SciBERT, RoBERTa, PubMedBERT — each fine-tuned on SciERC

\- Type-constrained entity canonicalization via fuzzy string matching

\- Retrospective gap validation: candidate gaps are predicted from pre-cutoff data and checked against a post-cutoff literature holdout



\*\*Not yet implemented\*\*

\- Temporal knowledge graph construction (Neo4j)

\- Dual-channel gap scoring (Node2Vec structural embeddings fused with SPECTER2 semantic embeddings)

\- Citation-velocity re-ranking

\- Interactive dashboard



\## Repository structure



​```

src/

├── ingestion/          Data collection and sampling (Component 1)

├── extraction/          Entity extraction across four model tiers (Component 2)

├── relation/            Span-pair relation classification (Component 2)

├── canonicalization/    Entity deduplication and canonical ID assignment (Component 2)

├── evaluation/           SciERC F1 scoring scripts

└── validation/           Retrospective gap validation (Component 3)



docs/                    Technical documentation, dataset sources, architecture notes

reports/                 Per-component result reports and summary statistics

reports/validation\_results/   Per-domain, per-cutoff validation outputs

​```

