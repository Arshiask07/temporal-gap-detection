# Component 4 — Temporal Knowledge Graph Builder

**Status:** Planning  
**Dependency:** Component 2 outputs (verified on disk)  
**External tool:** Neo4j graph database (server not yet installed on this machine)  
**Python driver:** `neo4j` 5.28.5.0 installed in Anaconda Python 3.12

---

## 1. Purpose

Build a Neo4j property graph from Component 2's extracted entities and relations, producing a temporal knowledge graph where:

- One `:Entity` node per canonical_id
- One relationship per typed relation, with the actual `relation_type` as the edge label
- `year` stored as an edge property (not a separate graph per year)
- Domain tracked as a node/edge property to prevent cross-domain bridging

This graph feeds Component 5's gap detection (node2vec + SPECTER2 + fusion) and Component 6's citation velocity re-ranking.

---

## 2. Verified input data

All counts below are **read from disk on 2026-09-14**, not from memory.

### 2.1 NLP entities

**File:** `component2_entity_relation_extraction/output/NLP_entities.json`

| Field | Value |
|---|---|
| Total entity records | 28,463 |
| Unique canonical_ids | **12,867** |
| Entity types | Method: 10,473 / Other: 8,997 / Task: 4,855 / Material: 3,028 / Metric: 1,110 |
| Year range | 2018–2024 (see distribution below) |

Year distribution:

```
2018: 3662
2019: 3780
2020: 3956
2021: 4016
2022: 4200
2023: 4332
2024: 4517
```

Schema (first record):

```json
{
  "entity_id": "e_N18-2.39_0",
  "canonical_id": "canon_Method_01191",
  "surface_form": "Word Analogy Testing",
  "normalized_form": "word analogy testing",
  "type": "Method",
  "char_start": 4,
  "char_end": 24,
  "confidence": 0.773,
  "paper_id": "N18-2.39",
  "year": 2018,
  "domain": "NLP"
}
```

### 2.2 NLP relations

**File:** `component2_entity_relation_extraction/output/NLP_relations.json`

| Field | Value |
|---|---|
| Total relation records | 22,690 |
| Unique relation_types | USED_FOR: 9,548 / ENTITY_ASSOCIATED_WITH_ENTITY: 7,738 / METHOD_EVALUATED_BY: 2,992 / METHOD_APPLIED_TO: 2,412 |

Schema (first record):

```json
{
  "relation_id": "r_N18-2.39_0",
  "source_entity_id": "e_N18-2.39_2",
  "target_entity_id": "e_N18-2.39_1",
  "relation_type": "METHOD_APPLIED_TO",
  "scierc_relation_type": "Used-for",
  "confidence": 0.6107,
  "paper_id": "N18-2.39",
  "year": 2018,
  "domain": "NLP"
}
```

### 2.3 NLP entity normalization map

**File:** `component2_entity_relation_extraction/output/NLP_entity_normalization_map.json`

- 15,084 entries
- Maps surface forms → canonical_ids (e.g. `"pre-trained language models" → "canon_Method_00038"`)

### 2.4 COVID entities

**File:** `component2_entity_relation_extraction/output/COVID_entities.json`

| Field | Value |
|---|---|
| Total entity records | 11,918 |
| Unique canonical_ids | **5,500** |
| Entity types | Other: 4,273 / Method: 3,602 / Material: 2,463 / Task: 1,319 / Metric: 261 |
| Year range | 2019–2024 |

### 2.5 COVID relations

**File:** `component2_entity_relation_extraction/output/COVID_relations.json`

| Field | Value |
|---|---|
| Total relation records | 8,081 |
| Unique relation_types | ENTITY_ASSOCIATED_WITH_ENTITY: 4,400 / USED_FOR: 2,560 / METHOD_APPLIED_TO: 801 / METHOD_EVALUATED_BY: 320 |

### 2.6 COVID entity normalization map

**File:** `component2_entity_relation_extraction/output/COVID_entity_normalization_map.json`

- 6,117 entries

### 2.7 Total graph size (expected)

| Metric | NLP | COVID | Total |
|---|---|---|---|
| Canonical entity nodes | 12,867 | 5,500 | **18,367** |
| Typed relation edges | 22,690 | 8,081 | **30,771** |

These are the validation targets after loading. The plan note from earlier conversation said "node count ≈ 12,867 (NLP) + 5,500 (COVID)" and "edge count ≈ 22,690 (NLP) + 8,081 (COVID)" — confirmed exactly on disk.

---

## 3. Architecture decision: single graph with domain isolation

**Decision:** One Neo4j database with a `domain` property on both nodes and edges, NOT two separate databases.

**Rationale:**

- The guardrail in `normalize_entities.py` already prevents cross-domain canonical ID collisions (NLP and COVID canonical_ids are in separate name spaces by construction — confirm this by checking that no canonical_id appears in both NLP and COVID entity files).
- A single database is simpler to back up, query, and maintain.
- Cross-domain bridging is prevented at query time by filtering on `domain` property, matching the design intent in the locked plan.

**CRITICAL FINDING (verified on disk 2026-09-14):** NLP and COVID canonical_ids overlap by **5,099 IDs**. This means a single graph with only `canonical_id` as the uniqueness key would merge NLP and COVID entities that share the same canonical_id. Two approaches:

**Option A (chosen): Composite uniqueness — `canonical_id` + `domain`**
```
CREATE CONSTRAINT entity_canonical_domain IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.canonical_id, e.domain) IS UNIQUE
```
This allows the same canonical_id to exist once per domain. Cross-domain bridging is prevented because queries always filter or match on both `canonical_id` AND `domain`.

**Option B:** Two separate Neo4j databases (one per domain). Simpler isolation, but two connections to manage.

**Decision:** Option A. One database, composite key, domain on every node and edge. The loader asserts that `entity_id` values (the per-mention IDs like `"e_N18-2.39_0"`) do not overlap between NLP and COVID (verified: 0 overlap), so the relations remapping is safe — each `entity_id` maps to exactly one `(canonical_id, domain)` pair.

**Canonical_id multiplicity (normal):** Canonical_ids are NOT unique within a domain — they appear multiple times because the same canonical concept is mentioned in many papers. This is expected and correct:
- NLP: 12,867 unique canonical_ids across 28,463 entity records (avg 2.2 mentions/canonical)
- COVID: 5,500 unique canonical_ids across 11,918 entity records (avg 2.2 mentions/canonical)

The graph node is per-canonical-id-per-domain (18,367 nodes total), not per-entity-record.

---

## 4. Node model

```
(:Entity {
  canonical_id: "canon_Method_01191",   // unique WITH domain (composite key)
  type: "Method",                         // Entity/Method/Task/Material/Metric
  surface_form: "Word Analogy Testing",  // representative surface form
  domain: "NLP",                          // "NLP" or "COVID" — part of uniqueness key
  first_seen: 2018,                       // earliest year this entity appears
  last_seen: 2024,                        // latest year
  paper_count: 5                          // number of distinct papers
})
```

**Constraints:**

- `CREATE CONSTRAINT entity_canonical_domain IF NOT EXISTS FOR (e:Entity) REQUIRE (e.canonical_id, e.domain) IS UNIQUE`
- This is a COMPOSITE uniqueness constraint — the same canonical_id can exist once per domain.
- Optional index on `:Entity(type)` for type-filtered queries (e.g. "find all Method nodes")

**Surface form selection:** When multiple surface forms map to the same canonical_id (expected — canonical concepts are mentioned many times), pick the most frequent surface_form. Tie-break by first occurrence. The normalization map (`*_entity_normalization_map.json`) gives surface→canonical mappings; the entities JSON gives per-mention surface forms.

---

## 5. Edge model

```
(:Entity)-[:USED_FOR {year: 2018, paper_id: "N18-2.39", domain: "NLP", confidence: 0.6107}]->(:Entity)
```

**Edge labels = actual `relation_type` values from the relations JSON:**

| Relation type | Count (NLP) | Count (COVID) | Direction |
|---|---|---|---|
| USED_FOR | 9,548 | 2,560 | source → target |
| ENTITY_ASSOCIATED_WITH_ENTITY | 7,738 | 4,400 | source → target |
| METHOD_EVALUATED_BY | 2,992 | 320 | source → target |
| METHOD_APPLIED_TO | 2,412 | 801 | source → target |

**Direction:** Keep the directed source→target from the extraction. Do NOT make edges undirected at load time — Component 5's co-occurrence analysis may need direction, and the paper-level relation records are directional.

**Edge properties:**

- `year` — publication year of the paper where this relation was extracted
- `paper_id` — source paper
- `domain` — "NLP" or "COVID"
- `confidence` — model confidence score
- `scierc_relation_type` — optional, the original SciERC label (e.g. "Used-for")

---

## 6. Batch loading strategy

Bulk loading via `UNWIND` with transactions of ~1,000 rows per commit, matching the locked plan.

### 6.1 Entity load

```cypher
UNWIND $batch AS row
MERGE (e:Entity {canonical_id: row.canonical_id, domain: row.domain})
SET e.type = row.type,
    e.first_seen = coalesce(e.first_seen, row.year),
    e.last_seen = row.year,
    e.paper_count = coalesce(e.paper_count, 0) + 1
```

The `SET e.domain = row.domain` is technically redundant since `domain` is in the MERGE key, but it ensures the property is set even on newly created nodes. The `paper_count` increment via `coalesce + 1` works correctly because each canonical_id appears in multiple entity records (one per paper mention), and each record carries its `paper_id`. Pre-computing distinct paper counts in Python is more accurate — count distinct `(canonical_id, domain, paper_id)` tuples per canonical_id and pass as `paper_count` in the batch row.

### 6.2 Relation load

```cypher
UNWIND $batch AS row
MATCH (src:Entity {canonical_id: row.source_canonical_id})
MATCH (tgt:Entity {canonical_id: row.target_canonical_id})
MERGE (src)-[r:row.relation_type]->(tgt)
SET r.year = row.year,
    r.paper_id = row.paper_id,
    r.domain = row.domain,
    r.confidence = row.confidence
```

**Critical detail:** Entity IDs in the relations JSON are `entity_id` (e.g. `"e_N18-2.39_2"`), NOT `canonical_id`. The loader must resolve `entity_id → canonical_id` using the entities JSON before building the edge batch. This is the same remapping logic Component 3's `_load_component2()` already does (lines 97–101 in `component3_retrospective_validation.py`).

**Cypher label from property:** Neo4j doesn't allow dynamic relationship types in `MERGE` via parameter in older versions, but Neo4j 5.x supports `apoc.create.relationship` or the newer `MERGE (a)-[r:TYPE]->(b)` where TYPE is constructed. The safe approach: use `apoc.merge.relationship` if APOC is installed, or construct the Cypher string per relation type in Python and submit per-type batches. Given there are only 4 distinct relation types per domain, per-type batching is clean.

Actually — in Neo4j 5.x, you CAN do:

```cypher
MERGE (a)-[:USED_FOR {year: 2018}]->(b)
```

But you can't parameterize the label name in `MERGE`. So build the query string with the label hardcoded per batch, where each batch is homogeneous in relation_type. This is fine — 4 relation types × 2 domains = 8 batches for edges.

---

## 7. Implementation plan

### Step 1: Install and start Neo4j server

**Status:** Blocked — `brew install neo4j` was interrupted on this machine.

**What's needed:**

```bash
brew install neo4j        # installs neo4j + cypher-shell + openjdk@21 dependency
brew services start neo4j # starts on localhost:7687 (bolt) / 7474 (http)
```

**Verification after install:**

```bash
neo4j status
cypher-shell -u neo4j -p neo4j "RETURN 1;"
```

Default credentials: `neo4j`/`neo4j` (forces password change on first connect).

**If Homebrew install continues to fail on this machine:** alternatives are (a) download the Neo4j community tarball directly from neo4j.com and run `bin/neo4j start`, or (b) use Docker `docker run -p 7687:7687 neo4j:5` if Docker is available. Check which is viable.

**Note:** The Python `neo4j` driver (5.28.5.0) is already installed in Anaconda Python. Only the server is missing.

### Step 2: Write the loader script

**File:** `component4/build_graph.py` (new file)

**Responsibilities:**

1. Process each domain's entities JSON separately. For each domain, pre-compute per-canonical_id aggregates: type (most common), representative surface_form (most frequent), first_seen (min year), last_seen (max year), paper_count (distinct paper_ids). Domain is known from the file, not from the entity record's `domain` field (which is always "NLP" or "COVID" anyway).

2. Connect to Neo4j via the Python driver.

3. Create the composite uniqueness constraint on `:Entity(canonical_id, domain)`.

4. Batch-load NLP entities (12,867 rows), then COVID entities (5,500 rows) via UNWIND, ~1,000 rows per transaction.

5. Build entity_id → (canonical_id, domain) lookup from both entities JSONs.

6. For each domain's relations JSON, remap source/target entity_ids to canonical_ids using the lookup. All relations are expected to resolve (verified: 0 missing).

7. Group resolved relations by (domain, relation_type). For each group, batch-load edges via UNWIND + MERGE with the relationship type hardcoded in the Cypher string. ~1,000 rows per transaction.

8. Print validation counts and write `graph_summary.json`.

**Key implementation detail — entity_id to canonical_id remapping:**

The relations JSON uses `source_entity_id` / `target_entity_id` which are the per-mention IDs (e.g. `"e_N18-2.39_2"`), not canonical_ids. Build a dict from the entities JSON: `{entity["entity_id"]: (entity["canonical_id"], entity["domain"])}`. Verified on disk: 0 relations have missing source or target entities, and 0 relations reference entities from the wrong domain. So every relation resolves cleanly to a `(source_canonical_id, source_domain)` / `(target_canonical_id, target_domain)` pair. The loader skips any relation that fails to resolve (defensive coding), but none are expected to fail.

**Key implementation detail — representative surface_form:**

For each canonical_id, pick the surface_form that appears most often across all entity records with that canonical_id. Tie-break by first occurrence. This gives a human-readable label for the node.

### Step 3: Validation queries

After loading, run these to confirm:

```cypher
// Total nodes
MATCH (e:Entity) RETURN count(e) AS node_count;

// Nodes per domain
MATCH (e:Entity) RETURN e.domain, count(e) AS count;

// Total edges
MATCH ()-[r]->() RETURN count(r) AS edge_count;

// Edges per type and domain
MATCH ()-[r]->() RETURN r.domain, type(r), count(r) AS count;

// Year coverage
MATCH ()-[r]->() RETURN r.year, count(r) AS count ORDER BY r.year;

// Verify no cross-domain edges
MATCH (a:Entity {domain: "NLP"})-[r]->(b:Entity {domain: "COVID"})
RETURN count(r) AS cross_domain_edges;  // should be 0
```

The loader should assert:

- NLP nodes == 12,867
- COVID nodes == 5,500
- NLP edges == 22,690
- COVID edges == 8,081

If any count is off, print the delta and the actual counts so the discrepancy is visible.

### Step 4: Export a graph summary

Write `component4/graph_summary.json` with:

```json
{
  "loaded_at": "2026-09-14T...",
  "neo4j_version": "...",
  "domains": {
    "NLP": {"nodes": 12867, "edges": 22690, "relation_types": [...]},
    "COVID": {"nodes": 5500, "edges": 8081, "relation_types": [...]}
  },
  "total_nodes": 18367,
  "total_edges": 30771,
  "cross_domain_edges": 0,
  "years_covered": [2018, 2019, 2020, 2021, 2022, 2023, 2024]
}
```

This is the checkpoint file for Component 4 — future steps (Component 5/6/7) read this to know the graph is loaded and what it contains.

---

## 8. What's NOT in scope for Component 4

- **No temporal snapshots.** The locked plan says "year as an edge property (not physically separate graphs per year)." Component 4 loads one graph with year on edges. Component 5 builds per-year subgraphs from this by filtering on `year <= t`.
- **No citation data.** Citation velocity (Component 6) uses Semantic Scholar API data collected in Component 1. Component 4 does not ingest citations.
- **No embeddings.** SPECTER2 and node2vec embeddings are Component 5's job. Component 4 only builds the structural graph.
- **No cross-domain bridging.** Even though it's one database, domain filtering is enforced at query time. The loader asserts zero cross-domain edges as a validation check.

---

## 9. Quality gates before moving to Component 5

- [ ] Neo4j server installed and running on this machine (or a reachable instance)
- [ ] `build_graph.py` runs end-to-end without errors
- [ ] NLP node count == 12,867 (±0)
- [ ] COVID node count == 5,500 (±0)
- [ ] NLP edge count == 22,690 (±0)
- [ ] COVID edge count == 8,081 (±0)
- [ ] Cross-domain edges == 0
- [ ] `graph_summary.json` written
- [ ] Can run a sample Cypher query from Python and get back results

---

## 10. Files

```
component4/
├── build_graph.py              # loader script (to write)
├── graph_summary.json          # output: load validation (to write)
└── README.md                   # this file
```

Input (read-only, from Component 2):

```
component2_entity_relation_extraction/output/
├── NLP_entities.json
├── NLP_relations.json
├── NLP_entity_normalization_map.json
├── COVID_entities.json
├── COVID_relations.json
└── COVID_entity_normalization_map.json
```

---

## 11. Open questions

1. **Neo4j install path.** `brew install neo4j` was interrupted on this machine. Need to complete it, or fall back to Docker/manual tarball. This is the only external dependency blocking Component 4. The Python `neo4j` driver (5.28.5.0) is already installed — only the server is missing.

2. **APOC plugin.** The edge loader uses per-type batching with hardcoded relationship labels in Cypher strings, which does NOT require APOC. No APOC dependency.

3. **Canonical_id namespace collision — RESOLVED.** NLP and COVID canonical_ids overlap by 5,099 IDs (verified on disk). The composite uniqueness constraint `(canonical_id, domain)` handles this. The entity_id values do not overlap between domains (0 overlap), so relation remapping is clean.

4. **Memory.** 18,367 nodes + 30,771 edges is small for Neo4j. Default JVM heap should handle it fine. No tuning needed unless loading proves slow.

5. **Paper nodes.** Not created. Year filtering is done on edges. If needed later, can be added without restructuring.

6. **What to add to the graph beyond the minimum:** The minimum viable graph is entities + typed relations with year + domain properties. Beyond that, the following are worth adding if they serve Component 5's gap detection:

   - **`surface_form` on nodes** (already in the node model) — needed for human-readable gap reporting in Components 5/6/7.
   - **`type` on nodes** (already in the node model) — Component 5's co-occurrence-based gap candidates should respect entity types (e.g. Method-Task pairs are more meaningful than Other-Other pairs).
   - **`paper_id` on edges** (already in the edge model) — enables traceability: "which paper established this relation?"
   - **`confidence` on edges** (already in the edge model) — enables confidence-thresholded queries for higher-precision subgraph extraction.
   - **Per-year edge indexing** — create indexes on `:Entity(first_seen)`, `:Entity(last_seen)`, and edge `year` property to speed up Component 5's `year <= t` subgraph extractions.
