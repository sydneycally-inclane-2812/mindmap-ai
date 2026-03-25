# Embedding Pipeline README (Current Implementation)

This document reflects what is currently implemented in this folder.

## Scope

The current pipeline does the following:

- splits documents into text chunks using a custom splitter (no LangChain dependency)
- creates embeddings with SentenceTransformers and stores vectors in FAISS
- extracts entities and relationships with a Groq LLM in batches
- stores relation evidence and ingestion artifacts in DuckDB
- writes entity graph and relationships to Neo4j
- exposes query helpers for graph traversal and evidence lookup

Primary modules:

- src/embedding/preprocessing.py
- src/embedding/llm.py
- src/embedding/evidence_store.py
- src/embedding/graph.py
- src/embedding/pipeline.py

---

## Current Data Stores

### DuckDB

Used for evidence and ingestion artifacts through EvidenceDuckDBStore.

Active tables:

- documents
- chunks
- evidence_units
- entity_registry
- relations
- relation_evidence

Current runtime path used by preprocessing and pipeline helpers:

- databases/evidence.duckdb

Note:

- EvidenceDuckDBStore has a default db_path of database/evidence.duckdb, but preprocessing passes databases/evidence.duckdb explicitly.

### FAISS

Used for chunk embedding persistence.

Current behavior:

- embeddings are generated with all-MiniLM-L6-v2
- index and text map are persisted under vector_storage

### Neo4j

Used for entity graph storage and traversal.

Current graph model in code:

- node label: Entity
- relationship type: RELATED_TO
- relationship property type stores semantic relation label (for example relies_on)

---

## Ingestion Flow (As Implemented)

The ingest() function in pipeline.py runs these stages:

1. open Neo4j connection
2. fetch existing Entity node names from Neo4j
3. pass those names as seed entities into preprocessing
4. preprocess documents (chunk, embed, extract, persist evidence)
5. build graph in Neo4j

Important current behavior:

- build_graph() currently clears Neo4j graph before writing new nodes and edges
- existing Neo4j entity names are still used as extraction hints before clear

---

## Chunking (No LangChain)

Chunking is implemented in preprocessing.py using custom logic.

Features:

- overlapping windows based on chunk_size and chunk_overlap
- prefers natural boundaries in this order:
  - paragraph breaks
  - line breaks
  - sentence boundaries
  - whitespace fallback

This replaced langchain_text_splitters.

---

## LLM Extraction

Implemented in llm.py.

### Entity extraction

- runs in batches over all chunks
- can receive seed_entities from Neo4j
- for each batch, only seed entities present in that batch text are injected as hints
- merged and deduplicated across all batches

### Relationship extraction with evidence

- runs in batches over all chunks
- constrained to fixed taxonomy:
  - relies_on
  - requires
  - influences
  - supports
  - contrasts_with
  - similar_to
  - depends_on
  - enables
  - prevents
  - related_to
- optional canonical entity constraints are filtered per batch to keep prompts bounded
- outputs include source, relation, target, evidence, confidence
- deduplicates records and keeps the highest-confidence variant per normalized key

---

## Neo4j Relationship Write Rule

add_relationships() in graph.py enforces one directed edge per pair A -> B at write time.

Cypher behavior:

- if any outgoing edge from A to B already exists, no new edge is created
- otherwise create one RELATED_TO edge with:
  - type = extracted relation label
  - count = 1

This keeps graph density controlled and avoids repeated edge growth for the same directed pair.

---

## Evidence Behavior

EvidenceDuckDBStore.store_ingestion() defaults to reset_existing=True.

On each ingestion run, it clears ingestion-scoped tables before inserting fresh rows:

- relation_evidence
- evidence_units
- relations
- chunks
- documents

This prevents evidence duplication across repeated runs for the same local test workflow.

---

## Public API (from src.embedding)

Exported functions/classes:

- ingest
- query
- query_batch
- get_shortest_path
- get_graph_summary
- get_entity_neighbors
- get_evidence_for_entity
- show_evidence_for_entity
- Neo4jGraphStore

---

## Query Helpers

### query(entity, graph_store)

Returns incoming and outgoing relationships plus total connections.

### query_batch(entities, graph_store)

Runs query() for multiple entities.

### get_shortest_path(source, target, graph_store)

Returns shortest path and hop length when a path exists.

### get_graph_summary(graph_store)

Returns node/edge counts, relation type histogram, average degree, and top entities.

### get_entity_neighbors(entity, graph_store, depth)

Returns neighbors and per-path details up to depth.

Note:

- this query can become expensive on dense/cyclic graphs due to variable-length path expansion.

### get_evidence_for_entity(entity, limit)

Reads DuckDB relation evidence rows for an entity and returns structured results.

---

## Environment Requirements

Required in .env:

- GROQ_API_KEY
- NEO4J_URI
- NEO4J_USERNAME
- NEO4J_PASSWORD

---

## Current Limitations

- build_graph() clears Neo4j each run, so graph is not incremental yet
- get_entity_neighbors() can be slow at higher depth on dense graphs
- FAISS operations show static type-check warnings in editor stubs, but runtime logic is intact
- relation model is intentionally simple (single RELATED_TO edge with a type property)

---

## Recommended Next Improvements

1. Make Neo4j build mode configurable:
   - replace or append
2. Add bounded neighbor traversal options:
   - max_neighbors
   - max_paths_per_neighbor
3. Add ingestion checkpoints for very large corpora
4. Add entity normalization policy (for example snake_case canonicalization) before graph writes
