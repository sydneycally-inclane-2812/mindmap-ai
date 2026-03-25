* **DuckDB** for text, metadata, evidence, entity registry, and mappings
* **FAISS** for semantic retrieval
* **Neo4j** for graph traversal and subgraph extraction
* **LLM** for mention extraction, relation extraction, and final mind map summary

Core flow:

```text
query -> FAISS semantic seeds -> map to graph nodes -> graph expansion -> subgraph pruning -> evidence hydration -> LLM answer / mind map
```

---

# 1. Core storage design

## DuckDB responsibilities

DuckDB stores:

* documents
* chunks
* evidence units
* chunk summaries
* entity registry
* aliases
* extracted mentions
* relation evidence
* FAISS ID mappings

## FAISS responsibilities

FAISS indexes:

* chunk text embeddings
* chunk summary embeddings
* entity description embeddings

## Neo4j responsibilities

Neo4j stores:

* canonical graph nodes
* semantic relationships
* provenance links
* graph traversal paths for mind map generation

---

# 2. Global entity registry format

Use this simplified Python-side registry:

```python
entities = {
    "gradient_descent": {
        "aliases": ["gradient descent", "gd"]
    },
    "machine_learning": {
        "aliases": ["machine learning", "ml"]
    }
}
```

Interpretation:

* key = canonical entity name
* value.aliases = alternate surface forms

For now:

* no separate entity IDs
* no entity types required yet
* canonical names use `snake_case`

This is acceptable for MVP.

## Canonical naming rule

All canonical entities should be stored in a normalized form:

* lowercase
* snake_case
* no punctuation unless necessary

Examples:

* `gradient_descent`
* `machine_learning`
* `support_vector_machine`

---

# 3. Graph schema

## Node types

* `Document`
* `Chunk`
* `Topic`
* `Concept`
* `Definition`
* `Formula`
* `Example`
* `Process`
* `Property`
* `Date`

## Relationship types

* `CONTAINS`
* `PART_OF`
* `DEFINES`
* `HAS_PROPERTY`
* `HAS_FORMULA`
* `HAS_EXAMPLE`
* `HAS_APPLICATION`
* `PREREQUISITE_OF`
* `DEPENDS_ON`
* `USES`
* `HAS_STEP`
* `NEXT_STEP`
* `SIMILAR_TO`
* `CONTRASTS_WITH`
* `MENTIONED_IN`
* `SUPPORTED_BY`

For MVP, not every node type must be used immediately. You can start mainly with:

* `Document`
* `Chunk`
* `Topic`
* `Concept`

and add the rest incrementally.

---

# 4. DuckDB tables

## `documents`

* `document_id`
* `title`
* `source_path`
* `doc_type`
* `checksum`

## `chunks`

* `chunk_id`
* `document_id`
* `chunk_text`
* `chunk_summary`
* `section_title`
* `chunk_order`
* `token_count`

## `evidence_units`

Start with paragraph-level evidence.

* `evidence_unit_id`
* `chunk_id`
* `unit_type`
* `unit_text`
* `unit_order`
* `start_char`
* `end_char`

## `entity_registry`

Stores canonical names.

* `canonical_name`
* `aliases_json`

Example row:

* `canonical_name = "gradient_descent"`
* `aliases_json = ["gradient descent", "gd"]`

## `entity_mentions`

* `mention_id`
* `evidence_unit_id`
* `surface_form`
* `canonical_name`
* `decision`
* `confidence`

Where `decision` is one of:

* `match_existing`
* `create_new`
* `uncertain`

## `relations`

* `relation_id`
* `source_canonical_name`
* `relation_type`
* `target_canonical_name`
* `confidence`
* `is_derived`

## `relation_evidence`

* `relation_id`
* `evidence_unit_id`
* `support_score`

## `faiss_id_map`

* `faiss_id`
* `index_name`
* `item_type`
* `item_id`

---

# 5. Ingestion pipeline

## Step 1: parse document into text

Input sources:

* PDF
* DOCX
* PPTX
* TXT

Output:

* raw text blocks

## Step 2: create retrieval chunks

Chunk size:

* about 250–500 tokens
* section-aware if possible

Do not make chunks sentence-sized.

Each chunk should keep:

* order
* section title
* document link

## Step 3: split chunks into evidence units

Inside each chunk, split into paragraph-level units.

Example:

* chunk `c12`

  * `p1`
  * `p2`
  * `p3`

Evidence units are for:

* grounding
* citations
* relation support

---

## Step 4: generate chunk summaries

For each chunk, generate a short summary.

Store in DuckDB and embed into FAISS.

---

## Step 5: mention extraction using the global registry

For each chunk, call the LLM with:

* chunk text
* evidence labels like `p1`, `p2`, `p3`
* full global entity registry

Prompt behavior:

* if a mention matches an existing canonical entity or alias, return that canonical name
* otherwise create a new canonical name in snake_case
* attach evidence unit references

Expected output shape:

```json
{
  "mentions": [
    {
      "surface_form": "GD",
      "canonical_name": "gradient_descent",
      "decision": "match_existing",
      "evidence_units": ["p2"]
    },
    {
      "surface_form": "differentiable objective function",
      "canonical_name": "differentiable_objective_function",
      "decision": "create_new",
      "evidence_units": ["p3"]
    }
  ]
}
```

---

## Step 6: normalize and resolve mentions

Even with the global registry prompt, do a code-side check.

### Normalize function

Use a deterministic normalizer:

```python
def normalize_entity(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = "_".join(text.split())
    return text
```

Examples:

* `"Gradient Descent"` -> `gradient_descent`
* `"gradient descent"` -> `gradient_descent`
* `"GD"` -> `gd`

### Resolution logic

For each extracted mention:

1. normalize the returned canonical name
2. check whether it exists as a canonical name in `entity_registry`
3. if not, check whether it matches an alias of an existing canonical entity
4. if yes, replace with the existing canonical entity
5. if no, create a new canonical entity row
6. if surface form is useful and not already present, add it as an alias

Example:

* mention: `"GD"`
* model returns: `gradient_descent`
* registry already contains `gradient_descent`
* store mention as linked to `gradient_descent`

Another example:

* mention: `"Gradient descent"`
* model returns: `gradient_descent`
* alias `"gradient descent"` gets stored if missing

This keeps the graph consistent.

---

## Step 7: relation extraction

For each chunk, call the LLM with:

* original chunk text
* evidence unit labels
* resolved canonical entity names found in that chunk

Ask it to extract only directly supported relations.

Example output:

```json
{
  "relations": [
    {
      "source": "gradient_descent",
      "relation": "HAS_FORMULA",
      "target": "gradient_descent_update_rule",
      "evidence_units": ["p2"],
      "confidence": 0.95
    },
    {
      "source": "gradient_descent",
      "relation": "DEPENDS_ON",
      "target": "differentiable_objective_function",
      "evidence_units": ["p3"],
      "confidence": 0.87
    }
  ]
}
```

Important:

* relation extraction must use the original chunk text
* not only the chunk summary

---

## Step 8: write to DuckDB

Store:

* chunk text
* chunk summary
* evidence units
* entity mentions
* canonical entity registry
* aliases
* relations
* relation evidence
* FAISS index mapping

---

## Step 9: write to Neo4j

Create:

* document nodes
* chunk nodes
* concept/topic/etc nodes
* semantic edges
* provenance edges

Typical writes:

* `Document -[:CONTAINS]-> Chunk`
* `Chunk -[:MENTIONED_IN]-> Concept` or equivalently `Concept -[:MENTIONED_IN]-> Chunk`
* `Concept -[:HAS_FORMULA]-> Formula`
* `Concept -[:DEPENDS_ON]-> Concept`
* `Concept -[:SUPPORTED_BY]-> Chunk` if you want direct evidence linkage there too

For MVP, keep provenance simple and rely on DuckDB for detailed evidence text.

---

## Step 10: embed into FAISS

Create embeddings for:

* chunk text
* chunk summary
* canonical entity descriptions

At first, if you do not yet have full descriptions, entity description can just be:

* canonical name
* aliases joined into a short string

Example:

```text
gradient_descent | aliases: gradient descent, gd
```

Store FAISS ID mappings in DuckDB.

---

# 6. Search pipeline

## Step 1: embed user query

Convert the user query into an embedding.

## Step 2: broad FAISS retrieval

Search:

* chunk index
* chunk summary index
* entity description index

Return top-k from each.

For MVP, broad retrieval is fine.

---

## Step 3: map FAISS hits to graph seeds

Convert hits into graph anchors.

Examples:

* chunk hit -> chunk node + entities mentioned in that chunk
* entity description hit -> concept/topic node
* chunk summary hit -> original chunk + linked entities

Then rerank with:

* FAISS similarity
* exact text overlap with query
* canonical name / alias overlap

---

## Step 4: graph expansion

Take top seeds and expand through Neo4j.

Use hard limits:

* max depth = 1 or 2
* max neighbors per node
* confidence threshold

Do not let expansion grow unchecked.

---

## Step 5: subgraph pruning

Prune by:

* relation confidence
* number of supporting evidence units
* graph distance from original seed
* query relevance

This is necessary so the mind map stays readable.

---

## Step 6: evidence hydration

For important nodes and edges in the subgraph:

* fetch supporting evidence units from DuckDB
* attach the paragraph text

This gives you:

* graph structure from Neo4j
* actual source text from DuckDB

That combination is what makes the output grounded.

---

## Step 7: build mind-map-ready structure

Before sending to the final LLM, compress the subgraph into something cleaner.

Example structure:

* center concept/topic
* first-level branches grouped by relation type or subtopic
* second-level nodes as related concepts
* evidence snippets attached

This is better than sending the raw graph.

---

## Step 8: final LLM generation

Send:

* compact subgraph
* evidence snippets
* user query

Generate:

* answer
* summary
* mind map text/JSON

---

# 7. Prompt rules

## Mention extraction prompt

Input:

* chunk text
* evidence unit labels
* global entity registry in the simplified dict form

Instruction:

* return canonical entity names exactly as they exist in the registry if matched
* if new, create a new canonical name in snake_case
* attach evidence unit labels
* do not invent alternate spellings for existing canonical names

## Relation extraction prompt

Input:

* chunk text
* evidence labels
* resolved canonical names in this chunk

Instruction:

* extract only relations directly supported by the text
* use the provided canonical names exactly
* attach evidence unit labels
* do not invent unsupported relations

---

# 8. Minimal entity registry policy

Your Python-side registry stays simple:

```python
entities = {
    "gradient_descent": {
        "aliases": ["gradient descent", "gd"]
    },
    "machine_learning": {
        "aliases": ["machine learning", "ml"]
    }
}
```

## Update rule during ingestion

When a new entity is created:

* add a new canonical key
* initialize aliases with the observed surface form if useful

Example:

```python
entities["differentiable_objective_function"] = {
    "aliases": ["differentiable objective function"]
}
```

When an existing entity is matched:

* add new alias if it is not already present

---

# 9. MVP constraints

To keep it buildable, start with:

* paragraph-level evidence only
* chunk size 250–500 tokens
* simple normalization
* shallow graph depth
* global registry attached to prompts
* broad FAISS retrieval
* limited node types actually used at first

Start with mostly:

* `Concept`
* `Topic`
* `Document`
* `Chunk`

Then add:

* `Definition`
* `Formula`
* `Example`
* `Process`

once the pipeline is stable.

---

# 10. Final end-to-end plan

## Ingestion

```text
document
-> parse text
-> split into chunks
-> split chunks into paragraph evidence units
-> generate chunk summaries
-> mention extraction using global registry
-> normalize and resolve mentions
-> update entity registry and aliases
-> relation extraction using resolved canonical names
-> store all metadata/text/evidence in DuckDB
-> write graph structure to Neo4j
-> create embeddings and index in FAISS
```

## Search

```text
user query
-> embed query
-> FAISS retrieval over chunks, summaries, entity descriptions
-> map hits to graph seeds
-> rerank
-> expand graph in Neo4j
-> prune subgraph
-> hydrate with evidence from DuckDB
-> build compact mind map structure
-> LLM generates final answer / mind map
```