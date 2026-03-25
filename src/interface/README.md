# Interface Module

This folder contains the visualization layer for the MindMap AI project.

Its responsibility is to take structured mindmap data and render it inside Streamlit.

> Not in scope:
> — file ingestion
> — text extraction
> — embeddings
> — FAISS indexing
> — agent reasoning

---

## Folder structure

```text
src/interface/
├── __init__.py
├── visualize.py
└── example.py
```

## Purpose

The interface module is responsible for:

- receiving graph data in a structured format
- converting it into a NetworkX graph
- rendering the graph in Streamlit
- supporting a standalone example page for testing

This separation keeps visualization logic independent from ingestion and retrieval logic.

## Expected input format

```json
{
  "nodes": [
    "Machine Learning",
    "Supervised Learning",
    "Regression"
  ],
  "edges": [
    {"source": "Machine Learning", "target": "Supervised Learning", "label": "contains"},
    {"source": "Supervised Learning", "target": "Regression", "label": "type"}
  ]
}
```

- `nodes` is a list of strings.
- `edges` is a list of objects with `source`, `target`, and optional `label`.

## Files

### `visualize.py`

Contains the main visualization logic.

- `get_sample_graph_data()` – returns example graph data for testing.
- `build_graph_from_nodes_edges(data)` – converts structured node/edge data into a NetworkX graph.
- `show_graph(G, prefer_pyvis=True)` – renders the graph using PyVis, with a matplotlib fallback.
- `render_mindmap_ui(data=None, show_debug=False, prefer_pyvis=True)` – main entry point for Streamlit rendering.

### `example.py`

Standalone Streamlit test page for validating that the interface layer works independently.

Useful for debugging visualization issues without involving file uploads, embeddings, vectorstores, or agent logic.

## How to run the standalone example

From the project root:

```bash
cd src/interface
streamlit run example.py
```

If code imports run from the project root in `app.py`:

```python
from src.interface.visualize import render_mindmap_ui
```

## Renderer behavior

The interface supports 2 rendering approaches:

1. **PyVis** (interactive graph visualization)
   - Pros: interactive, visually appealing, good for exploration.
   - Cons: may not render reliably in some Streamlit environments.

2. **Matplotlib fallback** (used if PyVis fails or is disabled)
   - Pros: more stable, good for debugging, ensures display even when interactive fails.

## Debugging workflow

1. run `example.py`
2. confirm sample graph data loads
3. test with PyVis enabled
4. if PyVis fails, switch to matplotlib fallback
5. once interface is validated, debug in `app.py`

This helps isolate whether an issue is:

- visualization layer
- renderer
- upstream pipeline logic

## Scope boundaries

The interface module should not:

- read raw files (.pdf, .pptx, .docx, .txt)
- perform chunking
- create embeddings
- query FAISS
- run the agent
- generate mindmaps from raw text

This module only renders already-structured graph data.

## Typical pipeline integration

1. uploaded file
2. text extraction
3. chunking / retrieval / agent processing
4. structured mindmap output
5. interface rendering

Example usage in `app.py`:

```python
mindmap_data = {
  "nodes": ["Topic A", "Subtopic B"],
  "edges": [
    {"source": "Topic A", "target": "Subtopic B", "label": "contains"}
  ]
}

render_mindmap_ui(mindmap_data, show_debug=True, prefer_pyvis=False)
```

## Dependencies

- streamlit
- networkx
- pyvis
- matplotlib

Install:

```bash
pip install streamlit networkx pyvis matplotlib
```

## Summary

This module keeps visualization isolated, testable, and reusable. If `example.py` works but the full app does not, the bug likely lies in:

- `app.py`
- ingestion logic
- agent output formatting
- graph data construction