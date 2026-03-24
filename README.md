# MindMap AI

Generate interactive mindmaps from multiple documents using RAG, FAISS, and agentic AI/LLM clustering.

## Features
- Upload multiple `.txt` documents
- All documents are chunked, embedded, and stored in a FAISS vector store
- Automatic clustering of topics using KMeans (user-selectable number of clusters)
- Mindmap nodes are labeled with extracted topic keywords
- Expand any topic node to generate a detailed mindmap for that topic using AI

## Usage
1. Install requirements:
       ```bash
       pip install -r requirements.txt
       ```
2. Run the app:
       ```bash
       streamlit run app.py
       ```
3. Upload your `.txt` files and generate the mindmap.

## Project Structure
- `app.py` — Streamlit UI and main logic
- `components/` — Modular code for vector storage, agent, visualization, etc.
- `data/` — Uploaded documents (gitignored)
- `vectorstorage/` — FAISS index and texts (gitignored)

## Notes
- Only `.txt` files are supported for upload.
- The number of clusters can be adjusted in the UI.
- Mindmap expansion uses only the relevant cluster's documents.
# MindmapAI


                ┌──────────────┐
                │  Documents   │ (PDF, txt, notes)
                └──────┬───────┘
                       ↓
              ┌────────────────┐
              │ Chunking       │
              │ (semantic)     │
              └──────┬─────────┘
                     ↓
        ┌─────────────────────────┐
        │ Embedding (free model)  │
        │ (sentence-transformers) │
        └──────┬──────────────────┘
               ↓
        ┌──────────────────┐
        │ Vector DB        │
        │ (FAISS)          │
        └──────┬───────────┘
               ↓
   ┌──────────────────────────────┐
   │ Query / Topic Input          │
   └───────────┬──────────────────┘
               ↓
        ┌──────────────────────────┐
        │ Retrieve Top-K Chunks    │
        └───────────┬──────────────┘
                    ↓
        ┌────────────────────────────┐
        │ LLM: Extract Entities +    │
        │ Relationships              │
        └───────────┬────────────────┘
                    ↓
        ┌────────────────────────────┐
        │ Build Graph (NetworkX)     │
        └───────────┬────────────────┘
                    ↓
        ┌────────────────────────────┐
        │ Visualize Mind Map         │
        │ (PyVis / D3.js)            │
        └────────────────────────────┘