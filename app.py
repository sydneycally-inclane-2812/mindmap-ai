import streamlit as st
import networkx as nx
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pptx import Presentation
from docx import Document
import io
import fitz # PyMuPDF

from components.documents_ingestion import chunk_text
from components.vectorstore import FAISSStore
from components.visualize import show_graph
from components.agent import run_agent

st.title("🧠 MindMap RAG + Agent (FAISS)")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()


def build_store(text):
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)

    store = FAISSStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    return store


def build_graph(nodes, edges):
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        G.add_edge(e["source"], e["target"], label=e["relation"])
    return G
from pypdf import PdfReader
from pptx import Presentation
import io


def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_file(uploaded_file):
    """Extract plain text from .txt, .pdf, or .pptx files."""
    name = uploaded_file.name.lower()
    
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    
    elif name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    
    elif name.endswith(".pptx"):
        prs = Presentation(io.BytesIO(uploaded_file.read()))
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text_frame.text)
        return "\n".join(texts)
    
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(para.text for para in doc.paragraphs if para.text)
    
    return ""  # unsupported type

# After
uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "pptx", "docx"], accept_multiple_files=True)

import os

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text_from_file(uploaded_file)
        # Save as .txt regardless of original format
        base_name = os.path.splitext(uploaded_file.name)[0]
        file_path = os.path.join(data_dir, base_name + ".txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    st.success(f"Uploaded and processed {len(uploaded_files)} files.")

# Ingest all documents in data_dir and build a single vector store
def ingest_all_documents():
    all_texts = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                all_texts.append(f.read())
    # Chunk and flatten
    all_chunks = []
    for text in all_texts:
        all_chunks.extend(chunk_text(text))
    embeddings = embed_model.encode(all_chunks)
    store = FAISSStore(dim=embeddings.shape[1])
    store.add(embeddings, all_chunks)
    return store, all_chunks


# Add slider for number of clusters
num_clusters = st.slider("Number of clusters", min_value=1, max_value=10, value=3)


# Initialize session state for graph
if 'nodes' not in st.session_state:
    st.session_state['nodes'] = []
if 'edges' not in st.session_state:
    st.session_state['edges'] = []
if 'cluster_chunks_map' not in st.session_state:
    st.session_state['cluster_chunks_map'] = None
if 'store' not in st.session_state:
    st.session_state['store'] = None
if 'all_chunks' not in st.session_state:
    st.session_state['all_chunks'] = None

if st.button("Generate Mindmap from All Documents"):
    with st.spinner("Building vector store and generating mindmap from all documents..."):
        store, all_chunks = ingest_all_documents()
        nodes, edges, cluster_chunks_map = run_agent(None, store, embed_model, all_chunks=all_chunks, num_clusters=num_clusters)
        st.session_state['nodes'] = nodes
        st.session_state['edges'] = edges
        st.session_state['cluster_chunks_map'] = cluster_chunks_map
        st.session_state['store'] = store
        st.session_state['all_chunks'] = all_chunks

# Only show graph if nodes exist
if st.session_state['nodes']:
    G = build_graph(st.session_state['nodes'], st.session_state['edges'])
    show_graph(G)

    # interactive expansion
    st.subheader("Expand Graph")
    selected = st.selectbox("Select node", list(G.nodes()))
    if st.button("Expand Selected Node"):
        # Only expand the selected cluster/topic using its chunks
        new_nodes, new_edges, _ = run_agent(
            selected,
            st.session_state['store'],
            embed_model,
            all_chunks=st.session_state['all_chunks'],
            num_clusters=num_clusters,
            cluster_chunks_map=st.session_state['cluster_chunks_map']
        )
        # Only add new nodes/edges that aren't already present
        st.session_state['nodes'].extend([n for n in new_nodes if n not in st.session_state['nodes']])
        st.session_state['edges'].extend([e for e in new_edges if e not in st.session_state['edges']])
        G = build_graph(st.session_state['nodes'], st.session_state['edges'])
        show_graph(G)