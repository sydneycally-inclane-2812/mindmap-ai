"""Preprocessing pipeline: chunk, embed, extract entities and relationships."""

from typing import List, Dict, Any
import sys
import os
import importlib.util

# Reduce transformer loading noise in CLI output.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import from components-tbd (which has hyphens, so we need to use importlib)
def _import_module(module_path, module_name):
    """Helper to import modules from paths with hyphens."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.join(os.path.dirname(__file__), '../..')
documents_ingestion = _import_module(
    os.path.join(base_path, 'components-tbd', 'documents_ingestion.py'),
    'documents_ingestion'
)
vectorstore = _import_module(
    os.path.join(base_path, 'components-tbd', 'vectorstore.py'),
    'vectorstore'
)

chunk_text = documents_ingestion.chunk_text
FAISSStore = vectorstore.FAISSStore

from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
from .llm import extract_entities, extract_relationships

hf_logging.set_verbosity_error()


def preprocessing(documents: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> Dict[str, Any]:
    """
    Preprocess documents: chunk, embed, extract entities and relationships.
    
    Args:
        documents: List of document texts
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary with keys:
        - chunks: List of text chunks
        - embeddings: List of embedding vectors
        - entities: Set of extracted entities
        - relationships: List of (entity1, relation_type, entity2) tuples
        - faiss_store: FAISSStore object with persisted embeddings
    """
    
    if not documents:
        raise ValueError("No documents provided")
    
    # Step 1: Chunk all documents
    print("Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Step 2: Embed chunks
    print("Embedding chunks...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    
    # Step 3: Store embeddings in FAISS
    print("Storing embeddings in FAISS...")
    faiss_store = FAISSStore(dim=embeddings.shape[1], storage_dir="vectorstorage", name="faiss_index")
    faiss_store.add(embeddings, all_chunks)
    
    # Step 4: Extract entities from all chunks
    print("Extracting entities...")
    entities_list = extract_entities(all_chunks)
    entities_set = set(entities_list)
    
    # Step 5: Extract relationships from all chunks
    print("Extracting relationships...")
    relationships = extract_relationships(all_chunks, entities=list(entities_set))
    
    print(f"Extracted {len(entities_set)} entities and {len(relationships)} relationships")
    
    return {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "entities": entities_set,
        "relationships": relationships,
        "faiss_store": faiss_store,
    }
