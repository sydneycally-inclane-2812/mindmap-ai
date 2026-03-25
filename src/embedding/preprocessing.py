"""Preprocessing pipeline: chunk, embed, extract entities and relationships."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import sys
import os
import importlib.util
import logging
import faiss
import numpy as np

import os
import pickle

from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
from .llm import extract_entities, extract_relationships_with_evidence
from .evidence_store import EvidenceDuckDBStore

hf_logging.set_verbosity_error()
logger = logging.getLogger(__name__)
# Reduce transformer loading noise in CLI output.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class FAISSStore:
    def __init__(self, dim, storage_dir="vectorstorage", name="faiss_index"):
        self.dim = dim
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, f"{name}.index")
        self.texts_path = os.path.join(storage_dir, f"{name}_texts.pkl")

        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        self.save()

    def search(self, query_embedding, k=5):
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            raise FileNotFoundError("No saved FAISS index or texts found.")
        
def chunk_text(text, chunk_size=500, chunk_overlap=100):
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap
	)
	return splitter.split_text(text)



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
    logger.info("Chunking documents...")
    all_chunks = []
    chunk_records = []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            all_chunks.append(chunk)
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": f"doc_{doc_idx}",
                    "chunk_text": chunk,
                    "chunk_order": chunk_idx,
                }
            )
    
    logger.info("Created %s chunks", len(all_chunks))
    
    # Step 2: Embed chunks
    logger.info("Embedding chunks...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    
    # Step 3: Store embeddings in FAISS
    logger.info("Storing embeddings in FAISS...")
    faiss_store = FAISSStore(dim=embeddings.shape[1], storage_dir="vector_storage", name="faiss_index")
    faiss_store.add(embeddings, all_chunks)
    
    # Step 4: Extract entities from all chunks
    logger.info("Extracting entities...")
    entities_list = extract_entities(all_chunks)
    entities_set = set(entities_list)
    
    # Step 5: Extract relationships from all chunks
    logger.info("Extracting relationships...")
    relation_records = extract_relationships_with_evidence(all_chunks, entities=list(entities_set))
    relationships = [(r["source"], r["relation"], r["target"]) for r in relation_records]

    # Step 6: Persist evidence and relation provenance in DuckDB
    logger.info("Persisting evidence in DuckDB...")
    evidence_store = EvidenceDuckDBStore(db_path="data/evidence.duckdb")
    evidence_store.store_ingestion(
        documents=documents,
        chunks=chunk_records,
        entities=sorted(entities_set),
        relation_records=relation_records,
    )
    evidence_store.close()
    
    logger.info(
        "Extracted %s entities and %s relationships",
        len(entities_set),
        len(relationships),
    )
    
    return {
        "chunks": all_chunks,
        "chunk_records": chunk_records,
        "embeddings": embeddings,
        "entities": entities_set,
        "relationships": relationships,
        "relation_records": relation_records,
        "faiss_store": faiss_store,
    }
