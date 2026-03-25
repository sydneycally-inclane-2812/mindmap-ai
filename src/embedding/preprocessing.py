from components.documents_ingestion import chunk_text
from components.vectorstore import FAISSStore


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
    relationships = extract_relationships(all_chunks)
    
    print(f"Extracted {len(entities_set)} entities and {len(relationships)} relationships")
    
    return {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "entities": entities_set,
        "relationships": relationships,
        "faiss_store": faiss_store,
    }
