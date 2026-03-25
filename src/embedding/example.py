"""Example usage and quick test of the embedding pipeline with Neo4j."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from embedding import ingest, query, get_graph_summary, query_batch, get_entity_neighbors


def main():
    """
    Example: Ingest documents and query the knowledge graph in Neo4j.
    """
    
    # Sample documents (in real usage, read from files)
    documents = [
        """
        Marketing relies heavily on social media platforms. Social media 
        enables direct customer engagement and brand awareness. Digital marketing 
        requires data analysis and audience segmentation. Marketing campaigns 
        support sales goals and revenue generation.
        """,
        """
        Machine Learning requires large amounts of quality data. Deep Learning 
        is a subset of Machine Learning that uses neural networks. Neural Networks 
        are inspired by biological neurons. Data preprocessing is essential for 
        ML model training. Feature engineering influences model performance.
        """,
    ]
    
    print("Document 1:", documents[0][:100], "...")
    print("Document 2:", documents[1][:100], "...")
    print()
    
    # Ingest documents and build graph in Neo4j
    try:
        graph_store = ingest(documents, chunk_size=500, chunk_overlap=100)
        
        # Print graph summary
        print("\nGraph Summary:")
        summary = get_graph_summary(graph_store)
        print(f"  Nodes: {summary['num_nodes']}")
        print(f"  Edges: {summary['num_edges']}")
        print(f"  Avg Degree: {summary['avg_degree']:.2f}")
        print(f"  Relationship types: {summary['relation_types']}")
        print(f"  Top entities: {summary['top_entities'][:3]}")
        print()
        
        # Query example 1: Single entity
        print("Query 1: Relationships for 'Marketing'")
        result = query("Marketing", graph_store)
        print(f"  Incoming ({len(result['incoming'])}):")
        for source, rel_type, target in result['incoming']:
            print(f"    {source} --[{rel_type}]--> {target}")
        print(f"  Outgoing ({len(result['outgoing'])}):")
        for source, rel_type, target in result['outgoing']:
            print(f"    {source} --[{rel_type}]--> {target}")
        print(f"  Total connections: {result['total_connections']}")
        print()
        
        # Query example 2: Neighbors
        print("Query 2: Neighbors of 'Marketing' (depth 1)")
        neighbors = get_entity_neighbors("Marketing", graph_store, depth=1)
        print(f"  Entity: {neighbors['entity']}")
        print(f"  Neighbors ({neighbors['num_neighbors']}):")
        for neighbor, rel_type in neighbors['neighbors']:
            print(f"    - {neighbor} ({rel_type})")
        print()
        
        # Query example 3: Batch query
        print("Query 3: Batch query for multiple entities")
        batch_results = query_batch(["Marketing", "Data", "Social Media"], graph_store)
        for entity, result in batch_results.items():
            print(f"  {entity}: {result['total_connections']} connections")
        
        # Keep connection open
        print("\nGraph store ready for queries. Call graph_store.close() to disconnect.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
