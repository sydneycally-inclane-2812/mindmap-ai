"""Pipeline orchestration with Neo4j: ingest documents and query graph."""

from typing import List, Dict, Any, Tuple, Optional
from .preprocessing import preprocessing
from .graph import build_graph, Neo4jGraphStore


def ingest(documents: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> Neo4jGraphStore:
    """
    Full ingestion pipeline: preprocess documents and build knowledge graph in Neo4j.
    
    Args:
        documents: List of document texts to ingest
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        
    Returns:
        Neo4jGraphStore: Connected graph database
    """
    
    print("=" * 60)
    print("Starting ingestion pipeline...")
    print("=" * 60)
    
    # Step 1: Preprocess
    print("\n[Step 1] Preprocessing documents...")
    processed_data = preprocessing(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Step 2: Build graph in Neo4j
    print("\n[Step 2] Building graph in Neo4j...")
    graph_store = build_graph(processed_data)
    
    print("\n" + "=" * 60)
    print("Ingestion pipeline complete!")
    print("=" * 60)
    
    return graph_store


def query(entity: str, graph_store: Neo4jGraphStore) -> Dict[str, Any]:
    """
    Query the graph for relationships of a given entity.
    
    Args:
        entity: Entity name to query
        graph_store: Neo4jGraphStore instance
        
    Returns:
        Dictionary with keys:
        - incoming: List of (source, relation_type, target) tuples where target is the query entity
        - outgoing: List of (source, relation_type, target) tuples where source is the query entity
        
    Example:
        >>> graph = ingest(documents)
        >>> result = query("marketing", graph)
        >>> print(result["outgoing"])
        [("marketing", "relies_on", "social_media")]
    """
    
    with graph_store.driver.session() as session:
        # Check if entity exists
        entity_exists = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e",
            name=entity
        ).single()
        
        if not entity_exists:
            return {
                "incoming": [],
                "outgoing": [],
                "message": f"Entity '{entity}' not found in graph"
            }
        
        # Get incoming relationships
        incoming_results = session.run(
            """MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity {name: $target})
            RETURN source.name as source, r.type as type, target.name as target""",
            target=entity
        ).data()
        incoming = [(row["source"], row["type"], row["target"]) for row in incoming_results]
        
        # Get outgoing relationships
        outgoing_results = session.run(
            """MATCH (source:Entity {name: $source})-[r:RELATED_TO]->(target:Entity)
            RETURN source.name as source, r.type as type, target.name as target""",
            source=entity
        ).data()
        outgoing = [(row["source"], row["type"], row["target"]) for row in outgoing_results]
    
    return {
        "incoming": incoming,
        "outgoing": outgoing,
        "entity": entity,
        "total_connections": len(incoming) + len(outgoing)
    }


def query_batch(entities: List[str], graph_store: Neo4jGraphStore) -> Dict[str, Dict[str, List]]:
    """
    Query relationships for multiple entities at once.
    
    Args:
        entities: List of entity names to query
        graph_store: Neo4jGraphStore instance
        
    Returns:
        Dictionary where keys are entities and values are query results
    """
    results = {}
    for entity in entities:
        results[entity] = query(entity, graph_store)
    return results


def get_shortest_path(source: str, target: str, graph_store: Neo4jGraphStore) -> Dict[str, Any]:
    """
    Find the shortest path between two entities in the graph.
    
    Args:
        source: Starting entity
        target: Target entity
        graph_store: Neo4jGraphStore instance
        
    Returns:
        Dictionary with:
        - path: List of entities in the shortest path
        - length: Number of hops
        - exists: Whether a path exists
    """
    
    with graph_store.driver.session() as session:
        result = session.run(
            """MATCH path = shortestPath(
                (source:Entity {name: $source})-[*]-(target:Entity {name: $target})
            )
            WHERE source <> target
            RETURN [node in nodes(path) | node.name] as path, length(path) - 1 as hops""",
            source=source,
            target=target
        ).single()
        
        if result:
            path_nodes = result["path"]
            return {
                "source": source,
                "target": target,
                "path": path_nodes,
                "length": result["hops"],
                "exists": True
            }
        else:
            return {
                "source": source,
                "target": target,
                "path": None,
                "length": None,
                "exists": False,
                "message": f"No path exists from '{source}' to '{target}'"
            }


def get_graph_summary(graph_store: Neo4jGraphStore) -> Dict[str, Any]:
    """
    Get summary statistics about the graph.
    
    Args:
        graph_store: Neo4jGraphStore instance
        
    Returns:
        Dictionary with graph statistics
    """
    stats = graph_store.get_graph_stats()
    
    with graph_store.driver.session() as session:
        # Calculate avg degree
        avg_degree_result = session.run(
            """MATCH (n:Entity)
            RETURN avg(size((n)-->()) + size((()<--(n)))) as avg_degree"""
        ).single()
        avg_degree = avg_degree_result["avg_degree"] if avg_degree_result else 0
    
    return {
        "num_nodes": stats["num_nodes"],
        "num_edges": stats["num_edges"],
        "relation_types": stats["relation_types"],
        "avg_degree": avg_degree,
        "top_entities": stats["top_entities"]
    }


def get_entity_neighbors(entity: str, graph_store: Neo4jGraphStore, depth: int = 1) -> Dict[str, Any]:
    """
    Get all neighbors of an entity up to a certain depth.
    
    Args:
        entity: Entity name
        graph_store: Neo4jGraphStore instance
        depth: Maximum relationship depth to traverse
        
    Returns:
        Dictionary with entity, neighbors, and their relationships
    """
    
    with graph_store.driver.session() as session:
        result = session.run(
            f"""MATCH (e:Entity {{name: $name}})-[r:RELATED_TO*1..{depth}]-(neighbor:Entity)
            RETURN DISTINCT neighbor.name as neighbor, 
                   min(r.type) as relation_type
            ORDER BY neighbor""",
            name=entity
        ).data()
        
        neighbors = [(row["neighbor"], row["relation_type"]) for row in result]
        
        return {
            "entity": entity,
            "neighbors": neighbors,
            "num_neighbors": len(neighbors),
            "depth": depth
        }

