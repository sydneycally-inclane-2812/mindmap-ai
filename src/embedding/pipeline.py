"""Pipeline orchestration with Neo4j: ingest documents and query graph."""

from typing import List, Dict, Any, Tuple, Optional
import logging
import duckdb
from pathlib import Path
from .preprocessing import preprocessing
from .graph import build_graph, Neo4jGraphStore
import os
logger = logging.getLogger(__name__)
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    logger.info("Starting ingestion pipeline...")
    
    graph_store = Neo4jGraphStore()

    # Step 1: Pull existing graph entities to guide extraction.
    existing_entities = graph_store.list_entities()
    logger.info("Loaded %s existing entities from Neo4j", len(existing_entities))

    # Step 2: Preprocess
    logger.info("Preprocessing documents...")
    processed_data = preprocessing(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seed_entities=existing_entities,
    )
    
    # Step 3: Build graph in Neo4j
    logger.info("Building graph in Neo4j...")
    graph_store = build_graph(processed_data, graph_store=graph_store)
    
    logger.info("Ingestion pipeline complete!")
    
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
                "entity": entity,
                "total_connections": 0,
                "message": f"Entity '{entity}' not found in graph"
            }

        # Avoid warning-heavy queries when this node has no relationships.
        connection_count_row = session.run(
            "MATCH (e:Entity {name: $name})-[r]-() RETURN count(r) as c",
            name=entity,
        ).single()
        connection_count = connection_count_row["c"] if connection_count_row else 0
        if connection_count == 0:
            return {
                "incoming": [],
                "outgoing": [],
                "entity": entity,
                "total_connections": 0,
            }
        
        # Get incoming relationships
        incoming_results = session.run(
            """MATCH (source:Entity)-[r]->(target:Entity {name: $target})
            RETURN source.name as source, coalesce(r.type, 'unknown') as type, target.name as target""",
            target=entity
        ).data()
        incoming = [(row["source"], row["type"], row["target"]) for row in incoming_results]
        
        # Get outgoing relationships
        outgoing_results = session.run(
            """MATCH (source:Entity {name: $source})-[r]->(target:Entity)
            RETURN source.name as source, coalesce(r.type, 'unknown') as type, target.name as target""",
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
        # Calculate avg degree (using COUNT {} syntax for newer Neo4j)
        avg_degree_result = session.run(
            """MATCH (n:Entity)
            WITH n, (COUNT {(n)-[]->()} + COUNT {(n)<-[]-()}) as degree
            RETURN avg(degree) as avg_degree"""
        ).single()
        raw_avg_degree = avg_degree_result["avg_degree"] if avg_degree_result else None
        avg_degree = float(raw_avg_degree) if raw_avg_degree is not None else 0.0
    
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
        # Return per-path details so callers can distinguish direct vs indirect neighbors.
        result = session.run(
            """
            MATCH path = (e:Entity {name: $name})-[rels*1..]-(neighbor:Entity)
            WHERE length(path) <= $depth
            WITH neighbor,
                 length(path) as hops,
                 [n IN nodes(path) | n.name] as node_path,
                 [r IN rels | coalesce(r.type, 'unknown')] as relation_path
            RETURN neighbor.name as neighbor,
                   collect(DISTINCT {
                       hops: hops,
                       node_path: node_path,
                       relation_path: relation_path
                   }) as paths
            ORDER BY neighbor
            """,
            name=entity,
            depth=depth,
        ).data()

        neighbors = []
        details = {}
        for row in result:
            neighbor_name = row["neighbor"]
            path_details = row["paths"]

            # Aggregate unique relation labels for quick summary.
            rel_types = sorted(
                {
                    rel
                    for path in path_details
                    for rel in path.get("relation_path", [])
                }
            )

            normalized_paths = []
            for path in path_details:
                node_path = path.get("node_path", [])
                normalized_paths.append(
                    {
                        "hops": path.get("hops", len(node_path) - 1),
                        "path": node_path,
                        "via": node_path[1:-1] if len(node_path) > 2 else [],
                        "relation_types": path.get("relation_path", []),
                    }
                )

            neighbors.append((neighbor_name, rel_types))
            details[neighbor_name] = normalized_paths

        return {
            "entity": entity,
            "neighbors": neighbors,
            "details": details,
            "num_neighbors": len(neighbors),
            "depth": depth,
        }

def get_evidence_for_entity(entity: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return relation evidence snippets for an entity from DuckDB."""
    evidence_db_path = Path(workspace_root) / "databases" / "evidence.duckdb"
    if not evidence_db_path.exists():
        logger.debug("Evidence DB not found at %s", evidence_db_path)
        return []

    with duckdb.connect(str(evidence_db_path), read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT
                r.source_canonical_name AS source,
                r.relation_type AS relation,
                r.target_canonical_name AS target,
                max(coalesce(re.support_score, 0.0)) AS score,
                coalesce(re.evidence_text, '') AS evidence
            FROM relations r
            JOIN relation_evidence re ON r.relation_id = re.relation_id
            WHERE lower(r.source_canonical_name) = lower(?)
               OR lower(r.target_canonical_name) = lower(?)
            GROUP BY source, relation, target, evidence
            ORDER BY score DESC, source, relation, target
            LIMIT ?
            """,
            [entity, entity, limit],
        ).fetchall()

    return [
        {
            "source": source,
            "relation": relation,
            "target": target,
            "score": float(score or 0.0),
            "evidence": evidence or "",
        }
        for source, relation, target, score, evidence in rows
    ]


def show_evidence_for_entity(entity: str, limit: int = 10) -> None:
    """Backwards-compatible logger view for entity evidence."""
    evidence_rows = get_evidence_for_entity(entity, limit=limit)
    if not evidence_rows:
        logger.debug("  No evidence found for '%s'", entity)
        return

    logger.debug("Evidence for '%s' (top %s):", entity, len(evidence_rows))
    for row in evidence_rows:
        logger.debug(
            "  %s --[%s]--> %s (score=%.2f)",
            row["source"],
            row["relation"],
            row["target"],
            row["score"],
        )
        logger.debug("    evidence: %s", row["evidence"] or "(empty)")