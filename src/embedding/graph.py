"""Graph building with Neo4j: construct knowledge graph with typed relationships."""

from typing import Dict, List, Tuple, Any
import os
from neo4j import GraphDatabase, driver
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jGraphStore:
    """Manages Neo4j graph database operations."""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = None
        print(f"Connecting to Neo4j at {uri}")
        
        # Test connection
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                print("✓ Neo4j connection successful")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")
    
    def clear_graph(self):
        """Clear all nodes and relationships from database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Graph cleared")
    
    def add_entities(self, entities: set):
        """
        Add entities as nodes to Neo4j.
        
        Args:
            entities: Set of entity names
        """
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    "MERGE (e:Entity {name: $name})",
                    name=entity
                )
        print(f"Added {len(entities)} entities to graph")
    
    def add_relationships(self, relationships: List[Tuple[str, str, str]]):
        """
        Add relationships between entities.
        
        Args:
            relationships: List of (entity1, relation_type, entity2) tuples
        """
        with self.driver.session() as session:
            for entity1, relation_type, entity2 in relationships:
                # Create relationship with type as property
                cypher = f"""
                MATCH (e1:Entity {{name: $entity1}})
                MATCH (e2:Entity {{name: $entity2}})
                MERGE (e1)-[r:RELATED_TO {{type: $relation_type}}]->(e2)
                ON CREATE SET r.count = 1
                ON MATCH SET r.count = r.count + 1
                """
                session.run(
                    cypher,
                    entity1=entity1,
                    entity2=entity2,
                    relation_type=relation_type
                )
        print(f"Added {len(relationships)} relationships to graph")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        with self.driver.session() as session:
            # Count nodes
            num_nodes = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            
            # Count relationships
            num_edges = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Get relationship types
            rel_types = session.run(
                "MATCH ()-[r]-() RETURN r.type as type, count(*) as count "
            ).data()
            relation_types = {row["type"]: row["count"] for row in rel_types}
            
            # Get top entities by degree
            top_entities = session.run(
                """MATCH (e:Entity)
                WITH e, (size((e)-->()) + size((()<--(e)))) as degree
                ORDER BY degree DESC LIMIT 5
                RETURN e.name as name, degree"""
            ).data()
            top_entities = [(row["name"], row["degree"]) for row in top_entities]
            
            return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "relation_types": relation_types,
                "top_entities": top_entities
            }


def build_graph(processed_data: Dict[str, Any], graph_store: Neo4jGraphStore = None) -> Neo4jGraphStore:
    """
    Build knowledge graph in Neo4j from preprocessed data.
    
    Args:
        processed_data: Output from preprocessing() containing:
            - entities: Set of entity names
            - relationships: List of (entity1, relation_type, entity2) tuples
        graph_store: Neo4jGraphStore instance (creates new if not provided)
            
    Returns:
        Neo4jGraphStore: The graph database connection
    """
    
    entities = processed_data.get("entities", set())
    relationships = processed_data.get("relationships", [])
    
    if graph_store is None:
        graph_store = Neo4jGraphStore()
    
    # Clear existing graph
    graph_store.clear_graph()
    
    # Add entities and relationships
    print(f"Building graph with {len(entities)} entities and {len(relationships)} relationships...")
    graph_store.add_entities(entities)
    graph_store.add_relationships(relationships)
    
    # Print stats
    stats = graph_store.get_graph_stats()
    print(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    return graph_store

