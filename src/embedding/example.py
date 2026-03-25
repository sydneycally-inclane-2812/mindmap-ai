"""Example usage and quick test of the embedding pipeline with Neo4j."""

import sys
import os
import logging
import logging.config
from pathlib import Path
import yaml

# Add workspace root to path so we can import src.embedding
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.embedding import ingest, query, get_graph_summary, query_batch, get_entity_neighbors


def configure_logging(mode: str = "dev"):
	"""Load logging config from YAML and return a mode-specific logger."""
	config_path = Path(workspace_root) / "config" / "logger_config.yaml"
	with open(config_path, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)
	logging.config.dictConfig(config)
	return logging.getLogger(mode)


logger = configure_logging("dev")


def format_relation_chain(path_nodes, relation_types):
	"""Format a path into chain notation: A -rel-> B -rel-> C."""
	if not path_nodes:
		return ""
	if not relation_types:
		return " -> ".join(path_nodes)

	parts = [path_nodes[0]]
	for idx, rel in enumerate(relation_types):
		if idx + 1 < len(path_nodes):
			parts.append(f"-{rel}->")
			parts.append(path_nodes[idx + 1])
	return " ".join(parts)


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
	
	# Ingest documents and build graph in Neo4j
	try:
		graph_store = ingest(documents, chunk_size=500, chunk_overlap=100)
		
		# Print graph summary
		logger.debug("Graph Summary:")
		summary = get_graph_summary(graph_store)
		logger.debug("  Nodes: %s", summary["num_nodes"])
		logger.debug("  Edges: %s", summary["num_edges"])
		logger.debug("  Avg Degree: %.2f", summary["avg_degree"])
		logger.debug("  Relationship types: %s", summary["relation_types"])
		logger.debug("  Top entities: %s", summary["top_entities"][:3])
		
		# Query example 1: Single entity
		logger.debug("Query 1: Relationships for 'Marketing'")
		result = query("Marketing", graph_store)
		logger.debug("  Incoming (%s):", len(result["incoming"]))
		for source, rel_type, target in result['incoming']:
			logger.debug("    %s --[%s]--> %s", source, rel_type, target)
		logger.debug("  Outgoing (%s):", len(result["outgoing"]))
		for source, rel_type, target in result['outgoing']:
			logger.debug("    %s --[%s]--> %s", source, rel_type, target)
		logger.debug("  Total connections: %s", result["total_connections"])
		
		# Query example 2: Neighbors
		logger.debug("Query 2: Neighbors of 'Marketing' (depth 2)")
		neighbors = get_entity_neighbors("Marketing", graph_store, depth=2)
		logger.debug("  Entity: %s", neighbors["entity"])
		logger.debug("  Neighbors (%s):", neighbors["num_neighbors"])
		for neighbor, _ in neighbors["neighbors"]:
			path_details = neighbors.get("details", {}).get(neighbor, [])
			for path in sorted(path_details, key=lambda p: p.get("hops", 0)):
				chain = format_relation_chain(path.get("path", []), path.get("relation_types", []))
				logger.debug("    - %s", chain)

		# Query example 2: Neighbors
		logger.debug("Query 2: Neighbors of 'Computer Vision' (depth 2)")
		neighbors = get_entity_neighbors("Computer Vision", graph_store, depth=2)
		logger.debug("  Entity: %s", neighbors["entity"])
		logger.debug("  Neighbors (%s):", neighbors["num_neighbors"])
		for neighbor, _ in neighbors["neighbors"]:
			path_details = neighbors.get("details", {}).get(neighbor, [])
			for path in sorted(path_details, key=lambda p: p.get("hops", 0)):
				chain = format_relation_chain(path.get("path", []), path.get("relation_types", []))
				logger.debug("    - %s", chain)
			
		# Query example 3: Batch query
		logger.debug("Query 3: Batch query for multiple entities")
		batch_results = query_batch(["Marketing", "Data", "Social Media"], graph_store)
		for entity, result in batch_results.items():
			logger.debug("  %s: %s connections", entity, result["total_connections"])
		
		# Keep connection open
		logger.debug("Graph store ready for queries. Call graph_store.close() to disconnect.")
	
	except Exception as e:
		logger.exception("Error during ingestion: %s", e)
	finally:
		graph_store.close()


if __name__ == "__main__":
	main()
