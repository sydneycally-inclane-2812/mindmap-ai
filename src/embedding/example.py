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

from src.embedding import ingest, query, get_graph_summary, query_batch, get_entity_neighbors, get_evidence_for_entity


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
		Marketing is the act of acquiring, satisfying and retaining customers.[3]It is one of the primary components of business management and commerce.[4]

		Marketing is usually conducted by the seller, typically a retailer or manufacturer. Products can be marketed to other businesses (B2B) or directly to consumers (B2C).[5] Sometimes tasks are contracted to dedicated marketing firms, like a media, market research, or advertising agency. Sometimes, a trade association or government agency (such as the Agricultural Marketing Service) advertises on behalf of an entire industry or locality, often a specific type of food (e.g. Got Milk?), food from a specific area, or a city or region as a tourism destination.

		Market orientations are philosophies concerning the factors that should go into market planning.[6] The marketing mix, which outlines the specifics of the product and how it will be sold, including the channels that will be used to advertise the product,[7][8] is affected by the environment surrounding the product,[9] the results of marketing research and market research,[10] and the characteristics of the product's target market.[11] Once these factors are determined, marketers must then decide what methods of promoting the product,[5] including use of coupons and other price inducements.[12] 
		""",
		"""
		Distribution is the process of making a product or service available for the consumer or business user who needs it, and a distributor is a business involved in the distribution stage of the value chain. Distribution can be done directly by the producer or service provider or by using indirect channels with distributors or intermediaries. Distribution (or place) is one of the four elements of the marketing mix: the other three elements being product, pricing, and promotion.

		Decisions about distribution need to be taken in line with a company's overall strategic vision and mission. Developing a coherent distribution plan is a central component of strategic planning. At the strategic level, as well as deciding whether to distribute directly or via a distribution network, there are three broad approaches to distribution, namely mass, selective and exclusive distribution. The number and type of intermediaries selected largely depends on the strategic approach. The overall distribution channel should add value to the consumer. 
		""",
	]
	
	# Ingest documents and build graph in Neo4j
	try:
		# Smaller chunks with overlap typically increase relation recall.
		graph_store = ingest(documents, chunk_size=350, chunk_overlap=120)
		
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

		# Evidence example: read supporting snippets stored during preprocessing.
		logger.debug("Query 4: Evidence snippets for 'Marketing'")
		evidence_rows = get_evidence_for_entity("Marketing", limit=5)
		if not evidence_rows:
			logger.debug("  No evidence found for 'Marketing'")
		else:
			for row in evidence_rows:
				logger.debug(
					"  %s --[%s]--> %s (score=%.2f)",
					row["source"],
					row["relation"],
					row["target"],
					row["score"],
				)
				logger.debug("    evidence: %s", row["evidence"] or "(empty)")
		
		# Keep connection open
		logger.debug("Graph store ready for queries. Call graph_store.close() to disconnect.")
	
	except Exception as e:
		logger.exception("Error during ingestion: %s", e)
	finally:
		graph_store.close()


if __name__ == "__main__":
	main()
