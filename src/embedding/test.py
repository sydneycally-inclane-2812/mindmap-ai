from typing import Dict, List, Tuple, Any
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path

homedir = Path(__file__).parent.parent.parent.resolve()

load_dotenv(homedir / '.env')

# Neo4j Aura connection configuration (read from .env)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print(NEO4J_URI)
print(NEO4J_USER)
print(NEO4J_PASSWORD)

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError(
        "Neo4j credentials not found in .env. "
        "Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD. "
        "For Neo4j Aura, use format: neo4j+s://xxxxx.databases.neo4j.io"
    )