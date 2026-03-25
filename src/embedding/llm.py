"""Groq LLM wrapper for entity and relationship extraction."""

import json
import os
from typing import List, Tuple
from functools import lru_cache
import time

from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env file.")

client = Groq(api_key=GROQ_API_KEY)

# Fixed taxonomy of relationship types
RELATIONSHIP_TYPES = {
    "relies_on",
    "requires",
    "influences",
    "supports",
    "contrasts_with",
    "similar_to",
    "depends_on",
    "enables",
    "prevents",
    "related_to",
}

# Simple in-memory cache to avoid duplicate API calls
_entity_cache = {}
_relationship_cache = {}


def extract_entities(texts: List[str], model: str = "mixtral-8x7b-32768") -> List[str]:
    """
    Extract top entities from a list of text chunks using Groq LLM.
    
    Args:
        texts: List of text chunks to extract entities from
        model: Groq model to use
        
    Returns:
        List of extracted entity names (deduplicated)
    """
    if not texts:
        return []
    
    # Create cache key
    cache_key = hash(tuple(texts))
    if cache_key in _entity_cache:
        return _entity_cache[cache_key]
    
    # Combine texts for extraction
    combined_text = "\n\n".join(texts[:5])  # Limit to first 5 chunks to avoid token limits
    
    prompt = f"""Extract the top 5-10 most important entities (concepts, objects, people, organizations, technologies) from the following text.
Return ONLY a JSON array of entity names, no explanations.

Example response format: ["Entity1", "Entity2", "Entity3"]

Text:
{combined_text}

Entities (as JSON array):"""
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Parse JSON response
        entities = json.loads(response_text)
        if not isinstance(entities, list):
            entities = [entities]
        
        # Deduplicate
        entities = list(set(entities))
        
        # Cache result
        _entity_cache[cache_key] = entities
        return entities
        
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse entity extraction response: {response_text}")
        return []
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def extract_relationships(texts: List[str], model: str = "mixtral-8x7b-32768") -> List[Tuple[str, str, str]]:
    """
    Extract typed relationships (subject, relation_type, object) from text chunks using Groq LLM.
    
    Args:
        texts: List of text chunks to extract relationships from
        model: Groq model to use
        
    Returns:
        List of (entity1, relationship_type, entity2) tuples
    """
    if not texts:
        return []
    
    # Create cache key
    cache_key = hash(tuple(texts))
    if cache_key in _relationship_cache:
        return _relationship_cache[cache_key]
    
    # Combine texts
    combined_text = "\n\n".join(texts[:5])  # Limit to first 5 chunks
    
    relation_types_str = ", ".join(sorted(RELATIONSHIP_TYPES))
    prompt = f"""Extract relationships from the following text in the format (entity1, relationship_type, entity2).
Use ONLY these relationship types: {relation_types_str}

Return ONLY a JSON array of triplets, no explanations.
Example response format: [["Entity1", "relies_on", "Entity2"], ["Entity3", "requires", "Entity4"]]

Text:
{combined_text}

Relationships (as JSON array):"""
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Parse JSON response
        relationships = json.loads(response_text)
        if not isinstance(relationships, list):
            relationships = [relationships]
        
        # Validate and filter relationships
        validated = []
        for rel in relationships:
            if isinstance(rel, list) and len(rel) == 3:
                entity1, rel_type, entity2 = rel
                if rel_type in RELATIONSHIP_TYPES:
                    validated.append((str(entity1), str(rel_type), str(entity2)))
        
        # Cache result
        _relationship_cache[cache_key] = validated
        return validated
        
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse relationship extraction response: {response_text}")
        return []
    except Exception as e:
        print(f"Error extracting relationships: {e}")
        return []


def clear_cache():
    """Clear the in-memory caches."""
    global _entity_cache, _relationship_cache
    _entity_cache.clear()
    _relationship_cache.clear()
