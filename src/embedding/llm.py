"""Groq LLM wrapper for entity and relationship extraction."""

import json
import os
from typing import List, Tuple

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


def _extract_json_payload(raw_text: str) -> str:
    """Extract a JSON payload from raw model output.

    Handles common LLM formatting such as fenced code blocks:
    ```json
    [...]
    ```
    """
    text = (raw_text or "").strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Keep only JSON-looking section (array or object).
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [idx for idx in (first_obj, first_arr) if idx != -1]
    if not starts:
        return text

    start = min(starts)
    last_obj = text.rfind("}")
    last_arr = text.rfind("]")
    end = max(last_obj, last_arr)
    if end == -1 or end < start:
        return text[start:].strip()
    return text[start : end + 1].strip()


def _normalize_relation_type(rel_type: str) -> str:
    """Normalize LLM relation labels to the fixed taxonomy format."""
    normalized = str(rel_type).strip().lower().replace("-", "_").replace(" ", "_")
    # Common variants from LLM outputs.
    if normalized.startswith("is_"):
        normalized = normalized[3:]
    if normalized.endswith("_to") and normalized not in RELATIONSHIP_TYPES:
        # e.g. similar_to remains unchanged if valid.
        pass
    if normalized == "is_similar_to":
        normalized = "similar_to"
    if normalized == "is_related_to":
        normalized = "related_to"
    return normalized


def extract_entities(texts: List[str], model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> List[str]:
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
        message = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.choices[0].message.content.strip()
        json_text = _extract_json_payload(response_text)
        
        # Parse JSON response
        entities = json.loads(json_text)
        if not isinstance(entities, list):
            entities = [entities]
        
        # Deduplicate
        entities = list(set(entities))
        
        print(f"[DEBUG] Extracted {len(entities)} entities")
        
        # Cache result
        _entity_cache[cache_key] = entities
        return entities
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse entity extraction response: {response_text}")
        print(f"JSON Error: {e}")
        return []
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def extract_relationships(
    texts: List[str],
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    entities: List[str] = None,
) -> List[Tuple[str, str, str]]:
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
    entity_constraint = ""
    if entities:
        canonical_entities = sorted({str(e).strip() for e in entities if str(e).strip()})
        entity_constraint = (
            "\nUse ONLY entities from this canonical list for entity1 and entity2. "
            "Do not invent new entity names.\n"
            f"Canonical entities: {json.dumps(canonical_entities)}\n"
        )

    prompt = f"""Extract relationships from the following text in the format (entity1, relationship_type, entity2).
Use ONLY these relationship types: {relation_types_str}. Do not use any other types of relationships.
{entity_constraint}

Return ONLY a JSON array of triplets, no explanations.
Example response format: [["Entity1", "relies_on", "Entity2"], ["Entity3", "requires", "Entity4"]]

Text:
{combined_text}

Relationships (as JSON array, DO NOT USE RELATIONSHIPS OUTSIDE OF THE LIST GIVEN):"""
    
    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.choices[0].message.content.strip()
        json_text = _extract_json_payload(response_text)
        print(f"[DEBUG] Relationship raw response (first 200 chars): {response_text[:200]}")
        
        # Parse JSON response
        relationships = json.loads(json_text)
        if not isinstance(relationships, list):
            relationships = [relationships]
        
        print(f"[DEBUG] Raw relationships count: {len(relationships)}")
        if relationships:
            print(f"[DEBUG] First raw relationship: {relationships[0]}")
        
        # Validate and filter relationships
        validated = []
        for rel in relationships:
            if isinstance(rel, list) and len(rel) == 3:
                entity1, rel_type, entity2 = rel
                normalized_rel_type = _normalize_relation_type(rel_type)
                print(
                    f"[DEBUG] Checking relationship: {entity1} -{normalized_rel_type}-> {entity2}, "
                    f"type in set: {normalized_rel_type in RELATIONSHIP_TYPES}"
                )
                if normalized_rel_type in RELATIONSHIP_TYPES:
                    validated.append((str(entity1), normalized_rel_type, str(entity2)))
        
        print(f"[DEBUG] Extracted {len(validated)} valid relationships from {len(relationships)} raw")
        
        # Cache result
        _relationship_cache[cache_key] = validated
        return validated
        
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}")
        print(f"[DEBUG] Response text: {response_text}")
        return []
    except Exception as e:
        print(f"Error extracting relationships: {e}")
        import traceback
        traceback.print_exc()
        return []


def clear_cache():
    """Clear the in-memory caches."""
    global _entity_cache, _relationship_cache
    _entity_cache.clear()
    _relationship_cache.clear()
