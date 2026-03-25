"""Groq LLM wrapper for entity and relationship extraction."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import logging

from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env file.")

client = Groq(api_key=GROQ_API_KEY)
logger = logging.getLogger(__name__)

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
_relationship_evidence_cache = {}


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
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = (message.choices[0].message.content or "").strip()
        json_text = _extract_json_payload(response_text)
        
        # Parse JSON response
        entities = json.loads(json_text)
        if not isinstance(entities, list):
            entities = [entities]
        
        # Deduplicate
        entities = list(set(entities))
        
        logger.debug("Extracted %s entities", len(entities))
        
        # Cache result
        _entity_cache[cache_key] = entities
        return entities
        
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse entity extraction response: %s", response_text)
        logger.warning("JSON Error: %s", e)
        return []
    except Exception as e:
        logger.error("Error extracting entities: %s", e)
        return []


def extract_relationships(
    texts: List[str],
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    entities: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Extract typed relationships (subject, relation_type, object) from text chunks using Groq LLM.
    
    Args:
        texts: List of text chunks to extract relationships from
        model: Groq model to use
        
    Returns:
        List of (entity1, relationship_type, entity2) tuples
    """
    records = extract_relationships_with_evidence(texts=texts, model=model, entities=entities)
    return [(r["source"], r["relation"], r["target"]) for r in records]


def extract_relationships_with_evidence(
    texts: List[str],
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    entities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract relationships with short evidence snippets and confidence."""
    if not texts:
        return []

    cache_key = hash((tuple(texts), tuple(sorted(entities or []))))
    if cache_key in _relationship_evidence_cache:
        return _relationship_evidence_cache[cache_key]

    combined_text = "\n\n".join(texts[:5])
    relation_types_str = ", ".join(sorted(RELATIONSHIP_TYPES))
    entity_constraint = ""
    if entities:
        canonical_entities = sorted({str(e).strip() for e in entities if str(e).strip()})
        entity_constraint = (
            "\nUse ONLY entities from this canonical list for source and target. "
            "Do not invent new entity names.\n"
            f"Canonical entities: {json.dumps(canonical_entities)}\n"
        )

    prompt = f"""Extract relations from the text.
Use ONLY these relation types: {relation_types_str}.
{entity_constraint}

Return ONLY a JSON array of objects, each object must contain:
- source (string)
- relation (string)
- target (string)
- evidence (short supporting quote or snippet, max 20 words)
- confidence (number from 0 to 1)

Example:
[
  {{"source": "marketing", "relation": "relies_on", "target": "social_media", "evidence": "Marketing relies heavily on social media platforms.", "confidence": 0.92}}
]

Text:
{combined_text}
"""

    try:
        message = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = (message.choices[0].message.content or "").strip()
        json_text = _extract_json_payload(response_text)
        logger.debug("Relationship+evidence raw response (first 200 chars): %s", response_text[:200])

        raw_records = json.loads(json_text)
        if not isinstance(raw_records, list):
            raw_records = [raw_records]

        validated: List[Dict[str, Any]] = []
        for rec in raw_records:
            if not isinstance(rec, dict):
                continue

            source = str(rec.get("source", "")).strip()
            target = str(rec.get("target", "")).strip()
            rel_type = _normalize_relation_type(str(rec.get("relation", "")).strip())
            evidence = str(rec.get("evidence", "")).strip()
            confidence = rec.get("confidence", 0.0)

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            if source and target and rel_type in RELATIONSHIP_TYPES:
                validated.append(
                    {
                        "source": source,
                        "relation": rel_type,
                        "target": target,
                        "evidence": evidence,
                        "confidence": confidence,
                    }
                )

        logger.debug("Extracted %s relation-evidence records", len(validated))

        _relationship_evidence_cache[cache_key] = validated
        # Keep backward-compatible tuple cache too.
        _relationship_cache[hash(tuple(texts))] = [
            (r["source"], r["relation"], r["target"]) for r in validated
        ]
        return validated

    except json.JSONDecodeError as e:
        logger.debug("JSON parse error for relation evidence: %s", e)
        logger.debug("Response text: %s", response_text)
        return []
    except Exception as e:
        logger.exception("Error extracting relationships with evidence: %s", e)
        return []


def clear_cache():
    """Clear the in-memory caches."""
    global _entity_cache, _relationship_cache, _relationship_evidence_cache
    _entity_cache.clear()
    _relationship_cache.clear()
    _relationship_evidence_cache.clear()
