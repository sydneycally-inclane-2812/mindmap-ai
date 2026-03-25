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


def _batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """Split a list of chunks into fixed-size batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [texts[idx : idx + batch_size] for idx in range(0, len(texts), batch_size)]


def _filter_entities_for_batch(
    text: str,
    entities: List[str],
    max_entities: int = 80,
) -> List[str]:
    """Select a bounded subset of entities likely relevant to this text batch."""
    if not text or not entities:
        return []

    lowered_text = text.casefold()
    matched = []
    for entity in entities:
        normalized = str(entity).strip()
        if not normalized:
            continue
        if normalized.casefold() in lowered_text:
            matched.append(normalized)
        if len(matched) >= max_entities:
            break
    return matched


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


def extract_entities(
    texts: List[str],
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    batch_size: int = 4,
    seed_entities: Optional[List[str]] = None,
) -> List[str]:
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
    normalized_seeds = tuple(sorted({str(e).strip() for e in (seed_entities or []) if str(e).strip()}))
    cache_key = hash((tuple(texts), model, batch_size, normalized_seeds))
    if cache_key in _entity_cache:
        return _entity_cache[cache_key]

    try:
        entities_set = set()
        batches = _batch_texts(texts, batch_size=batch_size)

        for batch in batches:
            combined_text = "\n\n".join(batch)
            matched_seed_entities = _filter_entities_for_batch(combined_text, list(normalized_seeds))
            seed_hint = ""
            if matched_seed_entities:
                seed_hint = (
                    "\nPrefer these existing graph entity names when they appear in the text. "
                    "Keep exact casing/spelling where possible.\n"
                    f"Existing graph entities in context: {json.dumps(matched_seed_entities)}\n"
                )

            prompt = f"""Extract up to 20 important entities (concepts, objects, people, organizations, technologies) from the following text.
{seed_hint}
Return ONLY a JSON array of entity names, no explanations.

Example response format: ["Entity1", "Entity2", "Entity3"]

Text:
{combined_text}

Entities (as JSON array):"""

            message = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = (message.choices[0].message.content or "").strip()
            json_text = _extract_json_payload(response_text)
            batch_entities = json.loads(json_text)
            if not isinstance(batch_entities, list):
                batch_entities = [batch_entities]

            for entity in batch_entities:
                normalized = str(entity).strip()
                if normalized:
                    entities_set.add(normalized)

        entities = sorted(entities_set)
        logger.debug("Extracted %s entities across %s batches", len(entities), len(batches))

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
    batch_size: int = 3,
) -> List[Dict[str, Any]]:
    """Extract relationships with short evidence snippets and confidence."""
    if not texts:
        return []

    cache_key = hash((tuple(texts), tuple(sorted(entities or [])), model, batch_size))
    if cache_key in _relationship_evidence_cache:
        return _relationship_evidence_cache[cache_key]

    relation_types_str = ", ".join(sorted(RELATIONSHIP_TYPES))
    canonical_entities = sorted({str(e).strip() for e in entities or [] if str(e).strip()})

    try:
        merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        batches = _batch_texts(texts, batch_size=batch_size)

        for batch in batches:
            combined_text = "\n\n".join(batch)
            local_canonical = _filter_entities_for_batch(combined_text, canonical_entities)
            entity_constraint = ""
            if local_canonical:
                entity_constraint = (
                    "\nUse ONLY entities from this canonical list for source and target. "
                    "Do not invent new entity names.\n"
                    f"Canonical entities: {json.dumps(local_canonical)}\n"
                )

            prompt = f"""Extract relations from the text, as many as you can reasonably infer from the text.
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

            message = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = (message.choices[0].message.content or "").strip()
            json_text = _extract_json_payload(response_text)
            logger.debug("Relationship+evidence raw response (first 200 chars): %s", response_text[:200])

            raw_records = json.loads(json_text)
            if not isinstance(raw_records, list):
                raw_records = [raw_records]

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

                if not (source and target and rel_type in RELATIONSHIP_TYPES):
                    continue

                key = (source.casefold(), rel_type, target.casefold(), evidence.casefold())
                existing = merged.get(key)
                candidate = {
                    "source": source,
                    "relation": rel_type,
                    "target": target,
                    "evidence": evidence,
                    "confidence": confidence,
                }
                if existing is None or candidate["confidence"] > existing["confidence"]:
                    merged[key] = candidate

        validated = list(merged.values())
        logger.debug("Extracted %s relation-evidence records across %s batches", len(validated), len(batches))

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
