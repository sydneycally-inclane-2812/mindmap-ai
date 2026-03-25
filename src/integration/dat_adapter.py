from typing import Dict, Any, Set, List

from src.embedding.pipeline import query


def dat_query_result_to_mindmap_data(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Dat's graph query result into the interface format:
    {
        "nodes": [...],
        "edges": [{"source": ..., "target": ..., "label": ...}]
    }
    """

    nodes: Set[str] = set()
    edges: List[Dict[str, str]] = []

    entity = query_result.get("entity")
    if entity:
        nodes.add(str(entity))

    incoming = query_result.get("incoming", [])
    outgoing = query_result.get("outgoing", [])

    for rel in incoming:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue

        source, rel_type, target = rel
        source = str(source)
        rel_type = str(rel_type)
        target = str(target)

        nodes.add(source)
        nodes.add(target)

        edges.append({
            "source": source,
            "target": target,
            "label": rel_type,
        })

    for rel in outgoing:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue

        source, rel_type, target = rel
        source = str(source)
        rel_type = str(rel_type)
        target = str(target)

        nodes.add(source)
        nodes.add(target)

        edges.append({
            "source": source,
            "target": target,
            "label": rel_type,
        })

    return {
        "nodes": sorted(nodes),
        "edges": edges,
    }


def build_mindmap_from_entity(entity: str, graph_store) -> Dict[str, Any]:
    """
    Query Dat's graph using a single entity and convert the result
    into the UI-ready graph format.
    """
    query_result = query(entity, graph_store)
    return dat_query_result_to_mindmap_data(query_result)

from typing import Dict, List, Optional
from src.utils.graph_helpers import (
    normalize_text,
    safe_label,
    dedupe_edges,
    dedupe_nodes,
    build_degree_map,
)


ALLOWED_LABEL_MAP = {
    "contains": "contains",
    "include": "contains",
    "includes": "contains",
    "part of": "part of",
    "defines": "defines",
    "definition of": "defines",
    "example of": "example of",
    "depends on": "depends on",
    "used for": "used for",
    "applies to": "applies to",
    "related to": "related to",
    "connected to": "related to",
    "is a": "is a",
    "type of": "is a",
}


def normalize_relation_label(label: str) -> str:
    label = safe_label(label).lower()
    return ALLOWED_LABEL_MAP.get(label, label)


def _extract_edges_from_relationships(raw_relationships: List[dict]) -> List[dict]:
    edges = []

    for rel in raw_relationships:
        source = normalize_text(
            rel.get("source")
            or rel.get("from")
            or rel.get("start")
            or rel.get("subject")
        )
        target = normalize_text(
            rel.get("target")
            or rel.get("to")
            or rel.get("end")
            or rel.get("object")
        )
        label = normalize_relation_label(
            rel.get("label")
            or rel.get("relation")
            or rel.get("type")
            or "related to"
        )

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "label": label
            })

    return edges


def _extract_edges_from_incoming_outgoing(query_result: dict) -> List[dict]:
    edges = []

    outgoing = query_result.get("outgoing", [])
    incoming = query_result.get("incoming", [])

    for rel in outgoing:
        source = normalize_text(
            rel.get("source") or rel.get("from") or query_result.get("center")
        )
        target = normalize_text(rel.get("target") or rel.get("to"))
        label = normalize_relation_label(rel.get("label") or rel.get("type") or "related to")

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "label": label
            })

    for rel in incoming:
        source = normalize_text(rel.get("source") or rel.get("from"))
        target = normalize_text(
            rel.get("target") or rel.get("to") or query_result.get("center")
        )
        label = normalize_relation_label(rel.get("label") or rel.get("type") or "related to")

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "label": label
            })

    return edges


def _extract_edges(query_result: dict) -> List[dict]:
    if "relationships" in query_result:
        return _extract_edges_from_relationships(query_result["relationships"])

    if "edges" in query_result:
        return _extract_edges_from_relationships(query_result["edges"])

    if "incoming" in query_result or "outgoing" in query_result:
        return _extract_edges_from_incoming_outgoing(query_result)

    return []


def dat_result_to_mindmap_data(
    query_result: dict,
    center_topic: Optional[str] = None,
    max_neighbors: int = 8,
    prefer_outgoing: bool = True,
) -> Dict:
    """
    Convert Dat's graph query result into the app's standard format:
    {
      "nodes": [...],
      "edges": [{"source": ..., "target": ..., "label": ...}],
      "center": "..."
    }
    """
    raw_edges = _extract_edges(query_result)
    raw_edges = dedupe_edges(raw_edges)

    if not raw_edges:
        return {
            "nodes": [],
            "edges": [],
            "center": center_topic or query_result.get("center") or ""
        }

    nodes = []
    for edge in raw_edges:
        nodes.append(edge["source"])
        nodes.append(edge["target"])

    nodes = dedupe_nodes(nodes)

    resolved_center = normalize_text(
        center_topic
        or query_result.get("center")
        or _choose_center_by_degree(raw_edges)
    )

    pruned_edges = prune_graph(
        edges=raw_edges,
        center_topic=resolved_center,
        max_neighbors=max_neighbors,
        prefer_outgoing=prefer_outgoing,
    )

    pruned_nodes = []
    for edge in pruned_edges:
        pruned_nodes.append(edge["source"])
        pruned_nodes.append(edge["target"])

    pruned_nodes = dedupe_nodes(pruned_nodes)

    return {
        "nodes": pruned_nodes,
        "edges": pruned_edges,
        "center": resolved_center,
    }


def _choose_center_by_degree(edges: List[dict]) -> str:
    degree_map = build_degree_map(edges)
    if not degree_map:
        return ""
    return max(degree_map, key=degree_map.get)


def prune_graph(
    edges: List[dict],
    center_topic: str,
    max_neighbors: int = 8,
    prefer_outgoing: bool = True,
) -> List[dict]:
    """
    Keep graph readable:
    - prioritize edges connected to the center
    - keep max N first-level neighbors
    - keep useful relation labels
    """
    center_topic = normalize_text(center_topic)
    if not center_topic:
        return edges[:max_neighbors]

    outgoing = []
    incoming = []
    unrelated = []

    for edge in edges:
        s = normalize_text(edge["source"])
        t = normalize_text(edge["target"])

        if s == center_topic:
            outgoing.append(edge)
        elif t == center_topic:
            incoming.append(edge)
        else:
            unrelated.append(edge)

    if prefer_outgoing:
        first_priority = outgoing + incoming
    else:
        first_priority = incoming + outgoing

    kept_first_level = first_priority[:max_neighbors]
    allowed_nodes = {center_topic}

    for edge in kept_first_level:
        allowed_nodes.add(edge["source"])
        allowed_nodes.add(edge["target"])

    kept_second_level = []
    for edge in unrelated:
        s = normalize_text(edge["source"])
        t = normalize_text(edge["target"])

        if s in allowed_nodes or t in allowed_nodes:
            kept_second_level.append(edge)

    kept_second_level = kept_second_level[: max(4, max_neighbors // 2)]

    final_edges = dedupe_edges(kept_first_level + kept_second_level)
    return final_edges