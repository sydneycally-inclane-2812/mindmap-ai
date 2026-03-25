from typing import Dict, List, Optional
from src.utils.graph_helpers import (
    normalize_text,
    safe_label,
    dedupe_edges,
    dedupe_nodes,
    build_degree_map,
)


RELATION_CANONICAL_MAP = {
    "contains": "contains",
    "include": "contains",
    "includes": "contains",
    "has": "contains",
    "covers": "contains",

    "defines": "defines",
    "definition of": "defines",
    "means": "defines",
    "describes": "defines",
    "explains": "defines",

    "example of": "example of",
    "instance of": "example of",
    "sample of": "example of",
    "illustrates": "example of",

    "depends on": "depends on",
    "requires": "depends on",
    "needs": "depends on",
    "uses": "depends on",
    "built on": "depends on",

    "formula for": "formula for",
    "equation for": "formula for",
    "calculates": "formula for",

    "part of": "part of",
    "belongs to": "part of",
    "component of": "part of",

    "type of": "type of",
    "is a": "type of",

    "related to": "related to",
    "connected to": "related to",
    "associated with": "related to",
}


PREFERRED_LABEL_ORDER = [
    "defines",
    "contains",
    "part of",
    "type of",
    "example of",
    "depends on",
    "formula for",
    "related to",
]


def normalize_relation_label(label: str) -> str:
    raw = safe_label(label).lower().strip()
    return RELATION_CANONICAL_MAP.get(raw, "related to")


def label_priority(label: str) -> int:
    label = normalize_relation_label(label)
    if label in PREFERRED_LABEL_ORDER:
        return PREFERRED_LABEL_ORDER.index(label)
    return len(PREFERRED_LABEL_ORDER)


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
                "label": label,
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
        label = normalize_relation_label(
            rel.get("label") or rel.get("type") or "related to"
        )

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "label": label,
            })

    for rel in incoming:
        source = normalize_text(rel.get("source") or rel.get("from"))
        target = normalize_text(
            rel.get("target") or rel.get("to") or query_result.get("center")
        )
        label = normalize_relation_label(
            rel.get("label") or rel.get("type") or "related to"
        )

        if source and target and source != target:
            edges.append({
                "source": source,
                "target": target,
                "label": label,
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

    outgoing = sorted(outgoing, key=lambda e: label_priority(e.get("label", "")))
    incoming = sorted(incoming, key=lambda e: label_priority(e.get("label", "")))
    unrelated = sorted(unrelated, key=lambda e: label_priority(e.get("label", "")))

    first_priority = outgoing + incoming if prefer_outgoing else incoming + outgoing
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
    return dedupe_edges(kept_first_level + kept_second_level)


def dat_result_to_mindmap_data(
    query_result: dict,
    center_topic: Optional[str] = None,
    max_neighbors: int = 8,
    prefer_outgoing: bool = True,
) -> Dict:
    raw_edges = _extract_edges(query_result)
    raw_edges = dedupe_edges(raw_edges)

    if not raw_edges:
        return {
            "nodes": [],
            "edges": [],
            "center": center_topic or query_result.get("center") or "",
        }

    nodes = []
    for edge in raw_edges:
        nodes.append(edge["source"])
        nodes.append(edge["target"])

    nodes = dedupe_nodes(nodes)

    resolved_center = normalize_text(
        center_topic or query_result.get("center") or _choose_center_by_degree(raw_edges)
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