from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip()


def safe_label(label: str) -> str:
    label = normalize_text(label)
    if not label:
        return "related to"
    return label


def dedupe_edges(edges: List[dict]) -> List[dict]:
    seen = set()
    unique = []

    for edge in edges:
        source = normalize_text(edge.get("source"))
        target = normalize_text(edge.get("target"))
        label = safe_label(edge.get("label"))

        if not source or not target or source == target:
            continue

        key = (source.lower(), target.lower(), label.lower())
        if key not in seen:
            seen.add(key)
            unique.append({
                "source": source,
                "target": target,
                "label": label
            })

    return unique


def dedupe_nodes(nodes: List[str]) -> List[str]:
    seen = set()
    unique = []

    for node in nodes:
        n = normalize_text(node)
        if not n:
            continue
        key = n.lower()
        if key not in seen:
            seen.add(key)
            unique.append(n)

    return unique


def build_degree_map(edges: List[dict]) -> Dict[str, int]:
    degree = Counter()

    for edge in edges:
        s = normalize_text(edge.get("source"))
        t = normalize_text(edge.get("target"))
        if s:
            degree[s] += 1
        if t:
            degree[t] += 1

    return dict(degree)


def build_adjacency(edges: List[dict]) -> Dict[str, List[dict]]:
    adj = defaultdict(list)

    for edge in edges:
        source = normalize_text(edge.get("source"))
        target = normalize_text(edge.get("target"))
        label = safe_label(edge.get("label"))

        adj[source].append({
            "node": target,
            "label": label,
            "direction": "out"
        })

        adj[target].append({
            "node": source,
            "label": label,
            "direction": "in"
        })

    return dict(adj)


def count_relation_frequency(edges: List[dict]) -> Dict[Tuple[str, str], int]:
    freq = Counter()
    for edge in edges:
        source = normalize_text(edge.get("source"))
        target = normalize_text(edge.get("target"))
        if source and target:
            freq[(source, target)] += 1
    return dict(freq)