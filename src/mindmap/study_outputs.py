from collections import defaultdict
from typing import Dict, List
from src.utils.graph_helpers import normalize_text, dedupe_edges, dedupe_nodes


MAIN_BRANCH_LIMIT = 6
CHILDREN_PER_BRANCH_LIMIT = 3

LABEL_RANK = {
    "defines": 0,
    "contains": 1,
    "part of": 2,
    "type of": 3,
    "example of": 4,
    "depends on": 5,
    "formula for": 6,
    "related to": 7,
}


def rank_label(label: str) -> int:
    return LABEL_RANK.get(normalize_text(label).lower(), 99)


def structure_as_study_mindmap(data: Dict) -> Dict:
    center = normalize_text(data.get("center"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not center or not edges:
        return {
            "nodes": dedupe_nodes(nodes),
            "edges": dedupe_edges(edges),
            "center": center,
        }

    first_level = []
    second_level_candidates = defaultdict(list)

    for edge in edges:
        source = normalize_text(edge["source"])
        target = normalize_text(edge["target"])
        label = normalize_text(edge.get("label", "related to"))

        if source == center:
            first_level.append({"branch": target, "label": label})
        elif target == center:
            first_level.append({"branch": source, "label": label})

    first_level = sorted(first_level, key=lambda x: rank_label(x["label"]))

    seen = set()
    unique_first_level = []
    for item in first_level:
        branch = item["branch"]
        if branch.lower() not in seen:
            seen.add(branch.lower())
            unique_first_level.append(item)

    unique_first_level = unique_first_level[:MAIN_BRANCH_LIMIT]
    branch_names = {item["branch"] for item in unique_first_level}

    for edge in edges:
        source = normalize_text(edge["source"])
        target = normalize_text(edge["target"])
        label = normalize_text(edge.get("label", "related to"))

        for branch in branch_names:
            if source == branch and target != center:
                second_level_candidates[branch].append({"child": target, "label": label})
            elif target == branch and source != center:
                second_level_candidates[branch].append({"child": source, "label": label})

    structured_edges = []
    structured_nodes = [center]

    for item in unique_first_level:
        branch = item["branch"]
        label = item["label"] or "contains"

        structured_nodes.append(branch)
        structured_edges.append({
            "source": center,
            "target": branch,
            "label": label,
        })

    for branch, children in second_level_candidates.items():
        children = sorted(children, key=lambda x: rank_label(x["label"]))

        seen_children = set()
        kept = 0

        for child_item in children:
            child = child_item["child"]
            label = child_item["label"] or "contains"

            if child.lower() in seen_children:
                continue
            if child.lower() == center.lower():
                continue
            if child.lower() == branch.lower():
                continue

            seen_children.add(child.lower())
            structured_nodes.append(child)
            structured_edges.append({
                "source": branch,
                "target": child,
                "label": label,
            })
            kept += 1

            if kept >= CHILDREN_PER_BRANCH_LIMIT:
                break

    return {
        "nodes": dedupe_nodes(structured_nodes),
        "edges": dedupe_edges(structured_edges),
        "center": center,
    }