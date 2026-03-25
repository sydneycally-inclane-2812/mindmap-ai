from collections import defaultdict
from typing import Dict, List
from src.utils.graph_helpers import normalize_text, dedupe_edges, dedupe_nodes


MAIN_BRANCH_LIMIT = 6
CHILDREN_PER_BRANCH_LIMIT = 3


def structure_as_study_mindmap(data: Dict) -> Dict:
    """
    Convert a pruned graph into a cleaner study-style mind map.

    Input:
    {
        "nodes": [...],
        "edges": [{"source": ..., "target": ..., "label": ...}],
        "center": "..."
    }

    Output uses same format, but cleaner and more hierarchical.
    """
    center = normalize_text(data.get("center"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not center or not edges:
        return {
            "nodes": dedupe_nodes(nodes),
            "edges": dedupe_edges(edges),
            "center": center
        }

    first_level = []
    second_level_candidates = defaultdict(list)

    for edge in edges:
        source = normalize_text(edge["source"])
        target = normalize_text(edge["target"])
        label = normalize_text(edge.get("label", "related to"))

        if source == center:
            first_level.append({
                "branch": target,
                "label": label
            })
        elif target == center:
            first_level.append({
                "branch": source,
                "label": label
            })

    # Remove duplicates while preserving order
    seen = set()
    unique_first_level = []
    for item in first_level:
        branch = item["branch"]
        if branch.lower() not in seen:
            seen.add(branch.lower())
            unique_first_level.append(item)

    unique_first_level = unique_first_level[:MAIN_BRANCH_LIMIT]
    branch_names = {item["branch"] for item in unique_first_level}

    # Find second-level children for each branch
    for edge in edges:
        source = normalize_text(edge["source"])
        target = normalize_text(edge["target"])
        label = normalize_text(edge.get("label", "related to"))

        for branch in branch_names:
            if source == branch and target != center:
                second_level_candidates[branch].append({
                    "child": target,
                    "label": label
                })
            elif target == branch and source != center:
                second_level_candidates[branch].append({
                    "child": source,
                    "label": label
                })

    structured_edges = []
    structured_nodes = [center]

    # Center -> main branches
    for item in unique_first_level:
        branch = item["branch"]
        label = item["label"] or "contains"

        structured_nodes.append(branch)
        structured_edges.append({
            "source": center,
            "target": branch,
            "label": label
        })

    # Main branches -> children
    for branch, children in second_level_candidates.items():
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
                "label": label
            })
            kept += 1

            if kept >= CHILDREN_PER_BRANCH_LIMIT:
                break

    return {
        "nodes": dedupe_nodes(structured_nodes),
        "edges": dedupe_edges(structured_edges),
        "center": center
    }