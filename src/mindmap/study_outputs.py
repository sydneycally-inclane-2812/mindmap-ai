from collections import defaultdict
from typing import Dict, List
from src.utils.graph_helpers import normalize_text


def generate_study_outputs(mindmap_data: Dict) -> Dict:
    center = normalize_text(mindmap_data.get("center"))
    edges = mindmap_data.get("edges", [])

    branches = []
    children_by_branch = defaultdict(list)

    for edge in edges:
        source = normalize_text(edge["source"])
        target = normalize_text(edge["target"])
        label = normalize_text(edge.get("label", "related to"))

        if source == center:
            branches.append((target, label))
        else:
            children_by_branch[source].append((target, label))

    branch_names = [b[0] for b in branches]

    key_topics = branch_names

    revision_bullets = []
    for branch in branch_names:
        children = children_by_branch.get(branch, [])
        if children:
            child_names = ", ".join([c[0] for c in children[:3]])
            revision_bullets.append(f"{branch}: {child_names}")
        else:
            revision_bullets.append(f"{branch}: main concept branch")

    summary = build_summary(center, branch_names, children_by_branch)

    important_concepts = []
    for branch in branch_names:
        important_concepts.append(branch)
        for child, _ in children_by_branch.get(branch, [])[:2]:
            important_concepts.append(child)

    # remove duplicates preserving order
    seen = set()
    unique_concepts = []
    for concept in important_concepts:
        c = concept.lower()
        if c not in seen:
            seen.add(c)
            unique_concepts.append(concept)

    return {
        "key_topics": key_topics,
        "summary": summary,
        "important_concepts": unique_concepts[:12],
        "revision_bullets": revision_bullets
    }


def build_summary(center: str, branch_names: List[str], children_by_branch: Dict[str, List]) -> str:
    if not center:
        return "No clear central topic was identified."

    if not branch_names:
        return f"The main topic appears to be {center}, but the graph does not yet contain enough structured branches."

    summary_parts = [f"This mind map is centered on {center}."]
    summary_parts.append(
        f"The main branches are {', '.join(branch_names[:6])}."
    )

    detailed_parts = []
    for branch in branch_names[:3]:
        children = children_by_branch.get(branch, [])
        if children:
            child_names = ", ".join([c[0] for c in children[:3]])
            detailed_parts.append(f"{branch} connects to {child_names}")

    if detailed_parts:
        summary_parts.append("Key supporting links include " + "; ".join(detailed_parts) + ".")

    return " ".join(summary_parts)