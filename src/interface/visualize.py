import os
import tempfile

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from pyvis.network import Network


def get_sample_graph_data() -> dict:
    return {
        "nodes": [
            "Machine Learning",
            "Supervised Learning",
            "Unsupervised Learning",
            "Classification",
            "Regression",
            "Clustering",
            "Dimensionality Reduction",
        ],
        "edges": [
            {"source": "Machine Learning", "target": "Supervised Learning", "label": "contains"},
            {"source": "Machine Learning", "target": "Unsupervised Learning", "label": "contains"},
            {"source": "Supervised Learning", "target": "Classification", "label": "type"},
            {"source": "Supervised Learning", "target": "Regression", "label": "type"},
            {"source": "Unsupervised Learning", "target": "Clustering", "label": "type"},
            {"source": "Unsupervised Learning", "target": "Dimensionality Reduction", "label": "type"},
        ],
    }


def build_graph_from_nodes_edges(data: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    if not isinstance(data, dict):
        return G

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("id") or node.get("label")
            if node_id is not None:
                G.add_node(
                    str(node_id),
                    label=str(node.get("label", node_id)),
                    title=str(node.get("title", node.get("label", node_id))),
                )
        else:
            G.add_node(str(node), label=str(node), title=str(node))

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label", "")

        if source is not None and target is not None:
            G.add_edge(str(source), str(target), label=str(label))

    return G


def _build_pyvis_html(G: nx.DiGraph) -> str:
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
        directed=True,
    )

    net.force_atlas_2based()

    for node, attrs in G.nodes(data=True):
        degree = G.degree[node]
        size = 18 + degree * 4
        net.add_node(
            node,
            label=attrs.get("label", str(node)),
            title=attrs.get("title", str(node)),
            size=size,
        )

    for source, target, edge_data in G.edges(data=True):
        net.add_edge(source, target, label=edge_data.get("label", ""))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp_path = tmp.name

    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return html


def _show_graph_matplotlib(G: nx.DiGraph) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        ax=ax,
        node_size=2200,
        font_size=10,
        arrows=True,
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    st.pyplot(fig)
    plt.close(fig)


def show_graph(G: nx.DiGraph, prefer_pyvis: bool = True) -> None:
    if G.number_of_nodes() == 0:
        st.warning("Graph is empty.")
        return

    if prefer_pyvis:
        try:
            html = _build_pyvis_html(G)
            st.components.v1.html(html, height=650, scrolling=True)
            return
        except Exception as e:
            st.warning(f"PyVis failed, using fallback renderer instead. Error: {e}")

    _show_graph_matplotlib(G)


def render_mindmap_ui(
    data: dict | None = None,
    show_debug: bool = False,
    prefer_pyvis: bool = True,
) -> None:
    st.subheader("Mindmap Visualization")

    if data is None:
        data = get_sample_graph_data()

    if show_debug:
        st.write("Incoming data:")
        st.json(data)

    G = build_graph_from_nodes_edges(data)

    if show_debug:
        st.write(f"Nodes in graph: {G.number_of_nodes()}")
        st.write(f"Edges in graph: {G.number_of_edges()}")
        st.write("Graph nodes:", list(G.nodes(data=True)))
        st.write("Graph edges:", list(G.edges(data=True)))

    if G.number_of_nodes() == 0:
        st.warning("No nodes available to render.")
        return

    show_graph(G, prefer_pyvis=prefer_pyvis)