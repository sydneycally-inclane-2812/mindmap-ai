import streamlit as st
from pyvis.network import Network
import tempfile, os

def show_graph(G):
    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")
    net.force_atlas_2based()

    for node in G.nodes():
        net.add_node(node, label=node, size=15 + G.degree[node]*3)

    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, label=data.get("label",""))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    try:
        net.save_graph(tmp.name)
        tmp.close()  # Ensure file is closed before reading/unlinking
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=600)
    finally:
        os.unlink(tmp.name)