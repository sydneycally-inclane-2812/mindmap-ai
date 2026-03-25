import streamlit as st

from visualize import get_sample_graph_data, render_mindmap_uicd

st.set_page_config(page_title="Mindmap Visualizer Example", layout="wide")

st.title("Mindmap Visualizer Example")

use_pyvis = st.checkbox("Use PyVis renderer", value=True)
show_debug = st.checkbox("Show debug info", value=False)

sample_data = get_sample_graph_data()

render_mindmap_ui(
    data=sample_data,
    show_debug=show_debug,
    prefer_pyvis=use_pyvis,
)