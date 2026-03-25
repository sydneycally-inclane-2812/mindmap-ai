import streamlit as st
from typing import Optional, Tuple


def render_header():
    st.markdown(
        """
        <div class="mm-header">
            <div class="mm-title">MindMap AI</div>
            <div class="mm-subtitle">
                Upload study material, extract concepts, and turn them into a clean study mind map.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_card(title: str):
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-section-title">{title}</div>', unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_box(message: str):
    st.markdown(
        f'<div class="mm-status">{message}</div>',
        unsafe_allow_html=True,
    )


def render_upload_section() -> Optional[object]:
    open_card("Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF or text-based study document",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )
    st.markdown(
        '<div class="mm-muted">Supported for now: PDF, TXT, MD. Start with one file for a cleaner graph.</div>',
        unsafe_allow_html=True,
    )
    close_card()
    return uploaded_file


def render_controls_section(default_topic: str = "") -> Tuple[str, bool]:
    open_card("Controls")
    central_topic = st.text_input(
        "Central topic",
        value=default_topic,
        placeholder="Enter the topic to center the mind map around",
    )
    generate_clicked = st.button("Generate Study Mind Map", type="primary", use_container_width=True)
    close_card()
    return central_topic, generate_clicked


def render_sidebar_settings():
    with st.sidebar:
        st.header("Settings")
        show_debug = st.checkbox("Show debug output", value=False)
        prefer_pyvis = st.checkbox("Prefer pyvis renderer", value=True)
        max_neighbors = st.slider("Max first-level neighbors", 3, 12, 8)
        prefer_outgoing = st.checkbox("Prefer outgoing edges", value=True)
        show_source_preview = st.checkbox("Show source preview section", value=True)

    return {
        "show_debug": show_debug,
        "prefer_pyvis": prefer_pyvis,
        "max_neighbors": max_neighbors,
        "prefer_outgoing": prefer_outgoing,
        "show_source_preview": show_source_preview,
    }


def render_empty_state():
    st.info("Upload a file, choose a central topic, and generate the mind map.")


def render_files_uploaded_state(file_name: str):
    st.info(f"File uploaded: {file_name}. Set the topic and click generate.")


def render_graph_empty_state():
    st.warning("The graph was processed, but no useful mind map could be built from this content.")


def render_render_failed_state(error_message: str):
    st.error(f"Graph rendering failed. {error_message}")


def render_processing_state():
    st.info("Processing your file and building the mind map...")