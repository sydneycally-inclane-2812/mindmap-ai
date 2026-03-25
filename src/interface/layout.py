import streamlit as st
from typing import Dict, List, Optional, Tuple


def render_header() -> None:
    st.markdown(
        """
        <div class="mm-header">
            <div class="mm-title">MindMap AI</div>
            <div class="mm-subtitle">
                Upload your notes, slides, or readings and turn them into a cleaner study mind map.
            </div>
            <div class="mm-chip-row">
                <span class="mm-chip">Upload</span>
                <span class="mm-chip">Build graph</span>
                <span class="mm-chip">Study summary</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_card(title: str) -> None:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mm-section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_box(message: str) -> None:
    st.markdown(
        f'<div class="mm-status">{message}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_settings() -> Dict[str, object]:
    with st.sidebar:
        st.header("Settings")

        show_debug = st.checkbox("Show debug output", value=False)
        prefer_pyvis = st.checkbox("Prefer interactive renderer", value=True)
        max_neighbors = st.slider("Max first-level neighbors", 3, 12, 8)
        prefer_outgoing = st.checkbox("Prefer outgoing edges", value=True)
        show_source_preview = st.checkbox("Show source preview", value=True)

        st.divider()
        st.caption("Use one topic at a time for the cleanest study mind map.")

    return {
        "show_debug": show_debug,
        "prefer_pyvis": prefer_pyvis,
        "max_neighbors": max_neighbors,
        "prefer_outgoing": prefer_outgoing,
        "show_source_preview": show_source_preview,
    }


def render_upload_section() -> Optional[List[object]]:
    open_card("Upload documents")

    st.markdown(
        """
        <div class="mm-muted">
            Add one or more study files. The app will extract the content and build a graph from it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, PPTX, TXT",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    st.markdown(
        """
        <div class="mm-upload-note">
            Best results usually come from one focused topic or one lecture deck at a time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    close_card()
    return uploaded_files


def render_controls_section(default_topic: str = "") -> Tuple[str, bool]:
    open_card("Build mind map")

    st.markdown(
        """
        <div class="mm-muted">
            Choose the central topic you want the mind map to revolve around.
        </div>
        """,
        unsafe_allow_html=True,
    )

    central_topic = st.text_input(
        "Central topic",
        value=default_topic,
        placeholder="e.g. Machine Learning, Marketing, Supply Chain",
    )

    generate_clicked = st.button(
        "Generate Study Mind Map",
        type="primary",
        use_container_width=True,
    )

    close_card()
    return central_topic, generate_clicked


def render_uploaded_files_summary(file_names: List[str]) -> None:
    open_card("Uploaded files")

    if not file_names:
        st.write("No files uploaded yet.")
    else:
        st.markdown(
            f'<div class="mm-results-note">{len(file_names)} file(s) ready for processing.</div>',
            unsafe_allow_html=True,
        )
        for name in file_names:
            st.markdown(f"- {name}")

    close_card()


def render_empty_state() -> None:
    st.info("Upload one or more files, enter a central topic, and generate the mind map.")


def render_files_uploaded_state(file_name: str) -> None:
    st.info(f"Files loaded: {file_name}. Enter the topic and click the main action button.")


def render_processing_state() -> None:
    st.info("Processing documents and building the study mind map...")


def render_graph_empty_state() -> None:
    st.warning("The graph was built, but there was not enough useful structure to render a readable mind map.")


def render_render_failed_state(error_message: str) -> None:
    if error_message:
        st.error(f"Something failed while generating or rendering the mind map: {error_message}")
    else:
        st.error("Something failed while generating or rendering the mind map.")


def render_results_intro() -> None:
    st.markdown(
        """
        <div class="mm-results-note">
            Explore the visual mind map, review the study summary, and inspect extracted source text.
        </div>
        """,
        unsafe_allow_html=True,
    )