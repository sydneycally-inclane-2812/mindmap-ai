from dataclasses import dataclass, field
from typing import Any, Dict, Optional


APP_STATE_IDLE = "idle"
APP_STATE_FILES_UPLOADED = "files_uploaded"
APP_STATE_PROCESSING = "processing"
APP_STATE_GRAPH_READY = "graph_ready"
APP_STATE_GRAPH_EMPTY = "graph_empty"
APP_STATE_RENDER_FAILED = "render_failed"


@dataclass
class MindMapAppState:
    status: str = APP_STATE_IDLE
    uploaded_file_name: Optional[str] = None
    extracted_text: str = ""
    raw_result: Optional[Dict[str, Any]] = None
    adapted_data: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    study_outputs: Optional[Dict[str, Any]] = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "uploaded_file_name": self.uploaded_file_name,
            "extracted_text": self.extracted_text,
            "raw_result": self.raw_result,
            "adapted_data": self.adapted_data,
            "structured_data": self.structured_data,
            "study_outputs": self.study_outputs,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


def create_default_state() -> MindMapAppState:
    return MindMapAppState()


def set_status(state: MindMapAppState, status: str) -> MindMapAppState:
    state.status = status
    return state


def reset_processing_outputs(state: MindMapAppState) -> MindMapAppState:
    state.raw_result = None
    state.adapted_data = None
    state.structured_data = None
    state.study_outputs = None
    state.error_message = ""
    return state


def has_graph(state: MindMapAppState) -> bool:
    data = state.structured_data or {}
    return bool(data.get("nodes")) and bool(data.get("edges"))

# src/interface/styles.py

import streamlit as st


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 1.75rem;
            padding-bottom: 2rem;
        }

        .mm-header {
            border: 1px solid #e5e7eb;
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
        }

        .mm-title {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.15;
            color: #111827;
            margin-bottom: 0.35rem;
        }

        .mm-subtitle {
            font-size: 1rem;
            line-height: 1.5;
            color: #4b5563;
            margin-bottom: 0.4rem;
        }

        .mm-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.25rem;
        }

        .mm-chip {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #dbeafe;
            color: #374151;
            font-size: 0.88rem;
            font-weight: 500;
        }

        .mm-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }

        .mm-section-title {
            font-size: 1.08rem;
            font-weight: 650;
            color: #111827;
            margin-bottom: 0.75rem;
        }

        .mm-muted {
            color: #6b7280;
            font-size: 0.94rem;
            line-height: 1.45;
        }

        .mm-status {
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            padding: 0.9rem 1rem;
            color: #374151;
            margin-bottom: 0.2rem;
        }

        .mm-upload-note {
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            background: #f8fafc;
            margin-top: 0.8rem;
            color: #475569;
            font-size: 0.92rem;
        }

        .mm-results-note {
            font-size: 0.92rem;
            color: #6b7280;
            margin-top: -0.2rem;
            margin-bottom: 0.6rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            font-weight: 650;
            padding: 0.72rem 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stTextInput > div > div,
        .stTextArea textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            border-radius: 14px;
        }

        .stFileUploader {
            border-radius: 16px;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        hr {
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )