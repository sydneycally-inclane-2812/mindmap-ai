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