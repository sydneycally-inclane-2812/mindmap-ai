"""DuckDB-backed evidence storage for relation provenance."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)


class EvidenceDuckDBStore:
    """Persist ingestion artifacts and relation evidence to DuckDB."""

    def __init__(self, db_path: str = "data/evidence.duckdb"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create MVP schema aligned with the project README."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT,
                title TEXT,
                source_path TEXT,
                doc_type TEXT,
                checksum TEXT
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT,
                document_id TEXT,
                chunk_text TEXT,
                chunk_summary TEXT,
                section_title TEXT,
                chunk_order INTEGER,
                token_count INTEGER
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_units (
                evidence_unit_id TEXT,
                chunk_id TEXT,
                unit_type TEXT,
                unit_text TEXT,
                unit_order INTEGER,
                start_char INTEGER,
                end_char INTEGER
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_registry (
                canonical_name TEXT PRIMARY KEY,
                aliases_json TEXT
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT,
                source_canonical_name TEXT,
                relation_type TEXT,
                target_canonical_name TEXT,
                confidence DOUBLE,
                is_derived BOOLEAN
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_evidence (
                relation_id TEXT,
                evidence_unit_id TEXT,
                support_score DOUBLE,
                evidence_text TEXT
            )
            """
        )

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _find_chunk_id_for_evidence(chunks: List[Dict[str, Any]], evidence_text: str) -> Optional[str]:
        """Best-effort mapping from evidence snippet to a chunk ID."""
        if not chunks:
            return None
        if not evidence_text:
            return chunks[0]["chunk_id"]

        snippet = evidence_text.strip().lower()
        for chunk in chunks:
            chunk_text = (chunk.get("chunk_text") or "").lower()
            if snippet and snippet in chunk_text:
                return chunk["chunk_id"]
        return chunks[0]["chunk_id"]

    def upsert_entity_registry(self, entities: List[str]) -> None:
        for entity in sorted(set(entities)):
            self.conn.execute(
                """
                INSERT INTO entity_registry (canonical_name, aliases_json)
                VALUES (?, ?)
                ON CONFLICT(canonical_name) DO NOTHING
                """,
                [entity, json.dumps([entity])],
            )

    def store_ingestion(
        self,
        documents: List[str],
        chunks: List[Dict[str, Any]],
        entities: List[str],
        relation_records: List[Dict[str, Any]],
    ) -> None:
        """Persist documents, chunks, entities, and relation evidence."""
        for doc_idx, _doc_text in enumerate(documents):
            document_id = f"doc_{doc_idx}"
            self.conn.execute(
                """
                INSERT INTO documents (document_id, title, source_path, doc_type, checksum)
                VALUES (?, ?, ?, ?, ?)
                """,
                [document_id, f"Document {doc_idx}", None, "txt", None],
            )

        for chunk in chunks:
            self.conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, chunk_text, chunk_summary, section_title, chunk_order, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["chunk_text"],
                    None,
                    None,
                    chunk["chunk_order"],
                    len((chunk["chunk_text"] or "").split()),
                ],
            )

        self.upsert_entity_registry(entities)

        for rel in relation_records:
            relation_id = f"rel_{uuid.uuid4().hex}"
            source = rel.get("source")
            rel_type = rel.get("relation")
            target = rel.get("target")
            confidence = float(rel.get("confidence", 0.0) or 0.0)
            evidence_text = (rel.get("evidence") or "").strip()

            self.conn.execute(
                """
                INSERT INTO relations (relation_id, source_canonical_name, relation_type, target_canonical_name, confidence, is_derived)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [relation_id, source, rel_type, target, confidence, False],
            )

            evidence_unit_id = f"ev_{uuid.uuid4().hex}"
            chunk_id = self._find_chunk_id_for_evidence(chunks, evidence_text)
            self.conn.execute(
                """
                INSERT INTO evidence_units (evidence_unit_id, chunk_id, unit_type, unit_text, unit_order, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [evidence_unit_id, chunk_id, "llm_snippet", evidence_text, 0, None, None],
            )

            self.conn.execute(
                """
                INSERT INTO relation_evidence (relation_id, evidence_unit_id, support_score, evidence_text)
                VALUES (?, ?, ?, ?)
                """,
                [relation_id, evidence_unit_id, confidence, evidence_text],
            )

        logger.info(
            "Persisted DuckDB evidence: %s documents, %s chunks, %s entities, %s relations",
            len(documents),
            len(chunks),
            len(entities),
            len(relation_records),
        )
