from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from .types import ToolCandidate
from .util import cosine_similarity


def _encode_vec(vec: list[float]) -> bytes:
    # store as JSON for portability/simplicity
    return json.dumps(vec, separators=(",", ":")).encode("utf-8")


def _decode_vec(blob: bytes) -> list[float]:
    return json.loads(blob.decode("utf-8"))


@dataclass
class SQLiteStore:
    db_path: str = "tool_router.sqlite3"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tools (
                  tool_id TEXT PRIMARY KEY,
                  server TEXT NOT NULL,
                  name TEXT NOT NULL,
                  version TEXT,
                  content_hash TEXT NOT NULL,
                  text_repr TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  embedding BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tools_server ON tools(server);

                CREATE TABLE IF NOT EXISTS resources (
                  resource_id TEXT PRIMARY KEY,
                  server TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  name TEXT NOT NULL,
                  version TEXT,
                  content_hash TEXT NOT NULL,
                  descriptor_json TEXT NOT NULL,
                  resource_content_hash TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_resources_server ON resources(server);

                CREATE TABLE IF NOT EXISTS resource_chunks (
                  chunk_id TEXT PRIMARY KEY,
                  resource_id TEXT NOT NULL REFERENCES resources(resource_id) ON DELETE CASCADE,
                  server TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  content_hash TEXT NOT NULL,
                  text TEXT NOT NULL,
                  embedding BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_server ON resource_chunks(server);
                CREATE INDEX IF NOT EXISTS idx_chunks_resource ON resource_chunks(resource_id);
                """
            )
            # Lightweight migration: older DBs won't have resource_content_hash.
            try:
                conn.execute("SELECT resource_content_hash FROM resources LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE resources ADD COLUMN resource_content_hash TEXT")

    # --------------------
    # Tools
    # --------------------
    def get_tool_hash(self, tool_id: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM tools WHERE tool_id = ?",
                (tool_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def upsert_tool(
        self,
        *,
        tool_id: str,
        server: str,
        name: str,
        version: Optional[str],
        content_hash: str,
        text_repr: str,
        payload_json: str,
        embedding: list[float],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tools(tool_id, server, name, version, content_hash, text_repr, payload_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                  server=excluded.server,
                  name=excluded.name,
                  version=excluded.version,
                  content_hash=excluded.content_hash,
                  text_repr=excluded.text_repr,
                  payload_json=excluded.payload_json,
                  embedding=excluded.embedding
                """,
                (
                    tool_id,
                    server,
                    name,
                    version,
                    content_hash,
                    text_repr,
                    payload_json,
                    _encode_vec(embedding),
                ),
            )

    def delete_tools_not_in(self, tool_ids: set[str]) -> int:
        if not tool_ids:
            with self.connect() as conn:
                cur = conn.execute("DELETE FROM tools")
                return int(cur.rowcount)
        placeholders = ",".join("?" for _ in tool_ids)
        with self.connect() as conn:
            cur = conn.execute(
                f"DELETE FROM tools WHERE tool_id NOT IN ({placeholders})",
                tuple(tool_ids),
            )
            return int(cur.rowcount)

    def query_tools(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        server_filter: Optional[str] = None,
    ) -> list[ToolCandidate]:
        with self.connect() as conn:
            if server_filter:
                rows = conn.execute(
                    "SELECT tool_id, server, name, version, embedding FROM tools WHERE server = ?",
                    (server_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT tool_id, server, name, version, embedding FROM tools"
                ).fetchall()

        scored: list[ToolCandidate] = []
        for tool_id, server, name, version, emb_blob in rows:
            emb = _decode_vec(emb_blob)
            score = float(cosine_similarity(query_embedding, emb))
            scored.append(
                ToolCandidate(
                    tool_id=str(tool_id),
                    score=score,
                    server=str(server),
                    name=str(name),
                    version=None if version is None else str(version),
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: max(0, top_k)]

    def get_tool_payload_json(self, tool_id: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM tools WHERE tool_id = ?",
                (tool_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def get_tool_payloads_json(self, tool_ids: list[str]) -> dict[str, str]:
        if not tool_ids:
            return {}
        placeholders = ",".join("?" for _ in tool_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT tool_id, payload_json FROM tools WHERE tool_id IN ({placeholders})",
                tuple(tool_ids),
            ).fetchall()
        return {str(tool_id): str(payload_json) for tool_id, payload_json in rows}

    # --------------------
    # Resources + chunks
    # --------------------
    def get_resource_hash(self, resource_id: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM resources WHERE resource_id = ?",
                (resource_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def get_resource_content_hash(self, resource_id: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT resource_content_hash FROM resources WHERE resource_id = ?",
                (resource_id,),
            ).fetchone()
        if row is None:
            return None
        return None if row[0] is None else str(row[0])

    def upsert_resource(
        self,
        *,
        resource_id: str,
        server: str,
        uri: str,
        name: str,
        version: Optional[str],
        content_hash: str,
        descriptor_json: str,
        resource_content_hash: Optional[str] = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resources(resource_id, server, uri, name, version, content_hash, descriptor_json, resource_content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_id) DO UPDATE SET
                  server=excluded.server,
                  uri=excluded.uri,
                  name=excluded.name,
                  version=excluded.version,
                  content_hash=excluded.content_hash,
                  descriptor_json=excluded.descriptor_json,
                  resource_content_hash=excluded.resource_content_hash
                """,
                (
                    resource_id,
                    server,
                    uri,
                    name,
                    version,
                    content_hash,
                    descriptor_json,
                    resource_content_hash,
                ),
            )

    def delete_resources_not_in(self, resource_ids: set[str]) -> int:
        if not resource_ids:
            with self.connect() as conn:
                cur = conn.execute("DELETE FROM resources")
                return int(cur.rowcount)
        placeholders = ",".join("?" for _ in resource_ids)
        with self.connect() as conn:
            cur = conn.execute(
                f"DELETE FROM resources WHERE resource_id NOT IN ({placeholders})",
                tuple(resource_ids),
            )
            return int(cur.rowcount)

    def delete_chunks_for_resource(self, resource_id: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM resource_chunks WHERE resource_id = ?",
                (resource_id,),
            )
            return int(cur.rowcount)

    def upsert_resource_chunk(
        self,
        *,
        chunk_id: str,
        resource_id: str,
        server: str,
        uri: str,
        chunk_index: int,
        content_hash: str,
        text: str,
        embedding: list[float],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resource_chunks(chunk_id, resource_id, server, uri, chunk_index, content_hash, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                  resource_id=excluded.resource_id,
                  server=excluded.server,
                  uri=excluded.uri,
                  chunk_index=excluded.chunk_index,
                  content_hash=excluded.content_hash,
                  text=excluded.text,
                  embedding=excluded.embedding
                """,
                (
                    chunk_id,
                    resource_id,
                    server,
                    uri,
                    chunk_index,
                    content_hash,
                    text,
                    _encode_vec(embedding),
                ),
            )

    def query_resource_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        server_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if server_filter:
                rows = conn.execute(
                    "SELECT chunk_id, resource_id, server, uri, chunk_index, text, embedding FROM resource_chunks WHERE server = ?",
                    (server_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chunk_id, resource_id, server, uri, chunk_index, text, embedding FROM resource_chunks"
                ).fetchall()

        scored: list[dict[str, Any]] = []
        for chunk_id, resource_id, server, uri, chunk_index, text, emb_blob in rows:
            emb = _decode_vec(emb_blob)
            score = float(cosine_similarity(query_embedding, emb))
            scored.append(
                {
                    "chunk_id": str(chunk_id),
                    "resource_id": str(resource_id),
                    "server": str(server),
                    "uri": str(uri),
                    "chunk_index": int(chunk_index),
                    "text": str(text),
                    "score": score,
                }
            )
        scored.sort(key=lambda r: float(r["score"]), reverse=True)
        return scored[: max(0, top_k)]

