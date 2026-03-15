from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SQLiteVectorStore:
    """
    Simple persistent vector store with cosine similarity search.
    Vectors are stored as JSON arrays in SQLite.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                namespace TEXT NOT NULL,
                item_id TEXT NOT NULL,
                vector TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (namespace, item_id)
            )
            """
        )
        self._conn.commit()

    def upsert(
        self,
        namespace: str,
        item_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        # Normalize potential numpy scalar types (e.g. float32 from fastembed)
        # into plain Python floats so JSON serialization is stable.
        vector_json = json.dumps([float(v) for v in vector])
        self._conn.execute(
            """
            INSERT INTO vectors(namespace, item_id, vector, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, item_id) DO UPDATE SET
                vector = excluded.vector,
                payload = excluded.payload
            """,
            (namespace, item_id, vector_json, json.dumps(payload)),
        )
        self._conn.commit()

    def get_payload(self, namespace: str, item_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload FROM vectors WHERE namespace = ? AND item_id = ?",
            (namespace, item_id),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def list_items(self, namespace: str) -> List[Tuple[str, Dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT item_id, payload FROM vectors WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        return [(item_id, json.loads(payload_json)) for item_id, payload_json in rows]

    def delete(self, namespace: str, item_id: str) -> None:
        self._conn.execute(
            "DELETE FROM vectors WHERE namespace = ? AND item_id = ?",
            (namespace, item_id),
        )
        self._conn.commit()

    def search(
        self,
        namespace: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT item_id, vector, payload FROM vectors WHERE namespace = ?",
            (namespace,),
        ).fetchall()

        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for item_id, vector_json, payload_json in rows:
            vector = json.loads(vector_json)
            score = _cosine_similarity(query_vector, vector)
            scored.append((item_id, float(score), json.loads(payload_json)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
