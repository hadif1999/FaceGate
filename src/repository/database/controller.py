from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import numpy as np


class FaceDatabase:
    """SQLite-backed storage and lookup for 128-dimensional face encodings."""

    def __init__(self, db_path: str = "data/face_embeddings.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB NOT NULL,
                    encoding_dim INTEGER NOT NULL CHECK (encoding_dim = 128),
                    created_at TEXT NOT NULL
                )
                """
            )
            self._migrate_known_faces(conn)

    def _migrate_known_faces(self, conn: sqlite3.Connection) -> None:
        """Move legacy known_faces rows into the new user->id, embedding format."""
        legacy_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'known_faces'"
        ).fetchone()
        if legacy_table is None:
            return

        has_users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        if has_users:
            return

        legacy_rows = conn.execute(
            "SELECT encoding, encoding_dim, created_at FROM known_faces ORDER BY id"
        ).fetchall()
        conn.executemany(
            """
            INSERT INTO users (embedding, encoding_dim, created_at)
            VALUES (?, ?, ?)
            """,
            [
                (row["encoding"], row["encoding_dim"], row["created_at"])
                for row in legacy_rows
            ],
        )

    @staticmethod
    def _normalize_encoding(encoding: np.ndarray) -> np.ndarray:
        encoding_array = np.asarray(encoding, dtype=np.float64)
        if encoding_array.shape != (128,):
            raise ValueError(
                f"face_recognition encodings must be 128-dimensional; got {encoding_array.shape}"
            )
        return encoding_array

    @classmethod
    def _serialize_encoding(cls, encoding: np.ndarray) -> bytes:
        return cls._normalize_encoding(encoding).tobytes()

    @staticmethod
    def _deserialize_encoding(encoding_blob: bytes) -> np.ndarray:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        if encoding.shape != (128,):
            raise ValueError(f"stored face encoding is invalid: {encoding.shape}")
        return encoding

    def add_user(self, embedding: np.ndarray) -> int:
        """Add a new user embedding and return the assigned user id."""
        embedding_blob = self._serialize_encoding(embedding)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (embedding, encoding_dim, created_at)
                VALUES (?, ?, ?)
                """,
                (embedding_blob, 128, created_at),
            )
            return int(cursor.lastrowid)

    def add_face(self, name: str, encoding: np.ndarray) -> int:
        """Backward-compatible alias; user identity is the assigned SQLite id."""
        return self.add_user(encoding)

    def list_face_data(self) -> list[dict]:
        """Return users as {'id': int, 'embedding': np.ndarray, ...} dictionaries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding, encoding_dim, created_at FROM users ORDER BY id"
            ).fetchall()

        return [
            {
                "id": row["id"],
                "embedding": self._deserialize_encoding(row["embedding"]),
                "encoding_dim": row["encoding_dim"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


    def recognize_face(
        self,
        face_encoding: np.ndarray,
        tolerance: float = 0.6,
        metric: Literal["l2", "cosine"] = "cosine"
    ) -> tuple[int | None, float | None]:
        """
        Compare a 128D face encoding with registered SQLite encodings.

        Args:
            face_encoding: The query face embedding (128D).
            tolerance: Similarity threshold.
                - For "l2": maximum L2 distance (default 0.6, lower = more similar)
                - For "cosine": minimum cosine similarity (default 0.6, higher = more similar)
            metric: Distance metric - "l2" (Euclidean) or "cosine" (cosine similarity).

        Returns:
            (user_id, confidence) for the closest match, otherwise (None, None).
        """
        candidate_encoding = self._normalize_encoding(face_encoding)
        face_data: list[dict] = self.list_face_data()
        if not face_data:
            return None, None

        known_embeddings = np.asarray([user["embedding"] for user in face_data])

        if metric == "l2":
            # L2 (Euclidean) distance
            distances = np.linalg.norm(known_embeddings - candidate_encoding, axis=1)
            min_idx = int(np.argmin(distances))
            min_distance = float(distances[min_idx])

            if min_distance <= tolerance:
                confidence = 1.0 - min_distance
                return int(face_data[min_idx]["id"]), confidence

        elif metric == "cosine":
            # Cosine similarity (for normalized vectors, this equals dot product)
            # Range: [-1, 1], where 1 = identical direction
            similarities = np.dot(known_embeddings, candidate_encoding)
            max_idx = int(np.argmax(similarities))
            max_similarity = float(similarities[max_idx])

            if max_similarity >= tolerance:
                # Clip to [0, 1] for confidence (handles edge cases with non-normalized data)
                confidence = float(np.clip(max_similarity, 0.0, 1.0))
                return int(face_data[max_idx]["id"]), confidence

        return None, None
