from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    encoding_dim INTEGER NOT NULL CHECK (encoding_dim = 128),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_known_faces_name ON known_faces(name)"
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

    def add_face(self, name: str, encoding: np.ndarray) -> int:
        """Add a person's face encoding and return the SQLite row id."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name cannot be empty")

        encoding_blob = self._serialize_encoding(encoding)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO known_faces (name, encoding, encoding_dim, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, encoding_blob, 128, created_at),
            )
            return int(cursor.lastrowid)

    def list_faces(self) -> list[dict]:
        """Return registered faces without loading the embedding blobs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, encoding_dim, created_at FROM known_faces ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def load_encodings(self) -> tuple[list[str], list[np.ndarray]]:
        """Load all names and 128D encodings from SQLite."""
        with self._connect() as conn:
            rows = conn.execute("SELECT name, encoding FROM known_faces ORDER BY id").fetchall()

        names = [row["name"] for row in rows]
        encodings = [self._deserialize_encoding(row["encoding"]) for row in rows]
        return names, encodings

    def recognize_face(
        self, face_encoding: np.ndarray, tolerance: float = 0.6
    ) -> tuple[str | None, float | None]:
        """
        Compare a 128D face encoding with registered SQLite encodings.

        Returns (name, confidence) for the closest match, otherwise (None, None).
        """
        candidate_encoding = self._normalize_encoding(face_encoding)
        names, known_encodings = self.load_encodings()
        if not known_encodings:
            return None, None

        distances = np.linalg.norm(np.asarray(known_encodings) - candidate_encoding, axis=1)
        min_distance_idx = int(np.argmin(distances))
        min_distance = float(distances[min_distance_idx])

        if min_distance <= tolerance:
            return names[min_distance_idx], 1.0 - min_distance

        return None, None
