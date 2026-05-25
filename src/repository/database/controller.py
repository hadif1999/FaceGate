from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import numpy as np
from loguru import logger


class FaceDatabase:
    """SQLite-backed storage and lookup for 128-dimensional face encodings."""

    def __init__(self, db_path: str = "data/face_embeddings.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self):
        """Yield a connection and guarantee it is closed afterwards."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding   BLOB,
                    encoding_dim INTEGER NOT NULL CHECK (encoding_dim = 128),
                    member_id   INTEGER NOT NULL UNIQUE,
                    created_at  TEXT    NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_encoding(encoding: np.ndarray) -> np.ndarray:
        encoding_array = np.asarray(encoding, dtype=np.float64).reshape(-1)
        if encoding_array.shape != (128,):
            raise ValueError(
                f"face_recognition encodings must be 128-dimensional; "
                f"got {encoding_array.shape}"
            )
        return encoding_array

    @staticmethod
    def _unit_normalize(vec: np.ndarray) -> np.ndarray:
        """Return L2-unit-normalized copy of vec (safe against zero vector)."""
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero vector.")
        return vec / norm

    @classmethod
    def _serialize_encoding(cls, encoding: np.ndarray) -> bytes:
        return cls._normalize_encoding(encoding).tobytes()

    @staticmethod
    def _deserialize_encoding(blob: bytes) -> np.ndarray:
        encoding = np.frombuffer(blob, dtype=np.float64)
        if encoding.shape != (128,):
            raise ValueError(f"Stored face encoding has invalid shape: {encoding.shape}")
        return encoding

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_user(self, embedding: np.ndarray, member_id: int) -> int:
        """Insert a new user row and return the assigned id."""
        blob = self._serialize_encoding(embedding)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (embedding, encoding_dim, member_id, created_at) VALUES (?, ?, ?, ?)",
                (blob, 128, member_id, created_at),
            )
            return int(cursor.lastrowid)


    def add_face(self, encoding: np.ndarray|None, member_id: int) -> int:
        """Alias for add_user. Returns the assigned user id."""
        return self.add_user(encoding, member_id)


    def delete_face(self, face_id: int) -> bool:
        """
        Delete the user with the given id.

        Returns:
            True if a row was deleted, False if the id was not found.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (face_id,))
            return cursor.rowcount > 0


    def update_face(self, face_id: int, new_encoding: np.ndarray) -> bool:
        """
        Replace the embedding vector for an existing user.

        Args:
            face_id: The id of the user to update.
            new_encoding: New 128-dimensional face encoding.

        Returns:
            True if the row was updated, False if the id was not found.
        """
        blob = self._serialize_encoding(new_encoding)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET embedding = ? WHERE id = ?",
                (blob, face_id),
            )
            return cursor.rowcount > 0
        
        
    def get_face_by_member_id(self, member_id: int) -> dict | None:
        """
        Retrieve the record for a given member_id with all fields,
        including the deserialized embedding.

        Returns:
            A dict with keys: id, embedding (np.ndarray), encoding_dim, member_id, created_at.
            None if no such member exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, embedding, encoding_dim, member_id, created_at FROM users WHERE member_id = ?",
                (member_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "embedding": self._deserialize_encoding(row["embedding"]),
            "encoding_dim": row["encoding_dim"],
            "member_id": row["member_id"],
            "created_at": row["created_at"],
        }


    def del_by_member_id(self, member_id: int) -> bool:
        """
        Delete the record associated with the given member_id.

        Returns:
            True if a row was deleted, False if the member_id was not found.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE member_id = ?", (member_id,))
            return cursor.rowcount > 0


    def del_all(self) -> None:
        """Delete all records from the users table."""
        with self._connect() as conn:
            conn.execute("DELETE FROM users")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_face_data(self) -> list[dict]:
        """Return all users as a list of dicts with deserialized embeddings."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding, encoding_dim, member_id, created_at FROM users ORDER BY id"
            ).fetchall()
        data = [
            {
                "id": row["id"],
                "embedding": self._deserialize_encoding(row["embedding"]),
                "encoding_dim": row["encoding_dim"],
                "member_id": row["member_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return data


    def recognize_face(
        self,
        face_encoding: np.ndarray,
        tolerance: float = 0.6,
        metric: Literal["l2", "cosine"] = "cosine",
    ) -> tuple[int | None, float | None]:
        """
        Find the closest registered face to the query encoding.

        Args:
            face_encoding: Query embedding (128D).
            tolerance:
                - "l2":     maximum L2 distance to accept (lower = stricter).
                - "cosine": minimum cosine similarity to accept (higher = stricter).
            metric: "l2" or "cosine".

        Returns:
            (user_id, confidence) on match, (None, None) if no match.
        """
        candidate = self._normalize_encoding(face_encoding)
        face_data = self.list_face_data()
        if not face_data:
            return None, None

        known = np.asarray([u["embedding"] for u in face_data])  # (N, 128)

        if metric == "l2":
            distances = np.linalg.norm(known - candidate, axis=1)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist <= tolerance:
                # Map distance to a [0, 1] confidence (clamped)
                confidence = float(np.clip(1.0 - best_dist, 0.0, 1.0))
                return int(face_data[best_idx]["id"]), confidence

        elif metric == "cosine":
            # Normalize both sides so dot product == cosine similarity
            known_unit = known / np.linalg.norm(known, axis=1, keepdims=True)
            candidate_unit = self._unit_normalize(candidate)

            similarities = np.dot(known_unit, candidate_unit)  # (N,)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= tolerance:
                confidence = float(np.clip(best_sim, 0.0, 1.0))
                return int(face_data[best_idx]["id"]), confidence

        return None, None
