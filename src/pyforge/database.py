"""Persist and retrieve practice attempts in the local SQLite database.

    Attempts include submitted code, explanations, AI feedback, reference
    explanations, semantic similarity scores, and UTC submission timestamps.

"""
import sqlite3
from pathlib import Path

# Resolve storage relative to the project, independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "pyforge.db"


def initialize_database() -> None:
    
    """Create the attempts table and add missing assessment columns.

            Creates the data directory when needed. Existing attempts are preserved,
            and repeated calls do not add duplicate columns.

            Returns:
                None.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, 
                exercise_id INTEGER NOT NULL, 
                code TEXT NOT NULL,
                explanation TEXT NOT NULL DEFAULT '',
                feedback TEXT NOT NULL, 
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        # CREATE TABLE IF NOT EXISTS does not update an existing schema.
        # Inspect its columns before applying the additive migrations.
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(attempts)")
        }

        if "reference" not in columns:
            connection.execute(
                "ALTER TABLE attempts ADD COLUMN reference TEXT"
            )

        if "similarity" not in columns:
            connection.execute(
                "ALTER TABLE attempts ADD COLUMN similarity REAL"
            )


def save_attempt(
        exercise_id: int,
        code: str, 
        feedback: str,
        explanation: str = "",
        reference: str | None = None,
        similarity: float | None = None,    
) -> int:

    """Save a new attempt without overwriting previous submissions.

        Args:
            exercise_id: Identifier of the exercise being attempted.
            code: Submitted Python code, stored without execution.
            feedback: AI coaching feedback.
            explanation: The student's explanation in their own words.
            reference: AI reference explanation, or None when unavailable.
            similarity: Cosine similarity score, or None when not calculated.

        Returns:
            The generated ID of the saved attempt.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        
        # Bind values through placeholders rather than building SQL from input.
        # The connection context commits the INSERT on successful exit.
        cursor = connection.execute(
            """
            INSERT INTO attempts (
                exercise_id, code, explanation, feedback, reference, similarity
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (exercise_id, code, explanation, feedback, reference, similarity),
        )
        return cursor.lastrowid


def get_attempts(exercise_id: int):

    with sqlite3.connect(DATABASE_PATH) as connection:

        # Preserve column names so each result can become a dictionary.
        connection.row_factory= sqlite3.Row

    rows = connection.execute(
        """
        SELECT id, exercise_id, code, explanation, feedback, created_at, reference, similarity
        from attempts
        WHERE exercise_id = ?
        ORDER BY id DESC
        """,
        (exercise_id,),
    ).fetchall()

    return [dict(row) for row in rows]