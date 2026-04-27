"""MySQL persistence for per-candidate rubric labels.

Labels are the ground truth for the calibration loop:
  - per-dimension 0-10 scores supplied by a human labeler
  - an overall 0-10 score for the same candidate
  - optional short note

Schema is auto-created idempotently on first write. The primary key is
(profile_id, labeler, created_at) so the same person can re-label a candidate
and we keep history rather than silently overwriting — the calibrator uses
the most recent label per (profile_id, labeler) pair.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymysql
from dotenv import load_dotenv

LABELS_TABLE_NAME = "recruiter_rubric_labels"
MAX_NOTE_LENGTH = 1000
MAX_LABELER_LENGTH = 64
SCORE_LOWER_BOUND = 0.0
SCORE_UPPER_BOUND = 10.0


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")


def _open_connection() -> pymysql.connections.Connection:
    """Open a MySQL connection using DB_* vars from .env.

    Kept local (rather than reusing retriever._open_connection) so the labels
    store stays self-contained; calibrate.py still imports the retriever helper
    when it needs read access to the profiles table.
    """
    _load_environment()

    database_host = os.environ.get("DB_HOST", "").strip()
    database_user = os.environ.get("DB_USER", "").strip()
    database_password = os.environ.get("DB_PASSWORD", "")
    database_name = os.environ.get("DB_NAME", "").strip()
    database_port_text = os.environ.get("DB_PORT", "3306").strip()

    if not all((database_host, database_user, database_password, database_name)):
        missing_names = [
            name
            for name in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
            if not os.environ.get(name, "").strip()
        ]
        raise ValueError(f"Missing required DB env vars: {', '.join(missing_names)}")

    return pymysql.connect(
        host=database_host,
        user=database_user,
        password=database_password,
        database=database_name,
        port=int(database_port_text),
        charset="utf8mb4",
        connect_timeout=10,
        autocommit=True,
    )


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {LABELS_TABLE_NAME} (
    profile_id VARCHAR(128) NOT NULL,
    labeler VARCHAR({MAX_LABELER_LENGTH}) NOT NULL,
    dim_scores JSON NOT NULL,
    overall_score DECIMAL(4,2) NOT NULL,
    note TEXT NULL,
    created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (profile_id, labeler, created_at),
    KEY idx_labeler_created (labeler, created_at),
    KEY idx_profile_created (profile_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def ensure_labels_table() -> None:
    """Create the labels table if it does not already exist."""
    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(CREATE_TABLE_SQL)


def _validate_score(raw_score: Any, field_name: str) -> float:
    try:
        numeric_score = float(raw_score)
    except (TypeError, ValueError) as coercion_error:
        raise ValueError(f"{field_name} must be numeric, got {raw_score!r}") from coercion_error
    if not (SCORE_LOWER_BOUND <= numeric_score <= SCORE_UPPER_BOUND):
        raise ValueError(
            f"{field_name} must be within [{SCORE_LOWER_BOUND}, {SCORE_UPPER_BOUND}], got {numeric_score}"
        )
    return round(numeric_score, 2)


def _validate_dim_scores(raw_dim_scores: Dict[str, Any], required_keys: List[str]) -> Dict[str, float]:
    if not isinstance(raw_dim_scores, dict):
        raise ValueError(f"dim_scores must be a dict, got {type(raw_dim_scores).__name__}")
    missing_keys = [key for key in required_keys if key not in raw_dim_scores]
    if missing_keys:
        raise ValueError(f"dim_scores missing keys: {missing_keys}")
    cleaned: Dict[str, float] = {}
    for dimension_key in required_keys:
        cleaned[dimension_key] = _validate_score(raw_dim_scores[dimension_key], f"dim_scores['{dimension_key}']")
    return cleaned


def save_label(
    profile_id: str,
    labeler: str,
    dim_scores: Dict[str, float],
    overall_score: float,
    note: Optional[str],
    required_dimension_keys: List[str],
) -> None:
    """Insert a single label row. Auto-creates the table on first call."""
    profile_id_clean = (profile_id or "").strip()
    labeler_clean = (labeler or "").strip()[:MAX_LABELER_LENGTH]
    if not profile_id_clean:
        raise ValueError("profile_id must be non-empty")
    if not labeler_clean:
        raise ValueError("labeler must be non-empty")

    cleaned_dim_scores = _validate_dim_scores(dim_scores, required_dimension_keys)
    cleaned_overall = _validate_score(overall_score, "overall_score")
    cleaned_note = (note or "").strip()[:MAX_NOTE_LENGTH] or None

    ensure_labels_table()

    insert_sql = f"""
    INSERT INTO {LABELS_TABLE_NAME}
        (profile_id, labeler, dim_scores, overall_score, note, created_at)
    VALUES
        (%s, %s, %s, %s, %s, %s)
    """

    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(
                insert_sql,
                (
                    profile_id_clean,
                    labeler_clean,
                    json.dumps(cleaned_dim_scores),
                    cleaned_overall,
                    cleaned_note,
                    datetime.utcnow(),
                ),
            )


def _row_to_label(row: Any) -> Dict[str, Any]:
    profile_id_value, labeler_value, dim_scores_value, overall_value, note_value, created_at_value = row
    try:
        parsed_dim_scores = json.loads(dim_scores_value) if isinstance(dim_scores_value, str) else dim_scores_value
    except (TypeError, json.JSONDecodeError):
        parsed_dim_scores = {}
    return {
        "profile_id": profile_id_value,
        "labeler": labeler_value,
        "dim_scores": parsed_dim_scores or {},
        "overall_score": float(overall_value) if overall_value is not None else 0.0,
        "note": note_value or "",
        "created_at": created_at_value,
    }


def load_labels(
    labeler: Optional[str] = None,
    latest_per_profile_labeler: bool = True,
) -> List[Dict[str, Any]]:
    """Return all labels, optionally filtered by labeler.

    If `latest_per_profile_labeler` is True (default for the calibrator), only
    the most recent label per (profile_id, labeler) pair is returned, so a
    human who re-rated a candidate does not contribute two rows to the fit.
    """
    ensure_labels_table()

    if latest_per_profile_labeler:
        base_sql = f"""
        SELECT t.profile_id, t.labeler, t.dim_scores, t.overall_score, t.note, t.created_at
        FROM {LABELS_TABLE_NAME} t
        INNER JOIN (
            SELECT profile_id, labeler, MAX(created_at) AS max_created_at
            FROM {LABELS_TABLE_NAME}
            {{labeler_filter}}
            GROUP BY profile_id, labeler
        ) latest
            ON latest.profile_id = t.profile_id
           AND latest.labeler = t.labeler
           AND latest.max_created_at = t.created_at
        ORDER BY t.created_at DESC
        """
    else:
        base_sql = f"""
        SELECT profile_id, labeler, dim_scores, overall_score, note, created_at
        FROM {LABELS_TABLE_NAME}
        {{labeler_filter}}
        ORDER BY created_at DESC
        """

    query_params: List[Any] = []
    labeler_filter_clause = ""
    if labeler is not None and labeler.strip():
        labeler_filter_clause = "WHERE labeler = %s"
        query_params.append(labeler.strip())

    final_sql = base_sql.format(labeler_filter=labeler_filter_clause)

    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(final_sql, tuple(query_params))
            query_rows = database_cursor.fetchall()

    return [_row_to_label(row) for row in query_rows]


def fetch_unlabeled_candidates(
    labeler: str,
    limit: int = 20,
    prefer_diverse: bool = True,
) -> List[Dict[str, Any]]:
    """Pick profiles this labeler has NOT yet rated, ordered for fast labeling.

    Returns the same row shape as `src.retriever._row_to_candidate`
    (profile_id, name, headline, location, about_text, *_count, *_json) so
    the queue page can show the same candidate snippet the candidate cards
    do.

    Selection logic:
      1. SQL-side: exclude profiles where (profile_id, labeler) already has
         a row in `recruiter_rubric_labels`. So if you re-rate, you re-rate
         from the cards page, not from the queue.
      2. `prefer_diverse=True` (default): prioritize profiles with longer
         about_text and more experience entries. The reasoning: those
         profiles carry more signal per minute of labeler time, and their
         labels move the calibrator more than thin profiles whose dim
         scores are dominated by missing-data fallbacks.
      3. `prefer_diverse=False`: random order, useful as a sanity check
         that the diverse path isn't biasing the label distribution.

    Bounded at `limit` rows so the UI can render the whole queue at once
    in `st.session_state` without paging. Caller is expected to clamp
    `limit` to something sane (5-50).

    Notes:
      - Read-only on `linkedin_api_profiles_parsed`; the labels schema is
        only read via the LEFT JOIN.
      - Skips the active-learning "label profiles where heuristic
        disagrees with seen labels" angle for now; random-of-rich beats
        nothing for a single-labeler workflow at this scale (50/week).
    """
    cleaned_labeler = (labeler or "").strip()[:MAX_LABELER_LENGTH]
    if not cleaned_labeler:
        raise ValueError("labeler must be non-empty")

    bounded_limit = max(1, min(int(limit), 200))

    ensure_labels_table()

    if prefer_diverse:
        ordering_clause = (
            "ORDER BY "
            "  COALESCE(p.about_char_count, LENGTH(p.about_text), 0) DESC, "
            "  COALESCE(p.experience_count, 0) DESC, "
            "  p.profile_id ASC"
        )
    else:
        ordering_clause = "ORDER BY RAND()"

    fetch_sql = f"""
    SELECT
        p.profile_id,
        p.name,
        p.headline,
        p.location,
        p.about_text,
        COALESCE(p.skills_count, 0) AS skills_count,
        COALESCE(p.experience_count, 0) AS experience_count,
        COALESCE(p.education_count, 0) AS education_count,
        p.experience_json,
        p.education_json
    FROM linkedin_api_profiles_parsed p
    LEFT JOIN (
        SELECT DISTINCT profile_id
        FROM {LABELS_TABLE_NAME}
        WHERE labeler = %s
    ) labeled_set ON labeled_set.profile_id = p.profile_id
    WHERE labeled_set.profile_id IS NULL
    {ordering_clause}
    LIMIT %s
    """

    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(fetch_sql, (cleaned_labeler, bounded_limit))
            query_rows = database_cursor.fetchall()

    candidates: List[Dict[str, Any]] = []
    for row in query_rows:
        (
            profile_id_value,
            name_value,
            headline_value,
            location_value,
            about_text_value,
            skills_count_value,
            experience_count_value,
            education_count_value,
            experience_json_value,
            education_json_value,
        ) = row
        candidates.append(
            {
                "profile_id": profile_id_value,
                "name": name_value or "",
                "headline": headline_value or "",
                "location": location_value or "",
                "about_text": about_text_value or "",
                "skills_count": int(skills_count_value or 0),
                "experience_count": int(experience_count_value or 0),
                "education_count": int(education_count_value or 0),
                "experience_json": experience_json_value or "",
                "education_json": education_json_value or "",
            }
        )
    return candidates


def count_labels(labeler: Optional[str] = None) -> int:
    """Return the number of (profile_id, labeler) pairs with at least one label.

    Matches what `load_labels(latest_per_profile_labeler=True)` would return so
    the UI counter and the calibrator see the same N.
    """
    try:
        ensure_labels_table()
    except Exception:  # noqa: BLE001 - counter should never raise into Streamlit
        return 0

    base_sql = f"""
    SELECT COUNT(*) FROM (
        SELECT profile_id, labeler
        FROM {LABELS_TABLE_NAME}
        {{labeler_filter}}
        GROUP BY profile_id, labeler
    ) aggregated
    """
    query_params: List[Any] = []
    labeler_filter_clause = ""
    if labeler is not None and labeler.strip():
        labeler_filter_clause = "WHERE labeler = %s"
        query_params.append(labeler.strip())

    final_sql = base_sql.format(labeler_filter=labeler_filter_clause)

    try:
        with _open_connection() as database_connection:
            with database_connection.cursor() as database_cursor:
                database_cursor.execute(final_sql, tuple(query_params))
                count_row = database_cursor.fetchone()
    except Exception:  # noqa: BLE001 - counter should never raise into Streamlit
        return 0

    if not count_row:
        return 0
    return int(count_row[0] or 0)
