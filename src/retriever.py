"""MySQL retrieval helpers for the recruiter LangGraph workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pymysql
from dotenv import load_dotenv

DEFAULT_TOP_K = 8
DEFAULT_MYSQL_PORT = 3306
MAX_TOP_K = 20
MAX_KEYWORDS = 8
MIN_ABOUT_CHARS = 40


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")


def _open_connection() -> pymysql.connections.Connection:
    _load_environment()

    database_host = os.environ.get("DB_HOST", "").strip()
    database_user = os.environ.get("DB_USER", "").strip()
    database_password = os.environ.get("DB_PASSWORD", "")
    database_name = os.environ.get("DB_NAME", "").strip()

    if not all((database_host, database_user, database_password, database_name)):
        missing_names = [
            name
            for name in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
            if not os.environ.get(name, "").strip()
        ]
        raise ValueError(f"Missing required DB env vars: {', '.join(missing_names)}")

    port_text = os.environ.get("DB_PORT", str(DEFAULT_MYSQL_PORT)).strip()
    database_port = int(port_text)

    return pymysql.connect(
        host=database_host,
        user=database_user,
        password=database_password,
        database=database_name,
        port=database_port,
        charset="utf8mb4",
        connect_timeout=10,
    )


def _sanitize_limit(limit_value: int) -> int:
    return max(1, min(int(limit_value), MAX_TOP_K))


def _clean_keyword_list(raw_keywords: Optional[Sequence[str]]) -> List[str]:
    if not raw_keywords:
        return []
    deduplicated: List[str] = []
    seen: set = set()
    for keyword in raw_keywords:
        if not isinstance(keyword, str):
            continue
        cleaned_value = keyword.strip()
        if not cleaned_value:
            continue
        normalized_key = cleaned_value.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        deduplicated.append(cleaned_value)
        if len(deduplicated) >= MAX_KEYWORDS:
            break
    return deduplicated


def _row_to_candidate(row: Sequence[Any]) -> Dict[str, Any]:
    return {
        "profile_id": row[0],
        "name": row[1] or "",
        "headline": row[2] or "",
        "location": row[3] or "",
        "about_text": row[4] or "",
        "skills_count": int(row[5] or 0),
        "experience_count": int(row[6] or 0),
        "education_count": int(row[7] or 0),
        "experience_json": row[8] or "",
        "education_json": row[9] or "",
        "relevance_score": int(row[10] or 0),
    }


def search_profiles(
    role_keywords: Optional[Sequence[str]] = None,
    skills: Optional[Sequence[str]] = None,
    location: Optional[str] = None,
    min_experience_entries: int = 0,
    must_have_keywords: Optional[Sequence[str]] = None,
    nice_to_have_keywords: Optional[Sequence[str]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Retrieve candidates from linkedin_api_profiles_parsed using structured filters.

    Scoring is lexical: weighted LIKE matches across `about_text`, `headline`, and `name`.
    Structured filters (skills, location, min_experience_entries, must_have) are applied
    as SQL WHERE constraints so we do not waste LLM tokens on clearly unqualified rows.
    """
    role_terms = _clean_keyword_list(role_keywords)
    skill_terms = _clean_keyword_list(skills)
    must_have_terms = _clean_keyword_list(must_have_keywords)
    nice_to_have_terms = _clean_keyword_list(nice_to_have_keywords)
    location_clean = (location or "").strip()
    sanitized_limit = _sanitize_limit(top_k)
    minimum_experience = max(0, int(min_experience_entries or 0))

    score_clauses: List[str] = []
    score_params: List[Any] = []

    weighted_terms = (
        [(term, 3) for term in role_terms]
        + [(term, 4) for term in skill_terms]
        + [(term, 5) for term in must_have_terms]
        + [(term, 1) for term in nice_to_have_terms]
    )

    for term, weight in weighted_terms:
        score_clauses.append(
            "CASE WHEN LOWER(COALESCE(about_text, '')) LIKE LOWER(CONCAT('%%', %s, '%%')) THEN %s ELSE 0 END"
        )
        score_params.extend([term, weight])
        score_clauses.append(
            "CASE WHEN LOWER(COALESCE(headline, '')) LIKE LOWER(CONCAT('%%', %s, '%%')) THEN %s ELSE 0 END"
        )
        score_params.extend([term, max(1, weight - 1)])

    if not score_clauses:
        score_clauses.append("0")

    where_clauses: List[str] = [
        "about_text IS NOT NULL",
        "CHAR_LENGTH(about_text) > %s",
    ]
    where_params: List[Any] = [MIN_ABOUT_CHARS]

    if minimum_experience > 0:
        where_clauses.append("experience_count >= %s")
        where_params.append(minimum_experience)

    if location_clean:
        where_clauses.append("LOWER(COALESCE(location, '')) LIKE LOWER(CONCAT('%%', %s, '%%'))")
        where_params.append(location_clean)

    for must_have_term in must_have_terms:
        where_clauses.append(
            "("
            "LOWER(COALESCE(about_text, '')) LIKE LOWER(CONCAT('%%', %s, '%%')) "
            "OR LOWER(COALESCE(headline, '')) LIKE LOWER(CONCAT('%%', %s, '%%'))"
            ")"
        )
        where_params.extend([must_have_term, must_have_term])

    sql_query = f"""
    SELECT
        profile_id,
        name,
        headline,
        location,
        about_text,
        skills_count,
        experience_count,
        education_count,
        experience_json,
        education_json,
        ({' + '.join(score_clauses)}) AS relevance_score
    FROM linkedin_api_profiles_parsed
    WHERE {' AND '.join(where_clauses)}
    ORDER BY relevance_score DESC, about_char_count DESC
    LIMIT %s
    """

    query_params = tuple(score_params + where_params + [sanitized_limit])

    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(sql_query, query_params)
            query_rows = database_cursor.fetchall()

    return [_row_to_candidate(row) for row in query_rows]


def fetch_profile_candidates(question_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Backward-compatible retrieval using the raw question as a single keyword."""
    question_clean = (question_text or "").strip()
    keywords = [question_clean] if question_clean else []
    return search_profiles(
        role_keywords=keywords,
        top_k=top_k,
    )
