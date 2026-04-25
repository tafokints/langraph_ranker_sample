"""Typed schemas for the recruiter LangGraph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ParsedQuery(TypedDict):
    """Structured filters extracted from a free-text recruiter query."""

    role_keywords: List[str]
    skills: List[str]
    location: str
    min_experience_entries: int
    must_have_keywords: List[str]
    nice_to_have_keywords: List[str]


class RankedCandidate(TypedDict):
    """A candidate enriched with LLM or deterministic ranking output."""

    profile_id: str
    name: str
    headline: str
    location: str
    about_text: str
    skills_count: int
    experience_count: int
    education_count: int
    relevance_score: int
    rank_score: float
    match_reasons: List[str]
    risks: List[str]


class RecruiterGraphState(TypedDict, total=False):
    """Full state passed between nodes in the recruiter graph."""

    question_text: str
    top_k: int
    min_experience_entries: int
    parsed_query: ParsedQuery
    candidate_profiles: List[Dict[str, Any]]
    ranked_candidates: List[RankedCandidate]
    shortlist_summary: str
    error_messages: List[str]


DEFAULT_PARSED_QUERY: ParsedQuery = {
    "role_keywords": [],
    "skills": [],
    "location": "",
    "min_experience_entries": 0,
    "must_have_keywords": [],
    "nice_to_have_keywords": [],
}
