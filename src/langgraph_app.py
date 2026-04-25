"""Recruiter LangGraph workflow over linkedin_api_profiles_parsed.

Graph nodes:
    parse_query -> retrieve_candidates -> rank_candidates -> synthesize_shortlist
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .retriever import search_profiles
from .schemas import (
    DEFAULT_PARSED_QUERY,
    ParsedQuery,
    RankedCandidate,
    RecruiterGraphState,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_ABOUT_PREVIEW_CHARS = 1400
MAX_RANKING_CANDIDATES = 12
FALLBACK_RANK_SCORE = 5.0


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")


def _build_llm() -> Optional[ChatOpenAI]:
    _load_environment()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        return None
    llm_model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    return ChatOpenAI(model=llm_model_name, temperature=0)


def _append_error(state: RecruiterGraphState, message: str) -> None:
    current_errors = state.get("error_messages") or []
    current_errors.append(message)
    state["error_messages"] = current_errors


def _ensure_string_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if isinstance(item, (str, int, float)) and str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def _coerce_parsed_query(raw_parsed: Dict[str, Any]) -> ParsedQuery:
    min_experience_raw = raw_parsed.get("min_experience_entries", 0)
    try:
        min_experience_value = max(0, int(min_experience_raw))
    except (TypeError, ValueError):
        min_experience_value = 0

    location_raw = raw_parsed.get("location", "")
    location_value = location_raw.strip() if isinstance(location_raw, str) else ""

    return {
        "role_keywords": _ensure_string_list(raw_parsed.get("role_keywords")),
        "skills": _ensure_string_list(raw_parsed.get("skills")),
        "location": location_value,
        "min_experience_entries": min_experience_value,
        "must_have_keywords": _ensure_string_list(raw_parsed.get("must_have_keywords")),
        "nice_to_have_keywords": _ensure_string_list(raw_parsed.get("nice_to_have_keywords")),
    }


def _heuristic_parse(question_text: str, min_experience_override: int) -> ParsedQuery:
    cleaned_question = (question_text or "").strip()
    role_keywords = [cleaned_question] if cleaned_question else []
    parsed: ParsedQuery = dict(DEFAULT_PARSED_QUERY)  # type: ignore[assignment]
    parsed["role_keywords"] = role_keywords
    parsed["min_experience_entries"] = max(0, int(min_experience_override or 0))
    return parsed


def parse_query_node(state: RecruiterGraphState) -> RecruiterGraphState:
    question_text = state.get("question_text", "").strip()
    min_experience_override = int(state.get("min_experience_entries") or 0)

    if not question_text:
        state["parsed_query"] = _heuristic_parse("", min_experience_override)
        _append_error(state, "Empty question provided; skipping LLM query parsing.")
        return state

    llm_model = _build_llm()
    if llm_model is None:
        state["parsed_query"] = _heuristic_parse(question_text, min_experience_override)
        _append_error(state, "OPENAI_API_KEY missing; using heuristic query parsing.")
        return state

    schema_example = {
        "role_keywords": ["<short role/title keywords>"],
        "skills": ["<specific skills or tools>"],
        "location": "<city/region/country or empty string>",
        "min_experience_entries": 0,
        "must_have_keywords": ["<terms that MUST appear>"],
        "nice_to_have_keywords": ["<terms that are bonuses>"],
    }

    prompt_text = (
        "You convert recruiter queries into structured JSON filters.\n"
        "Return ONLY valid JSON matching this schema (no prose, no markdown fences):\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        "Rules:\n"
        "- Keep lists short (<= 6 items) and specific.\n"
        "- Use lowercase strings.\n"
        "- If a field is unknown, return [] or an empty string or 0.\n"
        "- Do not invent constraints the user did not express.\n\n"
        f"Recruiter query: {question_text}"
    )

    try:
        model_response = llm_model.invoke(prompt_text)
        response_text = model_response.content if isinstance(model_response.content, str) else str(model_response.content)
        raw_parsed = json.loads(response_text)
        if not isinstance(raw_parsed, dict):
            raise ValueError("LLM did not return a JSON object.")
        parsed_query = _coerce_parsed_query(raw_parsed)
    except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
        _append_error(state, f"parse_query fallback: {parse_error}")
        parsed_query = _heuristic_parse(question_text, min_experience_override)

    if min_experience_override > parsed_query["min_experience_entries"]:
        parsed_query["min_experience_entries"] = min_experience_override

    state["parsed_query"] = parsed_query
    return state


def retrieve_candidates_node(state: RecruiterGraphState) -> RecruiterGraphState:
    parsed_query: ParsedQuery = state.get("parsed_query") or dict(DEFAULT_PARSED_QUERY)  # type: ignore[assignment]
    top_k_value = int(state.get("top_k") or 8)

    try:
        retrieved_profiles = search_profiles(
            role_keywords=parsed_query.get("role_keywords"),
            skills=parsed_query.get("skills"),
            location=parsed_query.get("location") or None,
            min_experience_entries=parsed_query.get("min_experience_entries", 0),
            must_have_keywords=parsed_query.get("must_have_keywords"),
            nice_to_have_keywords=parsed_query.get("nice_to_have_keywords"),
            top_k=top_k_value,
        )
    except Exception as retrieval_error:  # noqa: BLE001 - surface DB errors cleanly
        _append_error(state, f"retrieve_candidates failed: {retrieval_error}")
        retrieved_profiles = []

    if not retrieved_profiles and any(
        (
            parsed_query.get("must_have_keywords"),
            parsed_query.get("location"),
            parsed_query.get("min_experience_entries", 0) > 0,
        )
    ):
        _append_error(state, "No strict matches; relaxing must_have/location/min_experience filters.")
        try:
            retrieved_profiles = search_profiles(
                role_keywords=parsed_query.get("role_keywords"),
                skills=parsed_query.get("skills"),
                location=None,
                min_experience_entries=0,
                must_have_keywords=None,
                nice_to_have_keywords=(parsed_query.get("nice_to_have_keywords") or [])
                + (parsed_query.get("must_have_keywords") or []),
                top_k=top_k_value,
            )
        except Exception as retry_error:  # noqa: BLE001
            _append_error(state, f"relaxed retrieval failed: {retry_error}")
            retrieved_profiles = []

    state["candidate_profiles"] = retrieved_profiles
    return state


def _truncate_about(about_text: str) -> str:
    if not about_text:
        return ""
    if len(about_text) <= MAX_ABOUT_PREVIEW_CHARS:
        return about_text
    return about_text[:MAX_ABOUT_PREVIEW_CHARS].rstrip() + "..."


def _deterministic_rank(profile: Dict[str, Any]) -> RankedCandidate:
    relevance_value = int(profile.get("relevance_score") or 0)
    experience_value = int(profile.get("experience_count") or 0)
    skills_value = int(profile.get("skills_count") or 0)

    normalized_score = min(10.0, (relevance_value * 1.0) + (experience_value * 0.25) + (skills_value * 0.05))
    if normalized_score <= 0:
        normalized_score = FALLBACK_RANK_SCORE

    reasons = []
    if relevance_value > 0:
        reasons.append(f"Keyword relevance score {relevance_value}.")
    if experience_value > 0:
        reasons.append(f"{experience_value} experience entries.")
    if skills_value > 0:
        reasons.append(f"{skills_value} listed skills.")
    if not reasons:
        reasons.append("Baseline match; no structured signals found.")

    return {
        "profile_id": profile.get("profile_id", ""),
        "name": profile.get("name", ""),
        "headline": profile.get("headline", ""),
        "location": profile.get("location", ""),
        "about_text": profile.get("about_text", ""),
        "skills_count": skills_value,
        "experience_count": experience_value,
        "education_count": int(profile.get("education_count") or 0),
        "relevance_score": relevance_value,
        "rank_score": round(normalized_score, 2),
        "match_reasons": reasons,
        "risks": [],
    }


def rank_candidates_node(state: RecruiterGraphState) -> RecruiterGraphState:
    candidate_profiles = state.get("candidate_profiles") or []
    if not candidate_profiles:
        state["ranked_candidates"] = []
        return state

    capped_candidates = candidate_profiles[:MAX_RANKING_CANDIDATES]

    llm_model = _build_llm()
    if llm_model is None:
        _append_error(state, "OPENAI_API_KEY missing; using deterministic ranking.")
        state["ranked_candidates"] = [_deterministic_rank(profile) for profile in capped_candidates]
        return state

    parsed_query = state.get("parsed_query") or dict(DEFAULT_PARSED_QUERY)

    ranking_payload = [
        {
            "profile_id": profile.get("profile_id"),
            "name": profile.get("name"),
            "headline": profile.get("headline"),
            "location": profile.get("location"),
            "skills_count": profile.get("skills_count"),
            "experience_count": profile.get("experience_count"),
            "education_count": profile.get("education_count"),
            "about_text": _truncate_about(profile.get("about_text", "")),
        }
        for profile in capped_candidates
    ]

    schema_example = [
        {
            "profile_id": "<string>",
            "rank_score": 0.0,
            "match_reasons": ["<short bullet>"],
            "risks": ["<optional short bullet>"],
        }
    ]

    prompt_text = (
        "You are a senior recruiter scoring LinkedIn candidates against a role brief.\n"
        "Return ONLY valid JSON: a list of objects following this schema (no prose):\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        "Rules:\n"
        "- rank_score is 0-10 (higher = better fit). Use decimals.\n"
        "- Keep match_reasons <= 3 bullets, grounded in the provided about_text/headline.\n"
        "- Include risks ONLY if concrete (missing skill, location mismatch, seniority gap); otherwise [].\n"
        "- Keep the same number of items you were given and keep profile_id exact.\n\n"
        f"Role brief (user query): {state.get('question_text', '')}\n\n"
        f"Parsed filters: {json.dumps(parsed_query, ensure_ascii=False)}\n\n"
        f"Candidates JSON:\n{json.dumps(ranking_payload, ensure_ascii=False)}\n"
    )

    try:
        model_response = llm_model.invoke(prompt_text)
        response_text = model_response.content if isinstance(model_response.content, str) else str(model_response.content)
        raw_rankings = json.loads(response_text)
        if not isinstance(raw_rankings, list):
            raise ValueError("LLM ranking response was not a JSON list.")
    except (json.JSONDecodeError, ValueError, TypeError) as rank_error:
        _append_error(state, f"rank_candidates fallback: {rank_error}")
        state["ranked_candidates"] = [_deterministic_rank(profile) for profile in capped_candidates]
        return state

    ranking_by_profile: Dict[str, Dict[str, Any]] = {}
    for ranking_entry in raw_rankings:
        if not isinstance(ranking_entry, dict):
            continue
        profile_key = ranking_entry.get("profile_id")
        if isinstance(profile_key, str) and profile_key:
            ranking_by_profile[profile_key] = ranking_entry

    ranked_candidates: List[RankedCandidate] = []
    for profile in capped_candidates:
        profile_id_value = profile.get("profile_id", "")
        llm_entry = ranking_by_profile.get(profile_id_value)
        if llm_entry is None:
            ranked_candidates.append(_deterministic_rank(profile))
            continue

        try:
            rank_score_value = float(llm_entry.get("rank_score", FALLBACK_RANK_SCORE))
        except (TypeError, ValueError):
            rank_score_value = FALLBACK_RANK_SCORE
        rank_score_value = max(0.0, min(10.0, rank_score_value))

        ranked_candidates.append(
            {
                "profile_id": profile_id_value,
                "name": profile.get("name", ""),
                "headline": profile.get("headline", ""),
                "location": profile.get("location", ""),
                "about_text": profile.get("about_text", ""),
                "skills_count": int(profile.get("skills_count") or 0),
                "experience_count": int(profile.get("experience_count") or 0),
                "education_count": int(profile.get("education_count") or 0),
                "relevance_score": int(profile.get("relevance_score") or 0),
                "rank_score": round(rank_score_value, 2),
                "match_reasons": _ensure_string_list(llm_entry.get("match_reasons")),
                "risks": _ensure_string_list(llm_entry.get("risks")),
            }
        )

    ranked_candidates.sort(key=lambda candidate: candidate["rank_score"], reverse=True)
    state["ranked_candidates"] = ranked_candidates
    return state


def synthesize_shortlist_node(state: RecruiterGraphState) -> RecruiterGraphState:
    ranked_candidates = state.get("ranked_candidates") or []
    if not ranked_candidates:
        state["shortlist_summary"] = "No candidates matched the current filters."
        return state

    top_candidates = ranked_candidates[:5]
    top_profile_ids = ", ".join(candidate["profile_id"] for candidate in top_candidates if candidate.get("profile_id"))

    llm_model = _build_llm()
    if llm_model is None:
        deterministic_lines = [
            f"{idx + 1}. {candidate.get('name') or candidate['profile_id']} "
            f"({candidate['profile_id']}) - score {candidate['rank_score']}"
            for idx, candidate in enumerate(top_candidates)
        ]
        deterministic_lines.append(f"Sources: {top_profile_ids}")
        state["shortlist_summary"] = "\n".join(deterministic_lines)
        _append_error(state, "OPENAI_API_KEY missing; produced deterministic shortlist summary.")
        return state

    summary_payload = [
        {
            "profile_id": candidate["profile_id"],
            "name": candidate["name"],
            "headline": candidate["headline"],
            "location": candidate["location"],
            "rank_score": candidate["rank_score"],
            "match_reasons": candidate["match_reasons"],
            "risks": candidate["risks"],
        }
        for candidate in top_candidates
    ]

    prompt_text = (
        "You are a recruiter summarizing a shortlist for a hiring manager.\n"
        "Write a concise 4-6 sentence summary that: (a) names the top 3 candidates with profile_id citations, "
        "(b) highlights why each fits, (c) notes any shared risks.\n"
        "End with a line 'Sources: <comma-separated profile_ids used>'.\n"
        "Do not invent details not in the JSON.\n\n"
        f"Role brief: {state.get('question_text', '')}\n\n"
        f"Shortlist JSON:\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}\n"
    )

    try:
        model_response = llm_model.invoke(prompt_text)
        summary_text = model_response.content if isinstance(model_response.content, str) else str(model_response.content)
    except Exception as summary_error:  # noqa: BLE001 - surface in UI
        _append_error(state, f"synthesize_shortlist fallback: {summary_error}")
        summary_text = f"Shortlist (deterministic fallback). Sources: {top_profile_ids}"

    state["shortlist_summary"] = summary_text.strip()
    return state


def build_graph() -> Any:
    graph_builder = StateGraph(RecruiterGraphState)
    graph_builder.add_node("parse_query", parse_query_node)
    graph_builder.add_node("retrieve_candidates", retrieve_candidates_node)
    graph_builder.add_node("rank_candidates", rank_candidates_node)
    graph_builder.add_node("synthesize_shortlist", synthesize_shortlist_node)

    graph_builder.set_entry_point("parse_query")
    graph_builder.add_edge("parse_query", "retrieve_candidates")
    graph_builder.add_edge("retrieve_candidates", "rank_candidates")
    graph_builder.add_edge("rank_candidates", "synthesize_shortlist")
    graph_builder.add_edge("synthesize_shortlist", END)

    return graph_builder.compile()


def run_recruiter_search(
    question_text: str,
    top_k: int = 8,
    min_experience_entries: int = 0,
) -> Dict[str, Any]:
    compiled_graph = build_graph()
    initial_state: RecruiterGraphState = {
        "question_text": question_text,
        "top_k": top_k,
        "min_experience_entries": min_experience_entries,
        "parsed_query": dict(DEFAULT_PARSED_QUERY),  # type: ignore[typeddict-item]
        "candidate_profiles": [],
        "ranked_candidates": [],
        "shortlist_summary": "",
        "error_messages": [],
    }
    final_state: RecruiterGraphState = compiled_graph.invoke(initial_state)
    return {
        "parsed_query": final_state.get("parsed_query", dict(DEFAULT_PARSED_QUERY)),
        "candidate_profiles": final_state.get("candidate_profiles", []),
        "ranked_candidates": final_state.get("ranked_candidates", []),
        "shortlist_summary": final_state.get("shortlist_summary", ""),
        "error_messages": final_state.get("error_messages", []),
    }


def run_profile_question(question_text: str, top_k: int = 8) -> Dict[str, Any]:
    """Backward-compatible wrapper used by the CLI."""
    result = run_recruiter_search(question_text=question_text, top_k=top_k)
    return {
        "answer_text": result["shortlist_summary"],
        "candidate_profiles": result["ranked_candidates"] or result["candidate_profiles"],
        "parsed_query": result["parsed_query"],
        "error_messages": result["error_messages"],
    }
