"""Recruiter LangGraph workflow over linkedin_api_profiles_parsed.

Graph nodes:
    parse_query -> retrieve_candidates -> rank_candidates -> synthesize_shortlist
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from . import weights_loader
from .retriever import search_profiles
from .schemas import (
    DEFAULT_PARSED_QUERY,
    DimensionBreakdown,
    ParsedQuery,
    RankedCandidate,
    RecruiterGraphState,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_ABOUT_PREVIEW_CHARS = 1400
MAX_RANKING_CANDIDATES = 12
FALLBACK_RANK_SCORE = 5.0

DIMENSION_KEYS = (
    "phd_researcher",
    "sf_location_fit",
    "technical_background",
    "education_prestige",
    "founder_experience",
)

DIMENSION_LABELS: Dict[str, str] = {
    "phd_researcher": "PhD / Researcher",
    "sf_location_fit": "SF location fit",
    "technical_background": "Technical background",
    "education_prestige": "Education prestige",
    "founder_experience": "Founder experience",
}

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "technical_background": 0.30,
    "founder_experience": 0.25,
    "phd_researcher": 0.15,
    "education_prestige": 0.15,
    "sf_location_fit": 0.15,
}

# Resolved at import time: reads `config/weights.json` if present and valid,
# otherwise falls back to DEFAULT_DIMENSION_WEIGHTS. Callers that need the
# baseline (e.g. the calibrator) can still reach DEFAULT_DIMENSION_WEIGHTS.
DIMENSION_WEIGHTS: Dict[str, float] = weights_loader.load_weights(DEFAULT_DIMENSION_WEIGHTS)

HIGH_WEIGHT_RISK_THRESHOLD = 0.20
LOW_SCORE_RISK_THRESHOLD = 2.0
STRONG_DIMENSION_THRESHOLD = 4.0

SF_BAY_LOCATION_TOKENS = (
    "san francisco", "sf bay", "bay area", "south bay", "east bay", "peninsula",
    "palo alto", "mountain view", "menlo park", "sunnyvale", "oakland", "berkeley",
    "san mateo", "redwood city", "cupertino", "san jose", "daly city", "foster city",
)
CA_TECH_TOKENS = (
    "california", ", ca", " ca,", " ca ", "ca usa",
    "los angeles", "san diego", "sacramento", "santa monica", "irvine",
)
US_TECH_HUB_TOKENS = (
    "new york", "nyc", ", ny", "seattle", "austin", "boston", "denver", "chicago",
    "washington, d.c.", "washington dc", "atlanta", "portland, or",
)
US_GENERIC_TOKENS = ("united states", "usa", "u.s.", "remote")

PHD_TITLE_TOKENS = ("phd", "ph.d", "ph. d", "doctor of philosophy", "doctorate", "d.phil")
RESEARCH_TITLE_TOKENS = (
    "research scientist", "research engineer", "research fellow", "postdoc",
    "post-doctoral", "post doctoral", "principal investigator", "phd candidate",
    "phd student", "doctoral candidate", "doctoral student",
)
PUBLICATION_VENUE_TOKENS = (
    "arxiv", "neurips", "icml", "cvpr", "acl", "emnlp", "iclr", "kdd",
    "published in", "publication", "co-authored", "ieee ", " acm ", "nature ",
)

FOUNDER_TITLE_TOKENS = (
    "founder", "co-founder", "cofounder", "founding engineer", "founding member",
    "founding team", "ceo", "cto", "chief executive", "chief technology",
    "chief operating", "coo ",
)
FOUNDER_CONTEXT_TOKENS = (
    "y combinator", " yc ", "techstars", "seed round", "series a", "series b",
    "raised $", "raised seed", "acquired by", "acquihire", "exited",
    "bootstrapped", "startup i founded", "startup i co-founded",
)

TECHNICAL_TITLE_TOKENS = (
    "software engineer", "software developer", "backend engineer", "frontend engineer",
    "full stack", "fullstack", "data scientist", "ml engineer", "machine learning",
    "applied scientist", "research engineer", "research scientist", "staff engineer",
    "principal engineer", "distinguished engineer", "devops", "sre ",
    "site reliability", "platform engineer", "infrastructure engineer",
    "artificial intelligence",
)
TECHNICAL_SKILL_TOKENS = (
    "python", "pytorch", "tensorflow", "jax", "kubernetes", " aws ", " gcp ",
    " azure ", "c++", "rust", "golang", "java ", "scala", "sql", "react", "node.js",
    "node ", "spark", " llm", " nlp", " cv ", "computer vision", "transformer",
    "typescript",
)

TOP_SCHOOL_TOKENS = (
    "stanford", "mit ", "massachusetts institute", "harvard", "princeton", "yale",
    "uc berkeley", "u.c. berkeley", "berkeley", "caltech", "carnegie mellon", "cmu ",
    "cornell", "columbia university", "university of chicago", "upenn",
    "university of pennsylvania", "johns hopkins", "oxford", "cambridge",
    "eth zurich", "eth zürich", "imperial college london", "tsinghua",
    "peking university", "iit ", "indian institute of technology",
)
STRONG_SCHOOL_TOKENS = (
    "university of california", "ucla", "ucsd", "ucsb", "georgia tech",
    "university of washington", "university of michigan", "ut austin",
    "university of texas", "uiuc", "illinois urbana", "purdue", "duke",
    "northwestern", "nyu ", "university of southern california", "usc ",
    "university of toronto", "mcgill", "waterloo",
)
MASTERS_DEGREE_TOKENS = ("master of", "master's", "masters", "m.s.", "m.sc", "msc ", "m.eng")
BACHELORS_DEGREE_TOKENS = ("bachelor of", "bachelor's", "bachelors", "b.s.", "b.sc", "bsc ", "b.eng", "undergrad")


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


def _combined_profile_text(profile: Dict[str, Any]) -> str:
    """Lowercased concatenation of fields used for keyword-based signal detection."""
    text_parts = [
        str(profile.get("headline") or ""),
        str(profile.get("about_text") or ""),
        str(profile.get("experience_json") or ""),
        str(profile.get("education_json") or ""),
    ]
    return " \n ".join(part.lower() for part in text_parts if part)


def _any_token_present(haystack: str, tokens: Sequence[str]) -> bool:
    return any(token in haystack for token in tokens)


def _count_tokens_present(haystack: str, tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if token in haystack)


def _clip_score(value: float) -> float:
    return max(0.0, min(10.0, float(value)))


def _score_phd_researcher(profile: Dict[str, Any]) -> tuple:
    combined = _combined_profile_text(profile)
    score = 0.0
    reasons: List[str] = []

    if _any_token_present(combined, PHD_TITLE_TOKENS):
        score += 7.0
        reasons.append("PhD/doctorate mentioned")
    research_hits = _count_tokens_present(combined, RESEARCH_TITLE_TOKENS)
    if research_hits:
        score += min(2.0, research_hits * 1.0)
        reasons.append(f"{research_hits} research role/title signal(s)")
    publication_hits = _count_tokens_present(combined, PUBLICATION_VENUE_TOKENS)
    if publication_hits:
        score += min(2.0, publication_hits * 0.6)
        reasons.append(f"{publication_hits} publication/venue signal(s)")

    return _clip_score(score), "; ".join(reasons) or "no PhD/research signal found"


def _score_sf_location_fit(profile: Dict[str, Any]) -> tuple:
    raw_location = (profile.get("location") or "").strip()
    location_lower = raw_location.lower()
    about_lower = (profile.get("about_text") or "").lower()

    if _any_token_present(location_lower, SF_BAY_LOCATION_TOKENS):
        return 10.0, f"SF/Bay Area location: '{raw_location}'"
    if _any_token_present(location_lower, CA_TECH_TOKENS):
        return 7.0, f"California tech region: '{raw_location}'"
    if _any_token_present(location_lower, US_TECH_HUB_TOKENS):
        return 5.0, f"US tech hub: '{raw_location}'"
    if _any_token_present(location_lower, US_GENERIC_TOKENS) or _any_token_present(about_lower, ("remote (us)", "based in the us")):
        return 4.0, f"US/remote: '{raw_location}'"
    if raw_location:
        return 2.0, f"Out-of-market: '{raw_location}'"
    return 1.0, "no location data"


def _score_technical_background(profile: Dict[str, Any]) -> tuple:
    combined = _combined_profile_text(profile)
    skills_count_value = int(profile.get("skills_count") or 0)
    score = 0.0
    reasons: List[str] = []

    title_hits = _count_tokens_present(combined, TECHNICAL_TITLE_TOKENS)
    skill_hits = _count_tokens_present(combined, TECHNICAL_SKILL_TOKENS)

    if title_hits:
        score += min(5.0, title_hits * 1.25)
        reasons.append(f"{title_hits} technical title hit(s)")
    if skill_hits:
        score += min(4.0, skill_hits * 0.8)
        reasons.append(f"{skill_hits} technical stack hit(s)")
    if skills_count_value >= 20:
        score += 1.0
        reasons.append(f"{skills_count_value} listed skills")
    elif skills_count_value >= 10:
        score += 0.5

    return _clip_score(score), "; ".join(reasons) or "no technical signal found"


def _score_education_prestige(profile: Dict[str, Any]) -> tuple:
    education_blob = (profile.get("education_json") or "").lower()
    about_blob = (profile.get("about_text") or "").lower()
    combined = education_blob + " \n " + about_blob
    education_count_value = int(profile.get("education_count") or 0)

    score = 0.0
    reasons: List[str] = []

    if _any_token_present(combined, TOP_SCHOOL_TOKENS):
        score += 7.0
        reasons.append("top-tier school match")
    elif _any_token_present(combined, STRONG_SCHOOL_TOKENS):
        score += 5.0
        reasons.append("strong school match")

    if _any_token_present(combined, PHD_TITLE_TOKENS):
        score += 2.0
        reasons.append("doctoral degree")
    elif _any_token_present(combined, MASTERS_DEGREE_TOKENS):
        score += 1.5
        reasons.append("master's degree")
    elif _any_token_present(combined, BACHELORS_DEGREE_TOKENS):
        score += 0.5
        reasons.append("bachelor's degree")

    if score == 0.0 and education_count_value > 0:
        score = 2.0
        reasons.append(f"{education_count_value} education entry(ies), no prestige match")

    return _clip_score(score), "; ".join(reasons) or "no education data"


def _score_founder_experience(profile: Dict[str, Any]) -> tuple:
    combined = _combined_profile_text(profile)
    score = 0.0
    reasons: List[str] = []

    title_hits = _count_tokens_present(combined, FOUNDER_TITLE_TOKENS)
    context_hits = _count_tokens_present(combined, FOUNDER_CONTEXT_TOKENS)

    if title_hits:
        score += min(7.0, title_hits * 2.5)
        reasons.append(f"{title_hits} founder/exec title hit(s)")
    if context_hits:
        score += min(3.0, context_hits * 1.0)
        reasons.append(f"{context_hits} startup context signal(s)")

    return _clip_score(score), "; ".join(reasons) or "no founder signal found"


_DIMENSION_SCORERS = (
    ("phd_researcher", _score_phd_researcher),
    ("sf_location_fit", _score_sf_location_fit),
    ("technical_background", _score_technical_background),
    ("education_prestige", _score_education_prestige),
    ("founder_experience", _score_founder_experience),
)


def _compute_dimension_scores(profile: Dict[str, Any]) -> tuple:
    """Compute all 5 sub-scores + short evidence reason per dimension."""
    dim_scores: Dict[str, float] = {}
    dim_reasons: Dict[str, str] = {}
    for dimension_key, scorer in _DIMENSION_SCORERS:
        score_value, reason_text = scorer(profile)
        dim_scores[dimension_key] = round(float(score_value), 2)
        dim_reasons[dimension_key] = reason_text
    assert set(dim_scores.keys()) == set(DIMENSION_KEYS), "dimension scores must cover all DIMENSION_KEYS"
    return dim_scores, dim_reasons


def _aggregate_rank_score(dimension_scores: Dict[str, float]) -> float:
    """Weighted aggregate of the 5 sub-scores, clipped to [0, 10]."""
    total_weighted = 0.0
    for dimension_key, weight in DIMENSION_WEIGHTS.items():
        total_weighted += float(dimension_scores.get(dimension_key, 0.0)) * weight
    return round(_clip_score(total_weighted), 2)


def _deterministic_rank(profile: Dict[str, Any]) -> RankedCandidate:
    """Baseline ranking built entirely from structured signals (no LLM).

    Computes the 5 dimension sub-scores, aggregates them using DIMENSION_WEIGHTS,
    and surfaces strong dimensions as match_reasons and weak high-weight dimensions
    as risks. Also serves as the fallback when the LLM path fails.
    """
    relevance_value = int(profile.get("relevance_score") or 0)
    experience_value = int(profile.get("experience_count") or 0)
    skills_value = int(profile.get("skills_count") or 0)
    education_value = int(profile.get("education_count") or 0)

    dim_scores, dim_reasons = _compute_dimension_scores(profile)
    overall_score = _aggregate_rank_score(dim_scores)

    match_reasons: List[str] = [
        f"{DIMENSION_LABELS[key]}: {dim_scores[key]}/10 - {dim_reasons[key]}"
        for key in DIMENSION_KEYS
        if dim_scores[key] >= STRONG_DIMENSION_THRESHOLD
    ]
    if not match_reasons:
        match_reasons = [
            f"{DIMENSION_LABELS[key]}: {dim_scores[key]}/10"
            for key in DIMENSION_KEYS
        ]

    risks: List[str] = [
        f"Low {DIMENSION_LABELS[key]} ({dim_scores[key]}/10)"
        for key in DIMENSION_KEYS
        if dim_scores[key] <= LOW_SCORE_RISK_THRESHOLD
        and DIMENSION_WEIGHTS[key] >= HIGH_WEIGHT_RISK_THRESHOLD
    ]

    if overall_score <= 0:
        overall_score = FALLBACK_RANK_SCORE

    return {
        "profile_id": profile.get("profile_id", ""),
        "name": profile.get("name", ""),
        "headline": profile.get("headline", ""),
        "location": profile.get("location", ""),
        "about_text": profile.get("about_text", ""),
        "skills_count": skills_value,
        "experience_count": experience_value,
        "education_count": education_value,
        "relevance_score": relevance_value,
        "rank_score": overall_score,
        "match_reasons": match_reasons,
        "risks": risks,
        "dimension_scores": dim_scores,  # type: ignore[typeddict-item]
        "dimension_reasons": dim_reasons,
    }


def _merge_llm_dimension_ranking(
    baseline_candidate: RankedCandidate,
    llm_entry: Dict[str, Any],
) -> RankedCandidate:
    """Overlay LLM-refined dimension scores/reasons onto the deterministic baseline.

    - Unknown/invalid per-dimension values are dropped (baseline retained).
    - Final rank_score is re-aggregated from the merged dimension scores, so the
      overall score is always consistent with its parts.
    """
    merged_scores: Dict[str, float] = dict(baseline_candidate["dimension_scores"])  # type: ignore[arg-type]
    merged_reasons: Dict[str, str] = dict(baseline_candidate["dimension_reasons"])

    llm_scores_raw = llm_entry.get("dimension_scores")
    if isinstance(llm_scores_raw, dict):
        for dimension_key in DIMENSION_KEYS:
            if dimension_key not in llm_scores_raw:
                continue
            try:
                numeric_value = float(llm_scores_raw[dimension_key])
            except (TypeError, ValueError):
                continue
            merged_scores[dimension_key] = round(_clip_score(numeric_value), 2)

    llm_reasons_raw = llm_entry.get("dimension_reasons")
    if isinstance(llm_reasons_raw, dict):
        for dimension_key in DIMENSION_KEYS:
            reason_value = llm_reasons_raw.get(dimension_key)
            if isinstance(reason_value, str) and reason_value.strip():
                merged_reasons[dimension_key] = reason_value.strip()

    refined_overall = _aggregate_rank_score(merged_scores)

    match_reasons = _ensure_string_list(llm_entry.get("match_reasons")) or baseline_candidate["match_reasons"]
    risks = _ensure_string_list(llm_entry.get("risks"))

    refined_candidate: RankedCandidate = {
        **baseline_candidate,  # type: ignore[misc]
        "rank_score": refined_overall,
        "match_reasons": match_reasons,
        "risks": risks,
        "dimension_scores": merged_scores,  # type: ignore[typeddict-item]
        "dimension_reasons": merged_reasons,
    }
    return refined_candidate


def rank_candidates_node(state: RecruiterGraphState) -> RecruiterGraphState:
    candidate_profiles = state.get("candidate_profiles") or []
    if not candidate_profiles:
        state["ranked_candidates"] = []
        return state

    capped_candidates = candidate_profiles[:MAX_RANKING_CANDIDATES]

    # Deterministic 5-dimension baseline for every candidate (always available as fallback).
    baseline_by_profile_id: Dict[str, RankedCandidate] = {}
    for profile in capped_candidates:
        baseline_candidate = _deterministic_rank(profile)
        baseline_by_profile_id[baseline_candidate["profile_id"]] = baseline_candidate

    llm_model = _build_llm()
    if llm_model is None:
        _append_error(state, "OPENAI_API_KEY missing; using deterministic 5-dimension ranking.")
        ranked_list = sorted(baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True)
        state["ranked_candidates"] = ranked_list
        return state

    parsed_query = state.get("parsed_query") or dict(DEFAULT_PARSED_QUERY)

    ranking_payload = [
        {
            "profile_id": baseline_by_profile_id[profile.get("profile_id", "")]["profile_id"],
            "name": profile.get("name"),
            "headline": profile.get("headline"),
            "location": profile.get("location"),
            "skills_count": profile.get("skills_count"),
            "experience_count": profile.get("experience_count"),
            "education_count": profile.get("education_count"),
            "about_text": _truncate_about(profile.get("about_text", "")),
            "baseline_dimension_scores": baseline_by_profile_id[profile.get("profile_id", "")]["dimension_scores"],
            "baseline_dimension_reasons": baseline_by_profile_id[profile.get("profile_id", "")]["dimension_reasons"],
        }
        for profile in capped_candidates
        if profile.get("profile_id", "") in baseline_by_profile_id
    ]

    schema_example = [
        {
            "profile_id": "<string>",
            "dimension_scores": {
                "phd_researcher": 0.0,
                "sf_location_fit": 0.0,
                "technical_background": 0.0,
                "education_prestige": 0.0,
                "founder_experience": 0.0,
            },
            "dimension_reasons": {
                "phd_researcher": "<evidence-based short reason>",
                "sf_location_fit": "<evidence-based short reason>",
                "technical_background": "<evidence-based short reason>",
                "education_prestige": "<evidence-based short reason>",
                "founder_experience": "<evidence-based short reason>",
            },
            "match_reasons": ["<short bullet>"],
            "risks": ["<optional short bullet>"],
        }
    ]

    prompt_text = (
        "You are a senior recruiter scoring candidates for a fast-growing SF-based startup.\n"
        "For each candidate score 5 dimensions (0-10, decimals ok):\n"
        "  - phd_researcher: doctoral training, research roles, publications.\n"
        "  - sf_location_fit: fit for an SF-based founding/early team. "
        "SF/Bay Area ~10, California tech ~7, major US tech hub ~5, US/remote ~4, else 1-3.\n"
        "  - technical_background: hands-on technical strength (engineer/ML/research titles, stack depth).\n"
        "  - education_prestige: prestige and rigor of highest education.\n"
        "  - founder_experience: prior founder, founding-team, or executive experience at startups.\n\n"
        "Use `baseline_dimension_scores` as your starting point. Adjust ONLY when about_text/headline "
        "clearly supports a different value. If there is no supporting evidence, keep the baseline.\n\n"
        "Return ONLY valid JSON: a list of objects following this schema (no prose, no markdown):\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        "Rules:\n"
        "- Keep profile_id exact; include one object per candidate; same count in as out.\n"
        "- Each dimension_reasons value <= 140 chars, grounded in the provided fields.\n"
        "- match_reasons <= 3 bullets; risks empty [] unless concrete (missing skill, location mismatch, seniority gap).\n"
        "- Do NOT output an overall rank_score; it will be re-aggregated from dimension_scores using fixed weights.\n\n"
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
        ranked_list = sorted(baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True)
        state["ranked_candidates"] = ranked_list
        return state

    for ranking_entry in raw_rankings:
        if not isinstance(ranking_entry, dict):
            continue
        profile_key = ranking_entry.get("profile_id")
        if not isinstance(profile_key, str) or profile_key not in baseline_by_profile_id:
            continue
        baseline_by_profile_id[profile_key] = _merge_llm_dimension_ranking(
            baseline_by_profile_id[profile_key],
            ranking_entry,
        )

    ranked_candidates = sorted(baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True)
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
