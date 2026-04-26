"""Recruiter LangGraph workflow over linkedin_api_profiles_parsed.

Graph nodes (Round 1):
    parse_query
        -> retrieve_candidates
            -> (conditional) enrich_low_info --> rank_candidates
            -> (conditional)                  --> rank_candidates
    rank_candidates (pointwise LLM + Stage-2 listwise rerank)
        -> synthesize_shortlist

LangChain / LangGraph idioms used:
- `llm.with_structured_output(PydanticModel)` in parse_query + rank_candidates
- `StateGraph.add_conditional_edges(...)` around retrieve_candidates
- `@traceable` from langsmith on every node + the top-level entry point
- Deterministic fallbacks at every node so missing OPENAI_API_KEY or
  validation errors degrade gracefully rather than failing the run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from . import weights_loader
from .embeddings_index import semantic_search
from .retriever import fetch_profiles_by_ids, search_profiles
from .schemas import (
    DEFAULT_PARSED_QUERY,
    DimensionBreakdown,
    DimensionRankingItem,
    DimensionRankingResponse,
    ListwiseRerankResponse,
    PairwiseDecision,
    ParsedQuery,
    ParsedQueryModel,
    RankedCandidate,
    RecruiterGraphState,
)

try:
    from langsmith import traceable  # type: ignore[import-not-found]
    from langsmith.run_helpers import get_current_run_tree  # type: ignore[import-not-found]
    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    _LANGSMITH_AVAILABLE = False

    def traceable(*_decorator_args: Any, **_decorator_kwargs: Any) -> Callable[..., Any]:
        """No-op fallback when langsmith isn't installed."""
        def _wrap(target_callable: Callable[..., Any]) -> Callable[..., Any]:
            return target_callable

        return _wrap

    def get_current_run_tree() -> Optional[Any]:  # pragma: no cover
        return None

DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_ABOUT_PREVIEW_CHARS = 1400
MAX_RANKING_CANDIDATES = 12
FALLBACK_RANK_SCORE = 5.0

# Round 1 constants
LOW_INFO_ABOUT_THRESHOLD = 200  # char count below which we enrich from exp/edu JSON
ENRICHED_ABOUT_MAX_CHARS = 1200  # cap on the synthetic about_text_enriched field
LISTWISE_RERANK_TOP_K = 5        # Stage-2 listwise rerank operates on top-K only
EVIDENCE_FREE_DELTA_CAP = 3.0    # reject LLM dim deltas > this when no evidence quoted
SEMANTIC_RECALL_TOP_K = 8        # embedding-search depth per retrieval call

# Stage-3 pairwise tie-break: when adjacent candidates' rank_scores are within
# PAIRWISE_TIEBREAK_THRESHOLD of each other, ask the LLM which is better. Only
# swap if the LLM's confidence clears PAIRWISE_TIEBREAK_MIN_CONFIDENCE.
# Capped to PAIRWISE_TIEBREAK_MAX_CALLS per query so a long run of near-ties
# can't explode the LLM budget.
PAIRWISE_TIEBREAK_THRESHOLD = 0.3
PAIRWISE_TIEBREAK_MIN_CONFIDENCE = 0.6
PAIRWISE_TIEBREAK_MAX_CALLS = 3
PAIRWISE_TIEBREAK_WINDOW = LISTWISE_RERANK_TOP_K  # only tie-break within top-K

# Round-3 token/cost counter. Prices are USD per 1K tokens, pulled from the
# OpenAI pricing page for gpt-4o-mini as of 2025-04. Not a billing source of
# truth; used only to surface a ballpark "this query cost ~$X" figure in the
# Streamlit sidebar. If you change models via OPENAI_MODEL, update this map
# or accept that the cost column will stay at $0.00.
TOKEN_PRICE_USD_PER_1K: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),       # (prompt, completion)
    "gpt-4o": (0.005, 0.015),
    "gpt-4.1-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
}


class TokenUsageCollector(BaseCallbackHandler):
    """LangChain callback that accumulates token usage across LLM invocations.

    Attached to ChatOpenAI via `config={"callbacks": [...]}` on each
    `.invoke(...)`. Reads either the legacy `response.llm_output["token_usage"]`
    shape or the newer `message.usage_metadata` shape so we don't break
    whichever langchain_openai revision is installed.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_calls: int = 0

    def on_llm_end(self, response: LLMResult, **_kwargs: Any) -> None:
        self.llm_calls += 1
        usage_dict = (response.llm_output or {}).get("token_usage") if response.llm_output else None
        if isinstance(usage_dict, dict):
            self.prompt_tokens += int(usage_dict.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage_dict.get("completion_tokens", 0) or 0)
            self.total_tokens += int(usage_dict.get("total_tokens", 0) or 0)
            return
        for generation_list in response.generations or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None) if message else None
                if isinstance(usage_metadata, dict):
                    self.prompt_tokens += int(usage_metadata.get("input_tokens", 0) or 0)
                    self.completion_tokens += int(usage_metadata.get("output_tokens", 0) or 0)
                    self.total_tokens += int(usage_metadata.get("total_tokens", 0) or 0)
                    return

    def estimated_cost_usd(self, model_name: str) -> float:
        prices = TOKEN_PRICE_USD_PER_1K.get(model_name)
        if not prices:
            return 0.0
        prompt_price, completion_price = prices
        return (
            (self.prompt_tokens / 1000.0) * prompt_price
            + (self.completion_tokens / 1000.0) * completion_price
        )


# Module-level list of active collectors. Each entry is attached to every
# LLM `.invoke(...)` inside the graph for the duration of a search. List,
# not a single slot, so nested/overlapping runs don't clobber each other —
# each run_recruiter_search call manages its own entry.
_ACTIVE_LLM_CALLBACKS: List[BaseCallbackHandler] = []


def _llm_invoke_config() -> Dict[str, Any]:
    """Returns a config dict to pass as the second arg of `.invoke(...)`.

    Empty dict when no collector is registered, so the overhead for the
    non-Streamlit smoke path is zero.
    """
    if not _ACTIVE_LLM_CALLBACKS:
        return {}
    return {"callbacks": list(_ACTIVE_LLM_CALLBACKS)}

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

# Per-dimension affine transform fitted by `scripts/calibrate.py`:
#   adjusted_dim_i = clip(gain_i * raw_dim_i + bias_i, 0, 10)
# Identity transform (gain=1, bias=0) per dimension when no calibration has
# been run, so behavior is unchanged unless calibration writes new values.
DEFAULT_DIMENSION_GAINS: Dict[str, Dict[str, float]] = weights_loader.default_dimension_gains(
    DEFAULT_DIMENSION_WEIGHTS
)
DIMENSION_GAINS: Dict[str, Dict[str, float]] = weights_loader.load_gains(DEFAULT_DIMENSION_WEIGHTS)

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

# --- Round 2 scorer fix tokens ---

# Phrases that, when within PHD_PROXIMITY_WINDOW chars of a PhD mention, mean
# the candidate is not a graduated PhD. E.g. "phd student" (still in program),
# "dropped out of the phd" (never finished), etc.
PHD_DISQUALIFIER_TOKENS = (
    "student", "candidate", "dropped out", "dropout", "unfinished",
    "did not complete", "never completed", "leave of absence",
)
PHD_PROXIMITY_WINDOW = 30
PHD_DISQUALIFIED_SCORE_CAP = 3.0

# Skill hedges: when these appear shortly BEFORE a technical skill token, count
# the skill at half weight. "Familiar with Kubernetes" should not read the same
# as "Kubernetes architect".
SKILL_HEDGE_TOKENS = (
    "familiar with", "learning", "exploring", "dabbled in", "some exposure to",
    "introductory", "reading about", "interested in",
)
SKILL_HEDGE_WINDOW = 40

# Graduation signals: at least one of these must appear for a top-tier school
# match to be trusted. Without one the bonus is halved.
GRADUATION_SIGNAL_TOKENS = (
    "graduated", "class of", "alumni", "alumnus", "alumna", "alum ",
    "b.s.", "b.sc", "bachelor of", "bachelor's", "m.s.", "m.sc",
    "master of", "master's", "phd", "ph.d", "doctorate", "j.d.", "mba",
)
DROPOUT_TOKENS = ("dropped out", "dropout", "did not complete", "never completed")
DROPOUT_PROXIMITY_WINDOW = 50
DROPOUT_SCHOOL_SCORE_CAP = 2.0

# SF-positive signals in about_text that override a non-SF `location` field.
# E.g. candidate is currently in NY but about_text says "moving to SF".
SF_RELOCATION_TOKENS = (
    "based in sf", "based in san francisco", "moving to sf",
    "moving to san francisco", "relocating to sf", "relocating to san francisco",
    "relocating to the bay area", "moving to the bay area",
)
SF_OPEN_TO_RELOCATE_TOKENS = (
    "open to relocate", "open to relocation", "willing to relocate",
    "happy to relocate", "can relocate", "relocate to sf",
)


def _word_boundary_patterns(tokens: Sequence[str]) -> Tuple[Pattern[str], ...]:
    """Compile case-insensitive word-boundary regex for each token.

    `re.escape` handles punctuation like `.` in `ph.d`. Whitespace-only tokens
    are skipped. Tokens are stripped so a token like `" yc "` becomes `\\byc\\b`.
    """
    compiled: List[Pattern[str]] = []
    for raw_token in tokens:
        cleaned_token = (raw_token or "").strip()
        if not cleaned_token:
            continue
        compiled.append(
            re.compile(r"\b" + re.escape(cleaned_token) + r"\b", re.IGNORECASE)
        )
    return tuple(compiled)


def _any_pattern_matches(haystack: str, patterns: Sequence[Pattern[str]]) -> bool:
    return any(p.search(haystack) is not None for p in patterns)


def _count_pattern_matches(haystack: str, patterns: Sequence[Pattern[str]]) -> int:
    return sum(1 for p in patterns if p.search(haystack) is not None)


def _iter_token_match_indices(haystack: str, token: str) -> List[int]:
    """All start-offsets of `token` in `haystack` (substring, already lowercased)."""
    if not haystack or not token:
        return []
    matches: List[int] = []
    current_index = 0
    while True:
        match_index = haystack.find(token, current_index)
        if match_index < 0:
            break
        matches.append(match_index)
        current_index = match_index + 1
    return matches


# Pre-compile the word-boundary patterns we use in every rank call. Re-compiling
# regex for every candidate on every query would be wasteful.
FOUNDER_TITLE_PATTERNS = _word_boundary_patterns(FOUNDER_TITLE_TOKENS)
FOUNDER_CONTEXT_PATTERNS = _word_boundary_patterns(FOUNDER_CONTEXT_TOKENS)


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
        "role_paraphrases": _ensure_string_list(raw_parsed.get("role_paraphrases"))[:3],
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


@traceable(name="parse_query", run_type="chain")
def parse_query_node(state: RecruiterGraphState) -> RecruiterGraphState:
    """Parse a free-text role brief into a `ParsedQuery`.

    Uses `llm.with_structured_output(ParsedQueryModel)` so validation happens
    at the OpenAI tool-calling boundary. Falls back to the heuristic parser
    on empty query, missing API key, or Pydantic validation errors.
    """
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

    system_instructions = (
        "You convert recruiter queries into structured filters.\n"
        "Rules:\n"
        "- Keep every list <= 6 items and specific.\n"
        "- Use lowercase strings.\n"
        "- If a field is unknown leave it empty ([] or '').\n"
        "- Do NOT invent constraints the user did not express.\n"
        "- `role_paraphrases` should be up to 3 short, commonly-used variants of the "
        "primary role phrase (e.g. 'ml engineer' -> ['machine learning engineer', "
        "'ml scientist', 'ai engineer']). Omit paraphrases if the role is already a "
        "canonical, unambiguous phrase."
    )
    user_message = f"Recruiter query: {question_text}"

    try:
        structured_llm = llm_model.with_structured_output(ParsedQueryModel)
        parsed_model: ParsedQueryModel = structured_llm.invoke(
            [("system", system_instructions), ("human", user_message)],
            _llm_invoke_config(),
        )
        parsed_query = parsed_model.to_typed_dict()
    except (ValidationError, ValueError, TypeError) as parse_error:
        _append_error(state, f"parse_query fallback: {parse_error}")
        parsed_query = _heuristic_parse(question_text, min_experience_override)
    except Exception as unexpected_error:  # noqa: BLE001 - network/auth errors
        _append_error(state, f"parse_query LLM error: {unexpected_error}")
        parsed_query = _heuristic_parse(question_text, min_experience_override)

    if min_experience_override > parsed_query["min_experience_entries"]:
        parsed_query["min_experience_entries"] = min_experience_override

    state["parsed_query"] = parsed_query
    return state


@traceable(name="retrieve_candidates", run_type="retriever")
def retrieve_candidates_node(state: RecruiterGraphState) -> RecruiterGraphState:
    parsed_query: ParsedQuery = state.get("parsed_query") or dict(DEFAULT_PARSED_QUERY)  # type: ignore[assignment]
    top_k_value = int(state.get("top_k") or 8)

    try:
        retrieved_profiles = search_profiles(
            role_keywords=parsed_query.get("role_keywords"),
            role_paraphrases=parsed_query.get("role_paraphrases"),
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
                role_paraphrases=parsed_query.get("role_paraphrases"),
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

    question_text = state.get("question_text", "").strip()
    if question_text:
        try:
            semantic_profile_ids = semantic_search(
                question_text, top_k=SEMANTIC_RECALL_TOP_K
            )
        except Exception as semantic_error:  # noqa: BLE001 - semantic is best-effort
            _append_error(state, f"semantic_search skipped: {semantic_error}")
            semantic_profile_ids = []

        existing_profile_ids = {p.get("profile_id") for p in retrieved_profiles}
        missing_profile_ids = [
            pid for pid in semantic_profile_ids if pid not in existing_profile_ids
        ]
        if missing_profile_ids:
            try:
                semantic_profiles = fetch_profiles_by_ids(missing_profile_ids)
                retrieved_profiles = retrieved_profiles + semantic_profiles
                _append_error(
                    state,
                    f"semantic_search: unioned {len(semantic_profiles)} additional candidate(s) "
                    f"from embedding recall.",
                )
            except Exception as fetch_error:  # noqa: BLE001
                _append_error(state, f"semantic fetch skipped: {fetch_error}")

    state["candidate_profiles"] = retrieved_profiles
    return state


def should_enrich(state: RecruiterGraphState) -> str:
    """Conditional-edge router: enrich when any retrieved candidate has a short
    about_text, otherwise skip straight to ranking.

    Returned strings are the keys in the mapping passed to
    `StateGraph.add_conditional_edges(...)`.
    """
    candidate_profiles = state.get("candidate_profiles") or []
    for profile in candidate_profiles:
        about_text_value = str(profile.get("about_text") or "")
        if len(about_text_value) < LOW_INFO_ABOUT_THRESHOLD:
            return "enrich"
    return "skip"


def _synthesize_enriched_about(profile: Dict[str, Any]) -> str:
    """Build a richer about_text for a low-info candidate from exp/edu JSON.

    We never overwrite the original `about_text`; we write to
    `about_text_enriched` so downstream nodes (and the UI) can show both.
    """
    about_text_value = str(profile.get("about_text") or "").strip()
    experience_json_text = str(profile.get("experience_json") or "").strip()
    education_json_text = str(profile.get("education_json") or "").strip()
    headline_value = str(profile.get("headline") or "").strip()

    fragments: List[str] = []
    if about_text_value:
        fragments.append(about_text_value)
    if headline_value and headline_value not in about_text_value:
        fragments.append(f"Headline: {headline_value}")
    if experience_json_text:
        fragments.append(f"Experience highlights: {experience_json_text[:600]}")
    if education_json_text:
        fragments.append(f"Education highlights: {education_json_text[:400]}")

    enriched_text = "\n".join(fragments)
    if len(enriched_text) > ENRICHED_ABOUT_MAX_CHARS:
        enriched_text = enriched_text[:ENRICHED_ABOUT_MAX_CHARS].rstrip() + "..."
    return enriched_text


@traceable(name="enrich_low_info", run_type="chain")
def enrich_low_info_node(state: RecruiterGraphState) -> RecruiterGraphState:
    """Deterministic enrichment: populate `about_text_enriched` on candidates
    whose original about_text is below `LOW_INFO_ABOUT_THRESHOLD`.

    No LLM call. Uses experience_json + education_json the retriever already
    pulls from MySQL. The ranker prompt prefers `about_text_enriched` when
    present so the LLM sees more signal on thin profiles.
    """
    candidate_profiles = state.get("candidate_profiles") or []
    if not candidate_profiles:
        return state

    enriched_count = 0
    for profile in candidate_profiles:
        about_text_value = str(profile.get("about_text") or "")
        if len(about_text_value) >= LOW_INFO_ABOUT_THRESHOLD:
            continue
        enriched_text = _synthesize_enriched_about(profile)
        if enriched_text and enriched_text != about_text_value:
            profile["about_text_enriched"] = enriched_text
            enriched_count += 1

    if enriched_count:
        _append_error(
            state,
            f"enrich_low_info: filled about_text_enriched on {enriched_count} low-info candidate(s).",
        )
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


def _phd_is_disqualified(combined_text: str) -> bool:
    """True when a PhD mention is within `PHD_PROXIMITY_WINDOW` chars of a
    disqualifier like "student" / "candidate" / "dropped out".

    Prevents "PhD student" and "dropped out of the PhD program" from reading
    as a completed doctorate.
    """
    for phd_token in PHD_TITLE_TOKENS:
        for phd_offset in _iter_token_match_indices(combined_text, phd_token):
            window_start = max(0, phd_offset - PHD_PROXIMITY_WINDOW)
            window_end = phd_offset + len(phd_token) + PHD_PROXIMITY_WINDOW
            window_text = combined_text[window_start:window_end]
            for disqualifier in PHD_DISQUALIFIER_TOKENS:
                if disqualifier in window_text:
                    return True
    return False


def _score_phd_researcher(profile: Dict[str, Any]) -> tuple:
    combined = _combined_profile_text(profile)
    score = 0.0
    reasons: List[str] = []

    if _any_token_present(combined, PHD_TITLE_TOKENS):
        if _phd_is_disqualified(combined):
            score += PHD_DISQUALIFIED_SCORE_CAP
            reasons.append("PhD mentioned but student/unfinished nearby -> capped")
        else:
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

    # about_text can declare an explicit SF move even when `location` is stale.
    about_says_sf = _any_token_present(about_lower, SF_RELOCATION_TOKENS)
    about_open_to_relocate = _any_token_present(about_lower, SF_OPEN_TO_RELOCATE_TOKENS)
    if about_says_sf:
        return 8.0, f"About text declares SF move (location='{raw_location}')"

    if _any_token_present(location_lower, CA_TECH_TOKENS):
        base_score = 7.0
        if about_open_to_relocate:
            return _clip_score(base_score + 1.0), (
                f"California tech region + open-to-relocate: '{raw_location}'"
            )
        return base_score, f"California tech region: '{raw_location}'"

    if _any_token_present(location_lower, US_TECH_HUB_TOKENS):
        base_score = 5.0
        if about_open_to_relocate:
            return _clip_score(base_score + 2.0), (
                f"US tech hub + open-to-relocate: '{raw_location}'"
            )
        return base_score, f"US tech hub: '{raw_location}'"

    if _any_token_present(location_lower, US_GENERIC_TOKENS) or _any_token_present(
        about_lower, ("remote (us)", "based in the us")
    ):
        base_score = 4.0
        if about_open_to_relocate:
            return _clip_score(base_score + 2.0), (
                f"US/remote + open-to-relocate: '{raw_location}'"
            )
        return base_score, f"US/remote: '{raw_location}'"

    if raw_location:
        base_score = 2.0
        if about_open_to_relocate:
            return _clip_score(base_score + 2.0), (
                f"Out-of-market but open-to-relocate: '{raw_location}'"
            )
        return base_score, f"Out-of-market: '{raw_location}'"

    return 1.0, "no location data"


def _count_skill_hits_with_hedge(combined_text: str) -> Tuple[float, int, int]:
    """Count technical-skill hits, awarding half credit when a hedge phrase
    ("familiar with", "learning", "exploring", ...) appears within
    `SKILL_HEDGE_WINDOW` chars before the skill.

    Returns (weighted_count, full_credit_count, half_credit_count).
    """
    weighted_count = 0.0
    full_credit_count = 0
    half_credit_count = 0
    for skill_token in TECHNICAL_SKILL_TOKENS:
        occurrence_indices = _iter_token_match_indices(combined_text, skill_token)
        if not occurrence_indices:
            continue
        first_offset = occurrence_indices[0]
        window_start = max(0, first_offset - SKILL_HEDGE_WINDOW)
        prefix_window_text = combined_text[window_start:first_offset]
        if any(hedge in prefix_window_text for hedge in SKILL_HEDGE_TOKENS):
            weighted_count += 0.5
            half_credit_count += 1
        else:
            weighted_count += 1.0
            full_credit_count += 1
    return weighted_count, full_credit_count, half_credit_count


def _score_technical_background(profile: Dict[str, Any]) -> tuple:
    combined = _combined_profile_text(profile)
    skills_count_value = int(profile.get("skills_count") or 0)
    score = 0.0
    reasons: List[str] = []

    title_hits = _count_tokens_present(combined, TECHNICAL_TITLE_TOKENS)
    weighted_skill_hits, full_skill_hits, half_skill_hits = _count_skill_hits_with_hedge(combined)

    if title_hits:
        score += min(5.0, title_hits * 1.25)
        reasons.append(f"{title_hits} technical title hit(s)")
    if weighted_skill_hits > 0:
        score += min(4.0, weighted_skill_hits * 0.8)
        hedge_note = (
            f" ({half_skill_hits} hedged)" if half_skill_hits else ""
        )
        reasons.append(
            f"{full_skill_hits + half_skill_hits} technical stack hit(s){hedge_note}"
        )
    if skills_count_value >= 20:
        score += 1.0
        reasons.append(f"{skills_count_value} listed skills")
    elif skills_count_value >= 10:
        score += 0.5

    return _clip_score(score), "; ".join(reasons) or "no technical signal found"


def _matched_school_offsets(haystack: str, school_tokens: Sequence[str]) -> List[int]:
    """All start offsets where any school token appears in haystack."""
    offsets: List[int] = []
    for school_token in school_tokens:
        offsets.extend(_iter_token_match_indices(haystack, school_token))
    return offsets


def _dropout_near_school(combined_text: str, school_tokens: Sequence[str]) -> bool:
    """True when 'dropped out' / 'dropout' is within
    DROPOUT_PROXIMITY_WINDOW chars of any matched school token.

    Prevents "Stanford dropout" from reading as a Stanford graduate.
    """
    for school_offset in _matched_school_offsets(combined_text, school_tokens):
        window_start = max(0, school_offset - DROPOUT_PROXIMITY_WINDOW)
        window_end = school_offset + DROPOUT_PROXIMITY_WINDOW
        window_text = combined_text[window_start:window_end]
        if any(dropout_token in window_text for dropout_token in DROPOUT_TOKENS):
            return True
    return False


def _score_education_prestige(profile: Dict[str, Any]) -> tuple:
    education_blob = (profile.get("education_json") or "").lower()
    about_blob = (profile.get("about_text") or "").lower()
    combined = education_blob + " \n " + about_blob
    education_count_value = int(profile.get("education_count") or 0)

    score = 0.0
    reasons: List[str] = []

    has_graduation_signal = _any_token_present(combined, GRADUATION_SIGNAL_TOKENS)
    # Defer the "phd" graduation credit if it's actually a "phd student".
    phd_graduation_is_real = not _phd_is_disqualified(combined)

    if _any_token_present(combined, TOP_SCHOOL_TOKENS):
        if _dropout_near_school(combined, TOP_SCHOOL_TOKENS):
            score += min(DROPOUT_SCHOOL_SCORE_CAP, 2.0)
            reasons.append("top-tier school but 'dropped out' nearby -> capped")
        elif has_graduation_signal:
            score += 7.0
            reasons.append("top-tier school match")
        else:
            score += 3.5
            reasons.append("top-tier school match (no graduation signal -> halved)")
    elif _any_token_present(combined, STRONG_SCHOOL_TOKENS):
        if _dropout_near_school(combined, STRONG_SCHOOL_TOKENS):
            score += min(DROPOUT_SCHOOL_SCORE_CAP, 1.5)
            reasons.append("strong school but 'dropped out' nearby -> capped")
        elif has_graduation_signal:
            score += 5.0
            reasons.append("strong school match")
        else:
            score += 2.5
            reasons.append("strong school match (no graduation signal -> halved)")

    if _any_token_present(combined, PHD_TITLE_TOKENS) and phd_graduation_is_real:
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

    # Word-boundary regex so "doctor" doesn't match "cto", "discovery" doesn't
    # match "coo", etc.
    title_hits = _count_pattern_matches(combined, FOUNDER_TITLE_PATTERNS)
    context_hits = _count_pattern_matches(combined, FOUNDER_CONTEXT_PATTERNS)

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


def _apply_dimension_gains(
    dimension_scores: Dict[str, float],
    gains: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Apply the per-dimension affine transform and clip each result to [0, 10].

    `gains` defaults to the module-level `DIMENSION_GAINS`. Dimensions with no
    gain entry pass through unchanged (identity). This helper is deterministic
    and the single source of truth for how raw scorer output gets adjusted
    before weighting.
    """
    active_gains = gains if gains is not None else DIMENSION_GAINS
    adjusted_scores: Dict[str, float] = {}
    for dimension_key, raw_dim_score in dimension_scores.items():
        dimension_gain_entry = active_gains.get(dimension_key) if active_gains else None
        if dimension_gain_entry is None:
            adjusted_scores[dimension_key] = float(raw_dim_score)
            continue
        gain_value = float(dimension_gain_entry.get("gain", 1.0))
        bias_value = float(dimension_gain_entry.get("bias", 0.0))
        transformed_score = gain_value * float(raw_dim_score) + bias_value
        adjusted_scores[dimension_key] = _clip_score(transformed_score)
    return adjusted_scores


def _aggregate_rank_score(
    dimension_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    gains: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Weighted aggregate of the 5 sub-scores, clipped to [0, 10].

    Applies `gains` first (per-dimension affine transform) and then sums with
    `weights`. Both default to the module-level config. Accepting overrides
    keeps the calibrator's fit loop (which needs to evaluate MAE under
    candidate gains/weights) pure.
    """
    active_weights = weights if weights is not None else DIMENSION_WEIGHTS
    adjusted_scores = _apply_dimension_gains(dimension_scores, gains=gains)
    total_weighted = 0.0
    for dimension_key, weight_value in active_weights.items():
        total_weighted += float(adjusted_scores.get(dimension_key, 0.0)) * float(weight_value)
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
    ranking_item: DimensionRankingItem,
) -> Tuple[RankedCandidate, List[str]]:
    """Overlay LLM-refined dimension scores/reasons onto the deterministic baseline.

    Evidence-grounded guardrail (R1e): if the LLM moves a dimension by more
    than `EVIDENCE_FREE_DELTA_CAP` from the baseline AND did not quote a
    non-empty evidence string for that dimension, the baseline value is kept.
    This caps hallucination-driven score swings without blocking refinements
    that cite the source.

    Returns (refined_candidate, rejection_messages). rejection_messages is
    appended to the graph's error_messages so the UI can surface how many
    deltas were rejected and why.
    """
    merged_scores: Dict[str, float] = dict(baseline_candidate["dimension_scores"])  # type: ignore[arg-type]
    merged_reasons: Dict[str, str] = dict(baseline_candidate["dimension_reasons"])
    rejection_messages: List[str] = []

    llm_scores_dict = ranking_item.dimension_scores.model_dump()
    llm_reasons_dict = ranking_item.dimension_reasons.model_dump()
    llm_evidence_dict = ranking_item.dimension_evidence.model_dump()

    for dimension_key in DIMENSION_KEYS:
        baseline_value = float(baseline_candidate["dimension_scores"].get(dimension_key, 0.0))  # type: ignore[call-overload]
        proposed_value = _clip_score(float(llm_scores_dict.get(dimension_key, baseline_value)))
        evidence_text = str(llm_evidence_dict.get(dimension_key) or "").strip()
        delta_magnitude = abs(proposed_value - baseline_value)

        if delta_magnitude > EVIDENCE_FREE_DELTA_CAP and not evidence_text:
            rejection_messages.append(
                f"{baseline_candidate['profile_id']}/{dimension_key}: "
                f"rejected LLM delta {delta_magnitude:.1f} (no evidence; kept baseline {baseline_value:.1f})."
            )
            merged_scores[dimension_key] = round(baseline_value, 2)
        else:
            merged_scores[dimension_key] = round(proposed_value, 2)

        reason_text = str(llm_reasons_dict.get(dimension_key) or "").strip()
        if reason_text:
            merged_reasons[dimension_key] = reason_text

    refined_overall = _aggregate_rank_score(merged_scores)
    match_reasons = _ensure_string_list(ranking_item.match_reasons) or baseline_candidate["match_reasons"]
    risks = _ensure_string_list(ranking_item.risks)

    refined_candidate: RankedCandidate = {
        **baseline_candidate,  # type: ignore[misc]
        "rank_score": refined_overall,
        "match_reasons": match_reasons,
        "risks": risks,
        "dimension_scores": merged_scores,  # type: ignore[typeddict-item]
        "dimension_reasons": merged_reasons,
    }
    return refined_candidate, rejection_messages


def _build_ranking_payload(
    capped_candidates: List[Dict[str, Any]],
    baseline_by_profile_id: Dict[str, RankedCandidate],
) -> List[Dict[str, Any]]:
    """Build the list-of-dicts payload fed to the pointwise ranker prompt.

    Prefers `about_text_enriched` (set by the enrich_low_info node) over the
    raw `about_text` so low-info candidates get meaningful context.
    """
    ranking_payload: List[Dict[str, Any]] = []
    for profile in capped_candidates:
        profile_id_value = profile.get("profile_id", "")
        if profile_id_value not in baseline_by_profile_id:
            continue
        about_for_llm = profile.get("about_text_enriched") or profile.get("about_text", "")
        baseline_candidate = baseline_by_profile_id[profile_id_value]
        ranking_payload.append(
            {
                "profile_id": profile_id_value,
                "name": profile.get("name"),
                "headline": profile.get("headline"),
                "location": profile.get("location"),
                "skills_count": profile.get("skills_count"),
                "experience_count": profile.get("experience_count"),
                "education_count": profile.get("education_count"),
                "about_text": _truncate_about(about_for_llm or ""),
                "baseline_dimension_scores": baseline_candidate["dimension_scores"],
                "baseline_dimension_reasons": baseline_candidate["dimension_reasons"],
            }
        )
    return ranking_payload


def _pointwise_rank_with_llm(
    llm_model: ChatOpenAI,
    ranking_payload: List[Dict[str, Any]],
    question_text: str,
    parsed_query: ParsedQuery,
) -> DimensionRankingResponse:
    """Stage-1: per-candidate 5-dimension refinement via structured output.

    Raises on LLM error/validation failure; caller handles fallback.
    """
    system_instructions = (
        "You are a senior recruiter scoring candidates for a fast-growing SF-based startup.\n"
        "Score each of 5 dimensions 0-10 (decimals ok):\n"
        "  - phd_researcher: doctoral training, research roles, publications.\n"
        "  - sf_location_fit: SF/Bay Area ~10, California tech ~7, US tech hub ~5, US/remote ~4, else 1-3.\n"
        "  - technical_background: hands-on strength (engineer/ML/research titles, stack depth).\n"
        "  - education_prestige: prestige and rigor of highest education.\n"
        "  - founder_experience: prior founder / founding-team / executive experience.\n\n"
        "Use `baseline_dimension_scores` as your starting point. Adjust ONLY when about_text/headline "
        "clearly supports a different value.\n\n"
        "For each dimension also fill `dimension_evidence` with a short quoted phrase (<=120 chars) "
        "copied VERBATIM from about_text/headline that justifies the score. If there is no quotable "
        "evidence, return '' and keep the baseline value - do not swing the score without evidence.\n\n"
        "Output exactly one object per candidate; keep profile_ids identical; do NOT emit an overall "
        "rank_score (it is re-aggregated from dimension_scores on the server)."
    )
    user_message = (
        f"Role brief (user query): {question_text}\n\n"
        f"Parsed filters: {json.dumps(parsed_query, ensure_ascii=False)}\n\n"
        f"Candidates JSON:\n{json.dumps(ranking_payload, ensure_ascii=False)}\n"
    )
    structured_llm = llm_model.with_structured_output(DimensionRankingResponse)
    return structured_llm.invoke(
        [("system", system_instructions), ("human", user_message)],
        _llm_invoke_config(),
    )


def _listwise_rerank_top_k(
    llm_model: ChatOpenAI,
    ranked_candidates: List[RankedCandidate],
    question_text: str,
    top_k: int = LISTWISE_RERANK_TOP_K,
) -> Tuple[List[RankedCandidate], Optional[str]]:
    """Stage-2: listwise LLM rerank of the top-K candidates.

    Takes the pointwise-sorted top-K, asks the LLM to re-order them as a
    recruiter would present to a hiring manager, and returns
    (new_ranked_candidates, overall_rationale). Candidates below top-K are
    kept in their pointwise order.
    """
    if len(ranked_candidates) <= 1:
        return ranked_candidates, None

    top_slice = ranked_candidates[:top_k]
    tail = ranked_candidates[top_k:]

    rerank_payload = [
        {
            "profile_id": c["profile_id"],
            "name": c.get("name"),
            "headline": c.get("headline"),
            "rank_score": c.get("rank_score"),
            "dimension_scores": c.get("dimension_scores"),
            "match_reasons": c.get("match_reasons") or [],
            "risks": c.get("risks") or [],
        }
        for c in top_slice
    ]

    system_instructions = (
        "You are a senior recruiter reranking the top candidates for a hiring manager.\n"
        "Re-order the candidates below from strongest to weakest for the role brief. "
        "Keep the same profile_ids; do not invent new ones; do not drop any. For each "
        "position write a one-sentence rationale (<=200 chars) grounded in the provided "
        "dimension_scores / match_reasons / risks. Prefer balanced strength over a single "
        "spiky dimension when the role calls for a founding-team generalist."
    )
    user_message = (
        f"Role brief: {question_text}\n\n"
        f"Top-{len(top_slice)} candidates (pointwise-ordered):\n"
        f"{json.dumps(rerank_payload, ensure_ascii=False)}\n"
    )
    structured_llm = llm_model.with_structured_output(ListwiseRerankResponse)
    rerank_response: ListwiseRerankResponse = structured_llm.invoke(
        [("system", system_instructions), ("human", user_message)],
        _llm_invoke_config(),
    )

    top_by_id: Dict[str, RankedCandidate] = {c["profile_id"]: c for c in top_slice}
    reordered_top: List[RankedCandidate] = []
    seen_ids: set = set()
    for rerank_item in rerank_response.ordered:
        candidate_key = rerank_item.profile_id
        if candidate_key in top_by_id and candidate_key not in seen_ids:
            reordered_top.append(top_by_id[candidate_key])
            seen_ids.add(candidate_key)
    for original in top_slice:
        if original["profile_id"] not in seen_ids:
            reordered_top.append(original)

    return reordered_top + tail, rerank_response.overall_rationale


def _candidate_snippet_for_pairwise(candidate: RankedCandidate) -> Dict[str, Any]:
    """Compact candidate snapshot used in the pairwise prompt. Keeps the
    payload small so we can afford the extra LLM call."""
    return {
        "profile_id": candidate["profile_id"],
        "name": candidate.get("name"),
        "headline": candidate.get("headline"),
        "location": candidate.get("location"),
        "rank_score": candidate.get("rank_score"),
        "dimension_scores": candidate.get("dimension_scores"),
        "match_reasons": (candidate.get("match_reasons") or [])[:3],
        "risks": (candidate.get("risks") or [])[:2],
    }


def _pairwise_tiebreak_adjacent(
    llm_model: ChatOpenAI,
    ranked_candidates: List[RankedCandidate],
    question_text: str,
) -> Tuple[List[RankedCandidate], int, int]:
    """Stage-3: adjacent-pair tie-break on near-ties within the top window.

    For each consecutive pair (i, i+1) whose rank_score gap is below
    `PAIRWISE_TIEBREAK_THRESHOLD`, issue one structured-output call. If the
    LLM picks the lower-ranked candidate with confidence >=
    `PAIRWISE_TIEBREAK_MIN_CONFIDENCE`, swap them. Budget-capped by
    `PAIRWISE_TIEBREAK_MAX_CALLS`.

    Returns (new_ranking, calls_made, swaps_applied).
    """
    if len(ranked_candidates) < 2:
        return ranked_candidates, 0, 0

    working_list: List[RankedCandidate] = list(ranked_candidates)
    window_end = min(PAIRWISE_TIEBREAK_WINDOW, len(working_list))

    structured_llm = llm_model.with_structured_output(PairwiseDecision)
    calls_made = 0
    swaps_applied = 0

    system_instructions = (
        "You are a senior recruiter comparing TWO candidates for the same role brief. "
        "Pick the candidate who better fits the brief. Prefer balanced strength over a "
        "single spiky dimension. Ground your rationale in the dimension_scores or "
        "match_reasons provided; do not invent facts. Respond with the winner's "
        "profile_id (must be one of the two), a one-sentence rationale, and a "
        "confidence in [0, 1]."
    )

    pair_index = 0
    while pair_index < window_end - 1 and calls_made < PAIRWISE_TIEBREAK_MAX_CALLS:
        higher_candidate = working_list[pair_index]
        lower_candidate = working_list[pair_index + 1]
        score_gap = abs(
            float(higher_candidate.get("rank_score") or 0.0)
            - float(lower_candidate.get("rank_score") or 0.0)
        )
        if score_gap >= PAIRWISE_TIEBREAK_THRESHOLD:
            pair_index += 1
            continue

        pair_payload = {
            "role_brief": question_text,
            "candidate_a": _candidate_snippet_for_pairwise(higher_candidate),
            "candidate_b": _candidate_snippet_for_pairwise(lower_candidate),
        }
        user_message = (
            "Pick the stronger of these two near-tie candidates.\n"
            f"{json.dumps(pair_payload, ensure_ascii=False)}\n"
        )
        try:
            pairwise_decision: PairwiseDecision = structured_llm.invoke(
                [("system", system_instructions), ("human", user_message)],
                _llm_invoke_config(),
            )
        except (ValidationError, ValueError, TypeError):
            # Fall back to the pointwise order for this pair; don't fail the
            # whole run because of one structured-output hiccup.
            pair_index += 1
            calls_made += 1
            continue
        except Exception:  # noqa: BLE001 - network/auth; keep the pairwise optional.
            pair_index += 1
            calls_made += 1
            continue

        calls_made += 1
        winner_id = (pairwise_decision.winner_profile_id or "").strip()
        lower_candidate_id = lower_candidate["profile_id"]
        higher_candidate_id = higher_candidate["profile_id"]
        should_swap = (
            winner_id == lower_candidate_id
            and winner_id != higher_candidate_id
            and pairwise_decision.confidence >= PAIRWISE_TIEBREAK_MIN_CONFIDENCE
        )
        if should_swap:
            working_list[pair_index], working_list[pair_index + 1] = (
                lower_candidate,
                higher_candidate,
            )
            swaps_applied += 1
        # Always advance. Re-asking (lower, higher) right after a swap just
        # burns budget to re-confirm the decision we already made.
        pair_index += 1

    return working_list, calls_made, swaps_applied


@traceable(name="rank_candidates", run_type="chain")
def rank_candidates_node(state: RecruiterGraphState) -> RecruiterGraphState:
    """Two-stage ranking:

    1. Pointwise 5-dimension scoring. Deterministic baseline is always computed;
       when the LLM is enabled, `with_structured_output(DimensionRankingResponse)`
       refines each sub-score with an evidence-grounded guardrail.
    2. Listwise rerank over the top-K via `with_structured_output(ListwiseRerankResponse)`.
    """
    candidate_profiles = state.get("candidate_profiles") or []
    if not candidate_profiles:
        state["ranked_candidates"] = []
        return state

    capped_candidates = candidate_profiles[:MAX_RANKING_CANDIDATES]

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
    ranking_payload = _build_ranking_payload(capped_candidates, baseline_by_profile_id)

    try:
        pointwise_response = _pointwise_rank_with_llm(
            llm_model=llm_model,
            ranking_payload=ranking_payload,
            question_text=state.get("question_text", ""),
            parsed_query=parsed_query,
        )
    except (ValidationError, ValueError, TypeError) as pointwise_error:
        _append_error(state, f"rank_candidates pointwise fallback: {pointwise_error}")
        ranked_list = sorted(baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True)
        state["ranked_candidates"] = ranked_list
        return state
    except Exception as unexpected_error:  # noqa: BLE001 - network/auth errors
        _append_error(state, f"rank_candidates LLM error: {unexpected_error}")
        ranked_list = sorted(baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True)
        state["ranked_candidates"] = ranked_list
        return state

    total_rejections = 0
    for ranking_item in pointwise_response.rankings:
        profile_key = ranking_item.profile_id
        if profile_key not in baseline_by_profile_id:
            continue
        merged_candidate, rejection_messages = _merge_llm_dimension_ranking(
            baseline_by_profile_id[profile_key], ranking_item
        )
        baseline_by_profile_id[profile_key] = merged_candidate
        total_rejections += len(rejection_messages)
    if total_rejections:
        _append_error(
            state,
            f"rank_candidates: rejected {total_rejections} evidence-free LLM delta(s) > {EVIDENCE_FREE_DELTA_CAP}.",
        )

    pointwise_ranked = sorted(
        baseline_by_profile_id.values(), key=lambda c: c["rank_score"], reverse=True
    )

    try:
        ranked_candidates, rerank_rationale = _listwise_rerank_top_k(
            llm_model=llm_model,
            ranked_candidates=pointwise_ranked,
            question_text=state.get("question_text", ""),
        )
        if rerank_rationale:
            _append_error(state, f"listwise_rerank: {rerank_rationale}")
    except (ValidationError, ValueError, TypeError) as rerank_error:
        _append_error(state, f"listwise_rerank fallback: {rerank_error}")
        ranked_candidates = pointwise_ranked
    except Exception as rerank_network_error:  # noqa: BLE001
        _append_error(state, f"listwise_rerank LLM error: {rerank_network_error}")
        ranked_candidates = pointwise_ranked

    # Stage-3: adjacent pairwise tie-break on near-ties in the top window.
    try:
        ranked_candidates, pairwise_calls, pairwise_swaps = _pairwise_tiebreak_adjacent(
            llm_model=llm_model,
            ranked_candidates=ranked_candidates,
            question_text=state.get("question_text", ""),
        )
        if pairwise_calls:
            _append_error(
                state,
                f"pairwise_tiebreak: {pairwise_calls} call(s), {pairwise_swaps} swap(s).",
            )
    except Exception as pairwise_error:  # noqa: BLE001 - optional stage, never block
        _append_error(state, f"pairwise_tiebreak skipped: {pairwise_error}")

    state["ranked_candidates"] = ranked_candidates
    return state


@traceable(name="synthesize_shortlist", run_type="chain")
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
        model_response = llm_model.invoke(prompt_text, _llm_invoke_config())
        summary_text = model_response.content if isinstance(model_response.content, str) else str(model_response.content)
    except Exception as summary_error:  # noqa: BLE001 - surface in UI
        _append_error(state, f"synthesize_shortlist fallback: {summary_error}")
        summary_text = f"Shortlist (deterministic fallback). Sources: {top_profile_ids}"

    state["shortlist_summary"] = summary_text.strip()
    return state


def build_graph() -> Any:
    """Compile the recruiter LangGraph.

    Conditional routing: `retrieve_candidates` fans out to either
    `enrich_low_info` (if any candidate has a short about_text) or directly
    to `rank_candidates`. This is the canonical `add_conditional_edges` idiom.
    """
    graph_builder = StateGraph(RecruiterGraphState)
    graph_builder.add_node("parse_query", parse_query_node)
    graph_builder.add_node("retrieve_candidates", retrieve_candidates_node)
    graph_builder.add_node("enrich_low_info", enrich_low_info_node)
    graph_builder.add_node("rank_candidates", rank_candidates_node)
    graph_builder.add_node("synthesize_shortlist", synthesize_shortlist_node)

    graph_builder.set_entry_point("parse_query")
    graph_builder.add_edge("parse_query", "retrieve_candidates")
    graph_builder.add_conditional_edges(
        "retrieve_candidates",
        should_enrich,
        {"enrich": "enrich_low_info", "skip": "rank_candidates"},
    )
    graph_builder.add_edge("enrich_low_info", "rank_candidates")
    graph_builder.add_edge("rank_candidates", "synthesize_shortlist")
    graph_builder.add_edge("synthesize_shortlist", END)

    return graph_builder.compile()


def _current_trace_url() -> Optional[str]:
    """Best-effort extraction of the active LangSmith trace URL.

    Silent no-op when langsmith isn't installed, tracing is disabled, or the
    SDK returns no active run tree. We try a few attribute names because
    langsmith's public API for this has moved around.
    """
    if not _LANGSMITH_AVAILABLE:
        return None
    try:
        run_tree = get_current_run_tree()
    except Exception:  # noqa: BLE001 - never fail the graph over tracing
        return None
    if run_tree is None:
        return None
    for attribute_name in ("url", "get_url"):
        attribute_value = getattr(run_tree, attribute_name, None)
        if callable(attribute_value):
            try:
                resolved_url = attribute_value()
            except Exception:  # noqa: BLE001
                resolved_url = None
            if isinstance(resolved_url, str) and resolved_url:
                return resolved_url
        elif isinstance(attribute_value, str) and attribute_value:
            return attribute_value
    return None


@traceable(name="run_recruiter_search", run_type="chain")
def run_recruiter_search(
    question_text: str,
    top_k: int = 8,
    min_experience_entries: int = 0,
) -> Dict[str, Any]:
    """Top-level entry point. Wrapped in `@traceable` so LangSmith captures
    the whole run as a single trace tree, with each node as a child run.
    """
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

    # Install a per-run token collector so the UI can surface a ballpark
    # cost without a LangSmith API round-trip. We append/pop instead of
    # replacing so nested calls (e.g. smoke test runs in parallel) don't
    # clobber each other's counters.
    token_collector = TokenUsageCollector()
    _ACTIVE_LLM_CALLBACKS.append(token_collector)
    try:
        final_state: RecruiterGraphState = compiled_graph.invoke(initial_state)
    finally:
        try:
            _ACTIVE_LLM_CALLBACKS.remove(token_collector)
        except ValueError:
            pass

    trace_url_value = _current_trace_url()
    model_name_in_use = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    token_usage_summary: Dict[str, Any] = {
        "model": model_name_in_use,
        "llm_calls": int(token_collector.llm_calls),
        "prompt_tokens": int(token_collector.prompt_tokens),
        "completion_tokens": int(token_collector.completion_tokens),
        "total_tokens": int(token_collector.total_tokens),
        "estimated_cost_usd": round(
            float(token_collector.estimated_cost_usd(model_name_in_use)), 6
        ),
    }
    return {
        "parsed_query": final_state.get("parsed_query", dict(DEFAULT_PARSED_QUERY)),
        "candidate_profiles": final_state.get("candidate_profiles", []),
        "ranked_candidates": final_state.get("ranked_candidates", []),
        "shortlist_summary": final_state.get("shortlist_summary", ""),
        "error_messages": final_state.get("error_messages", []),
        "trace_url": trace_url_value,
        "token_usage": token_usage_summary,
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
