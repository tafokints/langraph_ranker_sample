"""Typed schemas for the recruiter LangGraph pipeline.

Two parallel type systems live here on purpose:

- `TypedDict`s describe the mutable dict-shaped state that flows between
  LangGraph nodes. The graph itself is dict-native, so using `TypedDict`
  keeps node signatures readable and `state[...]` assignments valid.
- Pydantic `BaseModel`s mirror the LLM-boundary payloads. They are the
  types passed to `llm.with_structured_output(...)` so the LLM emits
  validated JSON via OpenAI tool-calling instead of hand-rolled
  `json.loads` with a try/except. Each model has a `to_typed_dict()`
  helper that converts back into the corresponding TypedDict shape the
  rest of the graph expects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ParsedQuery(TypedDict):
    """Structured filters extracted from a free-text recruiter query."""

    role_keywords: List[str]
    role_paraphrases: List[str]
    skills: List[str]
    location: str
    min_experience_entries: int
    must_have_keywords: List[str]
    nice_to_have_keywords: List[str]


class DimensionBreakdown(TypedDict):
    """Per-dimension sub-scores (0-10) that aggregate into rank_score.

    The five dimensions target an SF-based startup founding/early-team lens:
    research depth, location fit, technical strength, education rigor, and
    prior founder/executive experience.
    """

    phd_researcher: float
    sf_location_fit: float
    technical_background: float
    education_prestige: float
    founder_experience: float


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
    dimension_scores: DimensionBreakdown
    dimension_reasons: Dict[str, str]


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
    trace_url: str
    # Optional per-run override of DIMENSION_WEIGHTS / DIMENSION_GAINS; the
    # A/B harness (`scripts/ab_compare_weights.py`) sets these so two runs
    # can swap configs without restarting the process or rewriting the
    # module-level constants. None / missing -> use module defaults.
    weights_override: Dict[str, float]
    gains_override: Dict[str, Dict[str, float]]
    # Optional per-run feature flags for the ablation harness
    # (`scripts/ablation_table.py`). Each key maps a graph-stage name to a
    # boolean; True means "skip that stage and fall back to deterministic
    # behavior". Missing keys preserve the default behavior so production
    # paths are unaffected.
    feature_flags: Dict[str, bool]


DEFAULT_PARSED_QUERY: ParsedQuery = {
    "role_keywords": [],
    "role_paraphrases": [],
    "skills": [],
    "location": "",
    "min_experience_entries": 0,
    "must_have_keywords": [],
    "nice_to_have_keywords": [],
}


# --- Pydantic models used at the LLM boundary via with_structured_output ---


class ParsedQueryModel(BaseModel):
    """Structured-output target for the `parse_query` node.

    Mirrors `ParsedQuery` but with Pydantic validation so the LLM is forced
    (via OpenAI tool-calling under the hood) to emit a well-typed object.
    """

    role_keywords: List[str] = Field(
        default_factory=list,
        description="Short role/title keywords, lowercase, <= 6 items.",
    )
    role_paraphrases: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 short paraphrases of the role phrase, lowercase. "
            "Used by the retriever to union candidates across variants "
            "(e.g. 'ml engineer' -> 'machine learning engineer')."
        ),
        max_length=3,
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Specific skills/tools mentioned in the brief, lowercase.",
    )
    location: str = Field(
        default="",
        description="City/region/country or empty string if unspecified.",
    )
    min_experience_entries: int = Field(
        default=0,
        ge=0,
        description="Minimum number of experience records required.",
    )
    must_have_keywords: List[str] = Field(
        default_factory=list,
        description="Terms that MUST appear in a candidate's text, lowercase.",
    )
    nice_to_have_keywords: List[str] = Field(
        default_factory=list,
        description="Terms that are bonuses but not required, lowercase.",
    )

    def to_typed_dict(self) -> ParsedQuery:
        return {
            "role_keywords": list(self.role_keywords),
            "role_paraphrases": list(self.role_paraphrases)[:3],
            "skills": list(self.skills),
            "location": self.location,
            "min_experience_entries": int(self.min_experience_entries),
            "must_have_keywords": list(self.must_have_keywords),
            "nice_to_have_keywords": list(self.nice_to_have_keywords),
        }


class DimensionScoresModel(BaseModel):
    """Per-dimension sub-scores (0-10) returned by the LLM ranker."""

    phd_researcher: float = Field(ge=0.0, le=10.0)
    sf_location_fit: float = Field(ge=0.0, le=10.0)
    technical_background: float = Field(ge=0.0, le=10.0)
    education_prestige: float = Field(ge=0.0, le=10.0)
    founder_experience: float = Field(ge=0.0, le=10.0)


class DimensionReasonsModel(BaseModel):
    """Short, evidence-grounded reason per dimension (<= 140 chars each)."""

    phd_researcher: str = ""
    sf_location_fit: str = ""
    technical_background: str = ""
    education_prestige: str = ""
    founder_experience: str = ""


class DimensionEvidenceModel(BaseModel):
    """Quoted phrase per dimension copied verbatim from source text.

    Empty string is allowed. Used by the R1 guardrail: if the LLM swings a
    dimension score by > 3.0 from the deterministic baseline with no quoted
    evidence, the baseline is kept instead.
    """

    phd_researcher: str = ""
    sf_location_fit: str = ""
    technical_background: str = ""
    education_prestige: str = ""
    founder_experience: str = ""


class DimensionRankingItem(BaseModel):
    """One candidate's refined ranking as returned by the LLM."""

    profile_id: str
    dimension_scores: DimensionScoresModel
    dimension_reasons: DimensionReasonsModel = Field(default_factory=DimensionReasonsModel)
    dimension_evidence: DimensionEvidenceModel = Field(default_factory=DimensionEvidenceModel)
    match_reasons: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class DimensionRankingResponse(BaseModel):
    """Top-level wrapper required by `with_structured_output` (needs a BaseModel)."""

    rankings: List[DimensionRankingItem] = Field(default_factory=list)


class ListwiseRerankItem(BaseModel):
    """One position in the recruiter's preferred ordering."""

    profile_id: str
    rationale: str = Field(default="", max_length=200)


class ListwiseRerankResponse(BaseModel):
    """Stage-2 listwise reranker output: recruiter-ordered top-K with rationale."""

    ordered: List[ListwiseRerankItem] = Field(default_factory=list)
    overall_rationale: Optional[str] = Field(
        default=None,
        description="One-sentence why-this-order summary (optional).",
    )


class PairwiseDecision(BaseModel):
    """Output of the Round-2 pairwise tie-break call.

    Only triggered when two adjacent candidates' rank_scores differ by less
    than `PAIRWISE_TIEBREAK_THRESHOLD`. The LLM sees both profiles side-by-side
    and picks the stronger fit for the brief.
    """

    winner_profile_id: str = Field(
        description=(
            "profile_id of the candidate that better fits the recruiter brief. "
            "Must be one of the two profile_ids supplied."
        ),
    )
    rationale: str = Field(
        default="",
        max_length=200,
        description="One-sentence justification, ideally quoting the source.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 (coin flip) to 1.0 (clear winner). Swaps only apply when "
            "confidence >= PAIRWISE_TIEBREAK_MIN_CONFIDENCE."
        ),
    )
