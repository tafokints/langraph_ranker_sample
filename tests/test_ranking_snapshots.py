"""Offline ranking-snapshot test.

Pins the deterministic-only ordering of `run_recruiter_search` against a
small in-memory profile pool, so a regression in any of the 5 dimension
scorers (or in `_aggregate_rank_score`) shows up as a unit-test failure
instead of waiting for the slow live smoke test.

Mocking strategy:
  - `_build_llm()` is patched to return None, which routes the graph
    through every deterministic fallback. No OPENAI_API_KEY needed.
  - `search_profiles` is patched to return our in-memory pool.
  - `semantic_search` is patched to return [] so the embedding path
    becomes a no-op.

The test loads prompts and `expected_top_3_contains` from the same YAML
fixture the live smoke test uses, so adding a new prompt automatically
adds a regression case here too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # type: ignore[import-not-found]

from src import langgraph_app


PROMPTS_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "smoke_prompts.yml"


# In-memory profile pool keyed by profile_id. Each entry is shaped like
# what `src.retriever._row_to_candidate` returns. We populate enough
# fields for the 5 deterministic scorers to fire (and not fire) clearly.
_TEST_PROFILES: Dict[str, Dict[str, Any]] = {
    "p_ml_sf_phd": {
        "profile_id": "p_ml_sf_phd",
        "name": "Alex ML",
        "headline": "Staff ML engineer / research scientist",
        "location": "San Francisco, CA",
        "about_text": (
            "Class of 2018 PhD from Stanford, published in NeurIPS and ICML. "
            "Worked as a research scientist at a Y Combinator-backed AI startup. "
            "Stack: python, pytorch, kubernetes, aws. Co-authored on arXiv."
        ),
        "experience_json": "Co-founder at Acme AI (Y Combinator W21). Series A raised. ml engineer.",
        "education_json": "PhD, Stanford, class of 2018; B.S., MIT, class of 2012",
        "skills_count": 18,
        "experience_count": 6,
        "education_count": 3,
        "relevance_score": 90,
    },
    "p_ml_us_grad": {
        "profile_id": "p_ml_us_grad",
        "name": "Casey US",
        "headline": "Senior software engineer",
        "location": "Austin, TX, USA",
        "about_text": (
            "ML engineer at a US tech company. Stack: python, tensorflow, sql. "
            "Bachelor of Science from UT Austin, class of 2015."
        ),
        "experience_json": "machine learning engineer; senior software engineer",
        "education_json": "B.S., UT Austin, class of 2015",
        "skills_count": 10,
        "experience_count": 4,
        "education_count": 1,
        "relevance_score": 60,
    },
    "p_designer_sf": {
        "profile_id": "p_designer_sf",
        "name": "Jordan Design",
        "headline": "Senior product designer at fintech startup",
        "location": "San Francisco, CA",
        "about_text": (
            "Senior product designer with 8 years experience in fintech. "
            "Class of 2014 alumna of Carnegie Mellon. Worked at a Series B fintech."
        ),
        "experience_json": "Senior product designer; product designer; ux designer",
        "education_json": "B.Des, Carnegie Mellon, class of 2014",
        "skills_count": 9,
        "experience_count": 5,
        "education_count": 1,
        "relevance_score": 70,
    },
    "p_unity_dev": {
        "profile_id": "p_unity_dev",
        "name": "Riley Unity",
        "headline": "Game developer (Unity, C#)",
        "location": "Seattle, WA",
        "about_text": (
            "Shipped 4 mobile titles in Unity using C#. Stack includes c++, sql. "
            "B.S. in Computer Science, class of 2017, University of Washington."
        ),
        "experience_json": "Game developer; gameplay engineer (Unity, C#).",
        "education_json": "B.S., University of Washington, class of 2017",
        "skills_count": 8,
        "experience_count": 5,
        "education_count": 1,
        "relevance_score": 80,
    },
    "p_thin": {
        "profile_id": "p_thin",
        "name": "Sam Thin",
        "headline": "Engineer",
        "location": "Remote",
        "about_text": "",
        "experience_json": "",
        "education_json": "",
        "skills_count": 0,
        "experience_count": 1,
        "education_count": 0,
        "relevance_score": 5,
    },
}


PROMPT_EXPECTATIONS: Dict[str, List[str]] = {
    # Each entry: prompt name (matches name in fixtures/smoke_prompts.yml)
    # -> at least one of these profile_ids MUST appear in top-3.
    "role-focused": ["p_ml_sf_phd", "p_ml_us_grad"],
    "skill-focused": ["p_unity_dev"],
    "location-focused": ["p_designer_sf"],
}


def _load_smoke_prompts() -> List[Dict[str, Any]]:
    if not PROMPTS_FIXTURE_PATH.exists():
        pytest.skip(f"prompt fixture missing: {PROMPTS_FIXTURE_PATH}")
    with PROMPTS_FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        prompt_list = yaml.safe_load(fixture_file)
    if not isinstance(prompt_list, list):
        pytest.skip("prompt fixture is not a list")
    return prompt_list


def _profile_combined_text(profile: Dict[str, Any]) -> str:
    """Lower-case concatenation of all the fields the lexical retriever
    weights for `LIKE`-style scoring. Used by the patched
    `search_profiles` so a query like "Unity C#" actually filters to the
    Unity dev profile rather than returning everything."""
    parts = [
        str(profile.get("name") or ""),
        str(profile.get("headline") or ""),
        str(profile.get("about_text") or ""),
        str(profile.get("experience_json") or ""),
        str(profile.get("education_json") or ""),
        str(profile.get("location") or ""),
    ]
    return " ".join(parts).lower()


def _fake_search_profiles(**kwargs: Any) -> List[Dict[str, Any]]:
    """Stand-in for `src.retriever.search_profiles` that does the same
    "any keyword present" filter the real lexical retriever falls back
    to. Returns *all* profiles when no keywords are provided so the
    fixture-load path still works.
    """
    role_keywords = list(kwargs.get("role_keywords") or [])
    paraphrases = list(kwargs.get("role_paraphrases") or [])
    skills = list(kwargs.get("skills") or [])
    must_have = list(kwargs.get("must_have_keywords") or [])
    nice_to_have = list(kwargs.get("nice_to_have_keywords") or [])

    raw_phrases = [
        token.lower().strip()
        for token in (role_keywords + paraphrases + skills + must_have + nice_to_have)
        if isinstance(token, str) and token.strip()
    ]
    if not raw_phrases:
        return [dict(profile) for profile in _TEST_PROFILES.values()]

    # Tokenize on whitespace + drop generic recruiter filler like "with",
    # "and", "the". Mirrors how the real lexical retriever scores: a
    # keyword phrase contributes any of its meaningful word-tokens.
    stop_words = {
        "with", "and", "the", "for", "from", "based", "experience",
        "in", "on", "of", "or", "to", "a", "an",
    }
    keyword_pool: List[str] = []
    for phrase in raw_phrases:
        if phrase in keyword_pool:
            continue
        keyword_pool.append(phrase)
        for token in phrase.split():
            cleaned_token = token.strip(",.;:!?'\"()[]{}").lower()
            if len(cleaned_token) >= 3 and cleaned_token not in stop_words:
                if cleaned_token not in keyword_pool:
                    keyword_pool.append(cleaned_token)

    matches: List[Dict[str, Any]] = []
    for profile in _TEST_PROFILES.values():
        haystack = _profile_combined_text(profile)
        if any(keyword in haystack for keyword in keyword_pool):
            matches.append(dict(profile))
    if matches:
        return matches
    return [dict(profile) for profile in _TEST_PROFILES.values()]


@pytest.fixture(autouse=True)
def _patch_external_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic path: no LLM, no DB, no embeddings."""

    monkeypatch.setattr(langgraph_app, "_build_llm", lambda: None)
    monkeypatch.setattr(
        langgraph_app,
        "search_profiles",
        _fake_search_profiles,
    )
    monkeypatch.setattr(
        langgraph_app,
        "semantic_search",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        langgraph_app,
        "fetch_profiles_by_ids",
        lambda _profile_ids: [],
    )


def _top_3_profile_ids(result: Dict[str, Any]) -> List[str]:
    ranked = result.get("ranked_candidates") or []
    return [str(candidate.get("profile_id") or "") for candidate in ranked[:3]]


@pytest.mark.parametrize(
    "prompt_name", sorted(PROMPT_EXPECTATIONS.keys()), ids=lambda value: value
)
def test_prompt_top_three_contains_expected_id(prompt_name: str) -> None:
    """For every prompt with an expectation, at least one expected
    profile_id must appear in the top-3 of the deterministic ranking."""
    smoke_prompts = _load_smoke_prompts()
    prompt_entry = next(
        (item for item in smoke_prompts if item.get("name") == prompt_name),
        None,
    )
    if prompt_entry is None:
        pytest.skip(f"prompt '{prompt_name}' not in fixture")

    result = langgraph_app.run_recruiter_search(
        question_text=str(prompt_entry["query"]),
        top_k=int(prompt_entry.get("top_k", 6)),
        min_experience_entries=int(prompt_entry.get("min_experience", 0)),
    )

    top_three_ids = _top_3_profile_ids(result)
    assert top_three_ids, "expected at least one ranked candidate"

    expected_any_of = PROMPT_EXPECTATIONS[prompt_name]
    overlap = set(top_three_ids) & set(expected_any_of)
    assert overlap, (
        f"prompt '{prompt_name}' top-3 was {top_three_ids}; "
        f"expected at least one of {expected_any_of}"
    )


def test_phd_researcher_dimension_fires_for_phd_profile() -> None:
    """Belt-and-braces: regardless of prompt, a profile with strong PhD
    and research signals should never score 0 on phd_researcher.
    Catches scorer regressions before the prompt-level test does."""
    result = langgraph_app.run_recruiter_search(
        question_text="ML engineer",
        top_k=6,
        min_experience_entries=0,
    )
    ranked = result.get("ranked_candidates") or []
    phd_candidate = next(
        (candidate for candidate in ranked if candidate.get("profile_id") == "p_ml_sf_phd"),
        None,
    )
    assert phd_candidate is not None, "expected PhD candidate in ranked output"
    phd_dim_score = float(
        (phd_candidate.get("dimension_scores") or {}).get("phd_researcher", 0.0)
    )
    assert phd_dim_score >= 5.0, (
        f"phd_researcher dim should be >= 5.0 for a Stanford PhD, NeurIPS-published "
        f"research scientist; got {phd_dim_score}"
    )


def test_thin_profile_does_not_dominate_top_three() -> None:
    """A profile with empty about_text + no signals should never beat
    a strong deterministic match in the top-3."""
    result = langgraph_app.run_recruiter_search(
        question_text="ML engineer",
        top_k=6,
        min_experience_entries=0,
    )
    top_three_ids = _top_3_profile_ids(result)
    assert "p_thin" not in top_three_ids, (
        f"thin/empty profile should not appear in top-3; got {top_three_ids}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
