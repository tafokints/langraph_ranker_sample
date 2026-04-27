"""Unit tests for `src.schools_loader`.

These pin down the structured schools-corpus behavior that replaced the
keyword TOP_SCHOOL_TOKENS / STRONG_SCHOOL_TOKENS lists. The plan is
specific about what should differentiate from what:

- Stanford GSB part-time exec MUST score lower than Stanford BS CS.
- Non-US top schools (Oxford, Cambridge, ETH, Tsinghua) must register as tier 1.
- Tier 2 schools must register lower than tier 1 with the same program signal.
- Doctoral program > graduate (masters/MBA) > undergraduate at the same school.
- Misspelled or unlisted school => no match (graceful 0 score, not a crash).

The full integration into `_score_education_prestige` is also exercised
end-to-end so we lock the differentiation requirement at the public API.

Run with:  python -m pytest tests/test_schools_loader.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src import schools_loader
from src.langgraph_app import _score_education_prestige
from src.schools_loader import (
    best_school_score,
    find_school_matches,
    load_schools_config,
    program_modifier_at_offset,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    """The loader memoizes the production config; tests that mutate file
    state must start from a clean cache and leave a clean cache."""
    reset_cache()
    yield
    reset_cache()


def _profile(**overrides) -> Dict[str, Any]:
    base = {
        "headline": "",
        "about_text": "",
        "location": "",
        "experience_json": "",
        "education_json": "",
        "skills_count": 0,
        "education_count": 0,
    }
    base.update(overrides)
    return base


def test_production_config_loads_with_known_top_schools() -> None:
    """The committed config/schools.json must include Stanford, MIT,
    Harvard, Oxford, ETH at tier 1 — the README and rubric assume them."""
    production_config = load_schools_config(force_reload=True)
    indexed_aliases_lower = {alias for alias, _record in production_config["alias_index"]}
    expected_aliases_present = {
        "stanford",
        "mit",
        "harvard",
        "oxford",
        "cambridge",
        "eth zurich",
        "tsinghua",
    }
    missing_aliases = expected_aliases_present - indexed_aliases_lower
    assert not missing_aliases, f"corpus missing aliases: {missing_aliases}"

    name_to_tier = {
        record["name"]: record["tier"] for record in production_config["schools"]
    }
    assert name_to_tier["Stanford University"] == 1
    assert name_to_tier["University of Oxford"] == 1
    assert name_to_tier["ETH Zurich"] == 1


def test_find_school_matches_picks_longest_alias_first() -> None:
    """Long aliases like 'massachusetts institute of technology' should
    be tried before bare 'mit' so that the more specific alias wins when
    both fire on the same text."""
    text_lower = "massachusetts institute of technology, class of 2020".lower()
    matches = find_school_matches(text_lower)
    assert any(
        match["record"]["name"] == "Massachusetts Institute of Technology"
        for match in matches
    )


def test_short_alias_only_matches_at_word_boundary() -> None:
    """Bare 'mit' must not match inside 'committee', 'admittedly', etc."""
    text_lower = "served on the admittance committee at hometown high".lower()
    matches = find_school_matches(text_lower)
    matched_names = {match["record"]["name"] for match in matches}
    assert "Massachusetts Institute of Technology" not in matched_names


def test_program_modifier_picks_executive_over_default() -> None:
    """When the surrounding text says 'executive program', the modifier must
    fall to the executive bucket even though the school by itself is tier 1."""
    text_lower = "stanford graduate school of business executive program 2017".lower()
    school_offset = text_lower.find("stanford")
    modifier_value, modifier_label = program_modifier_at_offset(text_lower, school_offset)
    assert modifier_value < 0.6
    assert "executive" in modifier_label.lower()


def test_program_modifier_picks_doctoral_when_phd_present() -> None:
    text_lower = "phd, computer science, stanford university".lower()
    school_offset = text_lower.find("stanford")
    modifier_value, modifier_label = program_modifier_at_offset(text_lower, school_offset)
    assert modifier_value > 1.0
    assert "doctoral" in modifier_label.lower()


def test_best_school_score_prefers_tier_one_when_both_match() -> None:
    """Stanford (tier 1) beats UCLA (tier 2) at the same program modifier."""
    text_lower = "stanford university, b.s. computer science. ucla, m.s. computer science.".lower()
    best_match = best_school_score(text_lower)
    assert best_match is not None
    assert best_match["name"] == "Stanford University"
    assert best_match["tier"] == 1


def test_unlisted_school_returns_no_match() -> None:
    """The corpus is finite; misspellings or fringe schools must NOT match
    silently — the scorer should fall through to the 'no prestige match'
    branch rather than picking up a wrong school."""
    text_lower = "podunk junior college, associate of arts, 2010".lower()
    best_match = best_school_score(text_lower)
    assert best_match is None


def test_score_education_prestige_executive_program_lower_than_full_bs(
) -> None:
    """End-to-end: this is the differentiation the plan called out
    explicitly — 'Stanford GSB part-time exec' must score below 'Stanford
    BS CS'."""
    full_bs_profile = _profile(
        education_json='[{"school": "Stanford University", "degree": "B.S. Computer Science"}]',
        about_text="Class of 2018.",
        education_count=1,
    )
    executive_profile = _profile(
        education_json='[{"school": "Stanford Graduate School of Business", "degree": "Executive Program in General Management"}]',
        about_text="Stanford GSB Executive Program 2019.",
        education_count=1,
    )
    full_bs_score, _full_reason = _score_education_prestige(full_bs_profile)
    executive_score, _exec_reason = _score_education_prestige(executive_profile)
    assert full_bs_score > executive_score, (
        f"BS ({full_bs_score}) should outscore exec program ({executive_score})"
    )
    assert executive_score < 5.0, (
        f"executive-program score should clearly de-rank, got {executive_score}"
    )


def test_score_education_prestige_oxford_non_us_tier_one() -> None:
    """Hole flagged in the plan: 'misses non-US top schools'."""
    profile = _profile(
        education_json='[{"school": "University of Oxford", "degree": "M.Sc. Computer Science"}]',
        about_text="Master of Science, Oxford.",
        education_count=1,
    )
    score, _reason = _score_education_prestige(profile)
    assert score >= 7.0, f"Oxford master's should score >= 7.0, got {score}"


def test_score_education_prestige_tier_two_lower_than_tier_one() -> None:
    """Same program signal at UCLA (tier 2) must score below Stanford (tier 1)."""
    stanford_profile = _profile(
        education_json='[{"school": "Stanford University", "degree": "B.S. Computer Science"}]',
        about_text="Bachelor's, Class of 2018.",
        education_count=1,
    )
    ucla_profile = _profile(
        education_json='[{"school": "UCLA", "degree": "B.S. Computer Science"}]',
        about_text="Bachelor's, Class of 2018.",
        education_count=1,
    )
    stanford_score, _stanford_reason = _score_education_prestige(stanford_profile)
    ucla_score, _ucla_reason = _score_education_prestige(ucla_profile)
    assert stanford_score > ucla_score


def test_load_schools_config_handles_missing_file(tmp_path: Path) -> None:
    """First-run safety: a missing corpus must not crash the loader."""
    nonexistent_path = tmp_path / "no_such_schools.json"
    config_object = load_schools_config(path=nonexistent_path, force_reload=True)
    assert config_object["schools"] == []
    assert config_object["alias_index"] == []


def test_load_schools_config_handles_malformed_json(tmp_path: Path) -> None:
    """A broken corpus must not crash the loader; same fail-open contract
    as `weights_loader.load_weights`."""
    malformed_path = tmp_path / "bad_schools.json"
    malformed_path.write_text("{not valid json", encoding="utf-8")
    config_object = load_schools_config(path=malformed_path, force_reload=True)
    assert config_object["schools"] == []


def test_load_schools_config_drops_records_without_aliases(
    tmp_path: Path,
) -> None:
    """A school record missing aliases would be unmatchable; the loader
    must drop it rather than carry an unmatchable ghost in the corpus."""
    schools_payload = {
        "version": 1,
        "tier_base_scores": {"1": 7.0},
        "schools": [
            {"name": "Bad School", "aliases": [], "tier": 1, "country": "US"},
            {
                "name": "Good School",
                "aliases": ["good school"],
                "tier": 1,
                "country": "US",
            },
        ],
    }
    schools_path = tmp_path / "schools.json"
    schools_path.write_text(json.dumps(schools_payload), encoding="utf-8")

    config_object = load_schools_config(path=schools_path, force_reload=True)
    school_names = {record["name"] for record in config_object["schools"]}
    assert school_names == {"Good School"}
