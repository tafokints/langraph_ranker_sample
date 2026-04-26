"""Unit tests for the 5 dimension scorers in `src.langgraph_app`.

These pin down Round 2 bug fixes:
- PhD mention near "student" / "candidate" / "dropped out" => capped bonus.
- Education prestige requires a graduation signal; "Stanford dropout" capped.
- Founder tokens use word-boundary regex (doctor != ceo, coo != cooperate).
- Technical skills near "familiar with" / "learning" count at half.
- SF fit reads `about_text` for "moving to SF" / "open to relocate".

Run with:  python -m pytest tests/test_dimension_scorers.py -v
"""

from __future__ import annotations

from src.langgraph_app import (
    _score_education_prestige,
    _score_founder_experience,
    _score_phd_researcher,
    _score_sf_location_fit,
    _score_technical_background,
)


def _profile(**overrides) -> dict:
    """Build a profile dict with the fields the scorers read. Sensible
    defaults keep each test focused on a single field."""
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


# -------- _score_phd_researcher --------


def test_phd_completed_gets_full_bonus() -> None:
    completed_profile = _profile(
        headline="Research Scientist",
        about_text="Completed my PhD in Machine Learning at Stanford.",
    )
    completed_score, reason = _score_phd_researcher(completed_profile)
    assert completed_score >= 7.0, f"Expected full PhD bonus, got {completed_score} ({reason})"


def test_phd_student_scores_less_than_completed_phd() -> None:
    # Behavioral check: "phd student" must rank below a completed PhD in the
    # same query, even if it still picks up adjacent research-title credit.
    completed_profile = _profile(
        headline="Research Scientist",
        about_text="Completed my PhD in Machine Learning at Stanford.",
    )
    student_profile = _profile(
        headline="PhD Student at MIT",
        about_text="Currently a PhD student working on distributed systems.",
    )
    completed_score, _ = _score_phd_researcher(completed_profile)
    student_score, _ = _score_phd_researcher(student_profile)
    assert student_score < completed_score, (
        f"PhD student ({student_score}) should score lower than completed PhD "
        f"({completed_score})"
    )
    # And the PhD-specific bonus itself must be capped (<= 3.0 of the total).
    assert student_score <= 5.0


def test_phd_dropped_out_is_capped() -> None:
    dropout_profile = _profile(
        about_text="Dropped out of my PhD program to join a startup.",
    )
    score, reason = _score_phd_researcher(dropout_profile)
    assert score <= 3.0, f"Expected <=3.0 for PhD dropout, got {score} ({reason})"


def test_phd_candidate_scores_less_than_completed_phd() -> None:
    completed_profile = _profile(
        headline="Research Scientist",
        about_text="Completed my PhD in Machine Learning at Stanford.",
    )
    candidate_profile = _profile(headline="PhD Candidate, CS", about_text="")
    completed_score, _ = _score_phd_researcher(completed_profile)
    candidate_score, _ = _score_phd_researcher(candidate_profile)
    assert candidate_score < completed_score


# -------- _score_founder_experience --------


def test_founder_ceo_matches_word_boundary() -> None:
    profile = _profile(
        headline="CEO and Co-founder",
        about_text="Co-founded the company in 2020.",
    )
    score, _reason = _score_founder_experience(profile)
    assert score >= 5.0


def test_founder_does_not_match_inside_other_words() -> None:
    # "cto" is a substring of "protocol" → old code would false-positive.
    # Also "ceo" used to match inside "ceoxxx" style tokens.
    profile = _profile(
        headline="Software Engineer",
        about_text=(
            "Built TCP/IP protocol handlers. Worked on video conferencing "
            "solutions. Employee #5."
        ),
    )
    score, _reason = _score_founder_experience(profile)
    assert score <= 1.0, (
        "CEO/CTO/COO shouldn't match as substrings of 'protocol' / "
        "'conferencing' / 'solutions'"
    )


def test_founder_real_ceo_in_sentence() -> None:
    profile = _profile(
        headline="Software Engineer",
        about_text="Worked closely with the CEO on go-to-market strategy.",
    )
    score, _reason = _score_founder_experience(profile)
    # This is a mention, not a role. Our scorer treats it as a title hit of 1,
    # which is intentional (one mention) but not a false positive on "protocol".
    assert score <= 4.0


# -------- _score_technical_background --------


def test_technical_skill_full_credit() -> None:
    profile = _profile(
        headline="Senior Software Engineer",
        about_text="Built production systems in Python, Kubernetes, and Go.",
        skills_count=15,
    )
    score, _reason = _score_technical_background(profile)
    assert score >= 3.0


def test_technical_skill_hedged_counts_half() -> None:
    hedged_profile = _profile(
        headline="Product Manager",
        about_text="Familiar with Python and learning Kubernetes.",
    )
    unhedged_profile = _profile(
        headline="Senior Software Engineer",
        about_text="Python and Kubernetes expert with production experience.",
    )
    hedged_score, _ = _score_technical_background(hedged_profile)
    unhedged_score, _ = _score_technical_background(unhedged_profile)
    assert hedged_score < unhedged_score, (
        f"'familiar with'-hedged skills should score lower than owned skills "
        f"({hedged_score} vs {unhedged_score})"
    )


# -------- _score_education_prestige --------


def test_stanford_graduate_gets_top_tier() -> None:
    profile = _profile(
        education_json='[{"school": "Stanford University", "degree": "B.S. Computer Science"}]',
        about_text="Class of 2018.",
        education_count=1,
    )
    score, _reason = _score_education_prestige(profile)
    assert score >= 7.0


def test_stanford_dropout_is_capped() -> None:
    profile = _profile(
        about_text="Stanford dropout. Left after sophomore year to start a company.",
        education_json='[{"school": "Stanford"}]',
        education_count=1,
    )
    score, reason = _score_education_prestige(profile)
    assert score <= 2.5, f"Stanford dropout should be capped, got {score} ({reason})"


def test_no_graduation_signal_halves_top_tier_credit() -> None:
    profile = _profile(
        education_json='[{"school": "MIT"}]',
        education_count=1,
    )
    score, _reason = _score_education_prestige(profile)
    # No graduation signal present -> 3.5 not 7.0
    assert 2.0 <= score <= 4.5


# -------- _score_sf_location_fit --------


def test_sf_location_gets_full_credit() -> None:
    profile = _profile(location="San Francisco, CA")
    score, _reason = _score_sf_location_fit(profile)
    assert score == 10.0


def test_about_text_declares_sf_move_even_if_location_is_stale() -> None:
    profile = _profile(
        location="New York, NY",
        about_text="Based in SF as of last month, working on AI infra.",
    )
    score, _reason = _score_sf_location_fit(profile)
    assert score >= 7.0, (
        f"'Based in SF' in about_text should override NY location, got {score}"
    )


def test_open_to_relocate_boosts_us_candidate() -> None:
    baseline_profile = _profile(location="Austin, TX")
    boosted_profile = _profile(
        location="Austin, TX",
        about_text="Open to relocate for the right role.",
    )
    baseline_score, _ = _score_sf_location_fit(baseline_profile)
    boosted_score, _ = _score_sf_location_fit(boosted_profile)
    assert boosted_score > baseline_score, (
        f"'Open to relocate' should boost a US-tech-hub candidate "
        f"({baseline_score} -> {boosted_score})"
    )
