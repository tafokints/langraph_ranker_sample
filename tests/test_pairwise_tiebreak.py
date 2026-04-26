"""Unit tests for `_pairwise_tiebreak_adjacent`.

The helper makes real LLM calls in production. Here we swap in a fake
`with_structured_output(...)` chain so we can assert swapping logic, the
near-tie threshold, the max-call budget, and the min-confidence filter
without touching OpenAI.

Run with:  python -m pytest tests/test_pairwise_tiebreak.py -v
"""

from __future__ import annotations

from typing import Any, List

import pytest

from src.langgraph_app import (
    PAIRWISE_TIEBREAK_MAX_CALLS,
    PAIRWISE_TIEBREAK_MIN_CONFIDENCE,
    PAIRWISE_TIEBREAK_THRESHOLD,
    _pairwise_tiebreak_adjacent,
)
from src.schemas import PairwiseDecision


class _StubStructuredLLM:
    """Returns a scripted sequence of PairwiseDecision values.

    Matches the real `.with_structured_output(...).invoke(messages, config)`
    signature — we accept an optional second positional arg (the config dict
    that `_llm_invoke_config()` now passes through for token-usage tracking)
    and silently ignore it. Without this, production would hit `TypeError`
    and the exception handler would swallow the swap.
    """

    def __init__(self, decisions: List[PairwiseDecision]) -> None:
        self._decisions = list(decisions)
        self.call_count = 0

    def invoke(
        self,
        _messages: Any,
        _config: Any = None,
        **_kwargs: Any,
    ) -> PairwiseDecision:
        if not self._decisions:
            raise AssertionError("stub LLM ran out of scripted decisions")
        self.call_count += 1
        return self._decisions.pop(0)


class _StubChatModel:
    """Minimal stand-in for `ChatOpenAI` that wires `with_structured_output`
    to our scripted `_StubStructuredLLM`."""

    def __init__(self, decisions: List[PairwiseDecision]) -> None:
        self.stub_structured = _StubStructuredLLM(decisions)

    def with_structured_output(self, _schema: Any) -> _StubStructuredLLM:
        return self.stub_structured


def _candidate(profile_id: str, rank_score: float) -> dict:
    """Build a minimal ranked candidate dict with the fields the helper
    reads. All other fields are stubbed to realistic shapes."""
    return {
        "profile_id": profile_id,
        "name": f"Candidate {profile_id}",
        "headline": "",
        "location": "",
        "about_text": "",
        "skills_count": 0,
        "experience_count": 0,
        "education_count": 0,
        "relevance_score": 0,
        "rank_score": rank_score,
        "match_reasons": [],
        "risks": [],
        "dimension_scores": {
            "phd_researcher": 0.0,
            "sf_location_fit": 0.0,
            "technical_background": 0.0,
            "education_prestige": 0.0,
            "founder_experience": 0.0,
        },
        "dimension_reasons": {},
    }


def test_swap_happens_when_llm_picks_lower_with_high_confidence() -> None:
    ranked = [_candidate("A", 5.0), _candidate("B", 4.9)]  # gap 0.1 < 0.3
    stub_model = _StubChatModel(
        [PairwiseDecision(winner_profile_id="B", rationale="stronger", confidence=0.9)]
    )

    new_ranking, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )

    assert calls_made == 1
    assert swaps_applied == 1
    assert [c["profile_id"] for c in new_ranking] == ["B", "A"]


def test_no_swap_when_gap_exceeds_threshold() -> None:
    # Gap 0.5 > PAIRWISE_TIEBREAK_THRESHOLD (0.3) -> never call the LLM.
    assert PAIRWISE_TIEBREAK_THRESHOLD == 0.3
    ranked = [_candidate("A", 5.0), _candidate("B", 4.5)]
    stub_model = _StubChatModel([])  # no scripted decisions -> should never invoke

    new_ranking, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )

    assert calls_made == 0
    assert swaps_applied == 0
    assert [c["profile_id"] for c in new_ranking] == ["A", "B"]


def test_no_swap_when_llm_picks_higher() -> None:
    ranked = [_candidate("A", 5.0), _candidate("B", 4.9)]
    stub_model = _StubChatModel(
        [PairwiseDecision(winner_profile_id="A", rationale="stronger", confidence=0.9)]
    )

    _, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )

    assert calls_made == 1
    assert swaps_applied == 0


def test_low_confidence_does_not_swap() -> None:
    ranked = [_candidate("A", 5.0), _candidate("B", 4.9)]
    low_conf = max(0.0, PAIRWISE_TIEBREAK_MIN_CONFIDENCE - 0.1)
    stub_model = _StubChatModel(
        [PairwiseDecision(winner_profile_id="B", rationale="weak", confidence=low_conf)]
    )

    new_ranking, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )

    assert calls_made == 1
    assert swaps_applied == 0
    assert [c["profile_id"] for c in new_ranking] == ["A", "B"]


def test_budget_cap_limits_calls() -> None:
    # 6 near-tie candidates would imply up to 5 pairs; budget caps at
    # PAIRWISE_TIEBREAK_MAX_CALLS. Each call returns "keep higher" so we
    # count calls only.
    ranked = [_candidate(chr(ord("A") + i), 5.0 - 0.1 * i) for i in range(6)]
    # Decisions: always keep higher (winner = previous). Must return enough
    # to satisfy every call the helper makes.
    decisions = [
        PairwiseDecision(winner_profile_id=chr(ord("A") + i), rationale="", confidence=0.9)
        for i in range(PAIRWISE_TIEBREAK_MAX_CALLS)
    ]
    stub_model = _StubChatModel(decisions)

    _, calls_made, _ = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )

    assert calls_made <= PAIRWISE_TIEBREAK_MAX_CALLS


def test_empty_list_is_noop() -> None:
    stub_model = _StubChatModel([])
    new_ranking, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=[],
        question_text="role brief",
    )
    assert new_ranking == []
    assert calls_made == 0
    assert swaps_applied == 0


def test_single_candidate_is_noop() -> None:
    ranked = [_candidate("A", 5.0)]
    stub_model = _StubChatModel([])
    new_ranking, calls_made, swaps_applied = _pairwise_tiebreak_adjacent(
        llm_model=stub_model,  # type: ignore[arg-type]
        ranked_candidates=ranked,
        question_text="role brief",
    )
    assert [c["profile_id"] for c in new_ranking] == ["A"]
    assert calls_made == 0
    assert swaps_applied == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
