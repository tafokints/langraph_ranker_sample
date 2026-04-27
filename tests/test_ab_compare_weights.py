"""Unit tests for `scripts/ab_compare_weights.py`.

The script's I/O surface (LLM, DB) is exercised by the smoke test. These
tests pin the pure helpers — Jaccard, top-K extraction, markdown
formatting — so a refactor can't silently break the comparison output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import ab_compare_weights as ab_module


def test_jaccard_identical_returns_one() -> None:
    assert ab_module._jaccard(["a", "b", "c"], ["c", "b", "a"]) == 1.0


def test_jaccard_disjoint_returns_zero() -> None:
    assert ab_module._jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial_overlap() -> None:
    # |intersection| = 1 (b), |union| = 3 (a, b, c) -> 1/3
    overlap_value = ab_module._jaccard(["a", "b"], ["b", "c"])
    assert overlap_value == pytest.approx(1.0 / 3.0)


def test_jaccard_both_empty_returns_one() -> None:
    """Two empty top-K lists are trivially "the same"; treating this as
    NaN would make the markdown table awkward."""
    assert ab_module._jaccard([], []) == 1.0


def test_top_k_profile_ids_extracts_first_k() -> None:
    fake_result = {
        "ranked_candidates": [
            {"profile_id": "p_001"},
            {"profile_id": "p_002"},
            {"profile_id": "p_003"},
            {"profile_id": "p_004"},
        ]
    }
    assert ab_module._top_k_profile_ids(fake_result, 2) == ["p_001", "p_002"]
    assert ab_module._top_k_profile_ids(fake_result, 10) == [
        "p_001",
        "p_002",
        "p_003",
        "p_004",
    ]


def test_top_k_profile_ids_skips_blank_ids() -> None:
    fake_result = {
        "ranked_candidates": [
            {"profile_id": "p_001"},
            {"profile_id": ""},
            {"profile_id": "p_003"},
        ]
    }
    assert ab_module._top_k_profile_ids(fake_result, 5) == ["p_001", "p_003"]


def test_top_k_profile_ids_handles_missing_ranked_candidates() -> None:
    assert ab_module._top_k_profile_ids({}, 3) == []
    assert ab_module._top_k_profile_ids({"ranked_candidates": None}, 3) == []


def test_format_markdown_table_includes_header_and_rows() -> None:
    rows = [
        {"name": "alpha", "ids_a": ["p_001", "p_002"], "ids_b": ["p_002", "p_003"], "jaccard": 1.0 / 3.0},
        {"name": "beta", "ids_a": ["p_004"], "ids_b": ["p_004"], "jaccard": 1.0},
    ]
    table_text = ab_module._format_markdown_table(rows, top_k_value=2)
    assert "| Prompt | A top-2 | B top-2 |" in table_text
    assert "alpha" in table_text and "beta" in table_text
    assert "0.33" in table_text
    assert "1.00" in table_text
    # First row's ordering changed; second row's didn't
    alpha_line = next(line for line in table_text.splitlines() if "alpha" in line)
    beta_line = next(line for line in table_text.splitlines() if "beta" in line)
    assert alpha_line.endswith("yes |")
    assert beta_line.endswith("no |")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
