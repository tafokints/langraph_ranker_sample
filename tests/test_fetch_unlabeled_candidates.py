"""Unit tests for `fetch_unlabeled_candidates`.

Mocks `_open_connection` so the test never touches a real MySQL instance.
Validates:
  - empty / blank labeler raises ValueError
  - rows are mapped onto the candidate dict shape the queue UI expects
  - prefer_diverse=True/False both produce a valid SQL invocation
  - limit is bounded to a sane range (defensive against UI bugs)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import pytest

from src import labels_store


class _StubCursor:
    def __init__(self, scripted_rows: Iterable[Tuple[Any, ...]]) -> None:
        self.scripted_rows: List[Tuple[Any, ...]] = list(scripted_rows)
        self.last_sql: str = ""
        self.last_params: Tuple[Any, ...] = ()

    def execute(self, sql_text: str, params: Tuple[Any, ...] = ()) -> None:
        self.last_sql = sql_text
        self.last_params = params

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return self.scripted_rows

    def __enter__(self) -> "_StubCursor":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _StubConnection:
    def __init__(self, scripted_rows: Iterable[Tuple[Any, ...]]) -> None:
        self.cursor_obj = _StubCursor(scripted_rows)

    def cursor(self) -> _StubCursor:
        return self.cursor_obj

    def __enter__(self) -> "_StubConnection":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _patch_connection(monkeypatch: pytest.MonkeyPatch, scripted_rows: Iterable[Tuple[Any, ...]]) -> _StubConnection:
    stub_connection = _StubConnection(scripted_rows)
    monkeypatch.setattr(labels_store, "_open_connection", lambda: stub_connection)
    monkeypatch.setattr(labels_store, "ensure_labels_table", lambda: None)
    return stub_connection


def test_blank_labeler_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connection(monkeypatch, [])
    with pytest.raises(ValueError):
        labels_store.fetch_unlabeled_candidates(labeler="   ", limit=5)


def test_returns_canonical_candidate_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_row = (
        "p_001",
        "Ada Lovelace",
        "Software engineer",
        "London",
        "About text here.",
        12,
        4,
        2,
        '[{"company": "Acme"}]',
        '[{"school": "MIT"}]',
    )
    stub_connection = _patch_connection(monkeypatch, [sample_row])

    candidates = labels_store.fetch_unlabeled_candidates(
        labeler="me", limit=5, prefer_diverse=True
    )

    assert len(candidates) == 1
    only_candidate = candidates[0]
    assert only_candidate["profile_id"] == "p_001"
    assert only_candidate["name"] == "Ada Lovelace"
    assert only_candidate["headline"] == "Software engineer"
    assert only_candidate["location"] == "London"
    assert only_candidate["about_text"] == "About text here."
    assert only_candidate["skills_count"] == 12
    assert only_candidate["experience_count"] == 4
    assert only_candidate["education_count"] == 2
    assert only_candidate["experience_json"] == '[{"company": "Acme"}]'
    assert only_candidate["education_json"] == '[{"school": "MIT"}]'

    assert "%s" in stub_connection.cursor_obj.last_sql
    assert "linkedin_api_profiles_parsed" in stub_connection.cursor_obj.last_sql
    assert "LEFT JOIN" in stub_connection.cursor_obj.last_sql
    assert stub_connection.cursor_obj.last_params[0] == "me"
    assert int(stub_connection.cursor_obj.last_params[1]) == 5


def test_prefer_diverse_changes_ordering_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_connection_diverse = _patch_connection(monkeypatch, [])
    labels_store.fetch_unlabeled_candidates(
        labeler="me", limit=3, prefer_diverse=True
    )
    sql_for_diverse = stub_connection_diverse.cursor_obj.last_sql
    assert "about_char_count" in sql_for_diverse or "about_text" in sql_for_diverse
    assert "RAND()" not in sql_for_diverse

    stub_connection_random = _patch_connection(monkeypatch, [])
    labels_store.fetch_unlabeled_candidates(
        labeler="me", limit=3, prefer_diverse=False
    )
    sql_for_random = stub_connection_random.cursor_obj.last_sql
    assert "RAND()" in sql_for_random


def test_limit_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_connection = _patch_connection(monkeypatch, [])
    labels_store.fetch_unlabeled_candidates(labeler="me", limit=10000)
    bounded_limit = int(stub_connection.cursor_obj.last_params[1])
    assert bounded_limit <= 200, "limit should be clamped to <= 200"

    stub_connection_low = _patch_connection(monkeypatch, [])
    labels_store.fetch_unlabeled_candidates(labeler="me", limit=0)
    bounded_limit_low = int(stub_connection_low.cursor_obj.last_params[1])
    assert bounded_limit_low >= 1, "limit should be clamped to >= 1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
