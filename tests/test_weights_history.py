"""Unit tests for the weights-history archive helpers.

Covers:
- `archive_weights_snapshot` writes `config/weights.history/<date>_<sha>.json`
  byte-identical to the source `config/weights.json`.
- It returns None when the source file doesn't exist (so calibrate.py can
  call it unconditionally without crashing on first-ever run).
- `find_most_recent_archive` picks the newest archive by ISO date and ignores
  files whose name doesn't match `<YYYY-MM-DD>_<sha>.json`.
- `find_most_recent_archive(exclude_path=...)` skips the path you just wrote
  so callers can compare against the *previous* fit, not the current one.

Run with:  python -m pytest tests/test_weights_history.py -v
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.weights_loader import (
    archive_weights_snapshot,
    find_most_recent_archive,
    list_archived_weights,
)


def test_archive_weights_snapshot_writes_byte_identical_copy(tmp_path: Path) -> None:
    source_weights = tmp_path / "weights.json"
    source_weights.write_text('{"weights": {"phd_researcher": 0.2}}\n', encoding="utf-8")
    archive_dir = tmp_path / "weights.history"

    archive_path = archive_weights_snapshot(
        source_path=source_weights,
        archive_dir=archive_dir,
        timestamp=datetime(2026, 4, 25, 12, 0, 0),
        short_sha="abc1234",
    )

    assert archive_path is not None
    assert archive_path.name == "2026-04-25_abc1234.json"
    assert archive_path.parent == archive_dir
    assert archive_path.read_bytes() == source_weights.read_bytes()


def test_archive_weights_snapshot_returns_none_for_missing_source(
    tmp_path: Path,
) -> None:
    """First-ever fit will not have a previous weights file. Archiving in that
    case must be a no-op rather than a hard failure — calibrate.py calls
    archive unconditionally after save_weights."""
    nonexistent_source = tmp_path / "does_not_exist.json"
    archive_dir = tmp_path / "weights.history"

    result = archive_weights_snapshot(
        source_path=nonexistent_source,
        archive_dir=archive_dir,
    )

    assert result is None


def test_archive_weights_snapshot_overwrites_same_day_same_sha(
    tmp_path: Path,
) -> None:
    """Same date + same git SHA = same fit attempt. Most recent content wins."""
    source_weights = tmp_path / "weights.json"
    archive_dir = tmp_path / "weights.history"
    fixed_timestamp = datetime(2026, 4, 25, 12, 0, 0)

    source_weights.write_text("first content\n", encoding="utf-8")
    first_archive_path = archive_weights_snapshot(
        source_path=source_weights,
        archive_dir=archive_dir,
        timestamp=fixed_timestamp,
        short_sha="abc1234",
    )

    source_weights.write_text("second content\n", encoding="utf-8")
    second_archive_path = archive_weights_snapshot(
        source_path=source_weights,
        archive_dir=archive_dir,
        timestamp=fixed_timestamp,
        short_sha="abc1234",
    )

    assert first_archive_path == second_archive_path
    assert second_archive_path.read_text(encoding="utf-8") == "second content\n"


def test_list_archived_weights_skips_unrelated_filenames(tmp_path: Path) -> None:
    archive_dir = tmp_path / "weights.history"
    archive_dir.mkdir()

    valid_archive = archive_dir / "2026-04-20_aaaaaaa.json"
    valid_archive.write_text("{}", encoding="utf-8")

    junk_filename = archive_dir / "README.md"
    junk_filename.write_text("notes", encoding="utf-8")

    bad_date_filename = archive_dir / "2026-13-99_aaaaaaa.json"
    bad_date_filename.write_text("{}", encoding="utf-8")

    archived_entries = list_archived_weights(archive_dir=archive_dir)

    matched_paths = [path for _date, path in archived_entries]
    assert valid_archive in matched_paths
    assert junk_filename not in matched_paths
    assert bad_date_filename not in matched_paths


def test_find_most_recent_archive_picks_latest_date(tmp_path: Path) -> None:
    archive_dir = tmp_path / "weights.history"
    archive_dir.mkdir()

    older = archive_dir / "2026-04-10_aaaa1234.json"
    older.write_text("{}", encoding="utf-8")
    newer = archive_dir / "2026-04-20_bbbb5678.json"
    newer.write_text("{}", encoding="utf-8")

    most_recent = find_most_recent_archive(archive_dir=archive_dir)

    assert most_recent == newer


def test_find_most_recent_archive_excludes_specified_path(tmp_path: Path) -> None:
    """Right after archiving the just-written snapshot, calibrate.py wants the
    *previous* fit to compare against — not the one it just wrote."""
    archive_dir = tmp_path / "weights.history"
    archive_dir.mkdir()

    older = archive_dir / "2026-04-10_aaaa1234.json"
    older.write_text("{}", encoding="utf-8")
    newer = archive_dir / "2026-04-20_bbbb5678.json"
    newer.write_text("{}", encoding="utf-8")

    found = find_most_recent_archive(exclude_path=newer, archive_dir=archive_dir)

    assert found == older


def test_find_most_recent_archive_returns_none_when_only_excluded_exists(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "weights.history"
    archive_dir.mkdir()
    only_archive = archive_dir / "2026-04-20_cccc9999.json"
    only_archive.write_text("{}", encoding="utf-8")

    found = find_most_recent_archive(
        exclude_path=only_archive, archive_dir=archive_dir
    )

    assert found is None


def test_find_most_recent_archive_returns_none_when_dir_missing(
    tmp_path: Path,
) -> None:
    """`config/weights.history` may not exist on the very first calibration —
    listing has to handle that without raising."""
    nonexistent_dir = tmp_path / "no_such_dir"

    found = find_most_recent_archive(archive_dir=nonexistent_dir)

    assert found is None
