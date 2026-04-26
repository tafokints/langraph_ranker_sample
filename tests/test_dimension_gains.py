"""Unit tests for per-dimension affine gains.

Covers the back-compat contract for `config/weights.json` (missing `gains`
block => identity), the weights-loader round-trip, and the aggregation
helper's invariance under identity gains.

Run with:  python -m pytest tests/test_dimension_gains.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from src.langgraph_app import (
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSION_KEYS,
    _aggregate_rank_score,
    _apply_dimension_gains,
)
from src.weights_loader import (
    BIAS_MAX,
    BIAS_MIN,
    GAIN_MAX,
    GAIN_MIN,
    default_dimension_gains,
    load_gains,
    save_weights,
    weights_file_path,
)


@pytest.fixture
def weights_file_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect WEIGHTS_FILE_RELATIVE_PATH writes to a tmp dir so tests don't
    clobber the real `config/weights.json`. We patch the `_project_root` helper
    so both read and write go through the sandbox, and restore afterwards."""
    import src.weights_loader as weights_loader_module

    sandbox_root = tmp_path
    (sandbox_root / "config").mkdir()

    def _sandbox_project_root() -> Path:
        return sandbox_root

    monkeypatch.setattr(weights_loader_module, "_project_root", _sandbox_project_root)
    yield sandbox_root


def _sample_dims(scale: float = 5.0) -> Dict[str, float]:
    return {key: scale for key in DIMENSION_KEYS}


def test_default_gains_are_identity() -> None:
    identity_gains = default_dimension_gains(DEFAULT_DIMENSION_WEIGHTS)
    for key in DIMENSION_KEYS:
        assert identity_gains[key] == {"gain": 1.0, "bias": 0.0}


def test_apply_identity_gains_is_noop() -> None:
    identity_gains = default_dimension_gains(DEFAULT_DIMENSION_WEIGHTS)
    sample_dimension_scores = _sample_dims(scale=6.2)
    adjusted_dim_scores = _apply_dimension_gains(
        sample_dimension_scores, gains=identity_gains
    )
    for key in DIMENSION_KEYS:
        assert adjusted_dim_scores[key] == pytest.approx(6.2)


def test_apply_gain_scales_and_clips() -> None:
    scaled_gains = {
        key: {"gain": 2.0, "bias": 0.0} for key in DIMENSION_KEYS
    }
    # At raw=6.0 -> adjusted = 12.0 -> clipped to 10.0.
    dimension_scores_high = _sample_dims(scale=6.0)
    adjusted_dim_scores_high = _apply_dimension_gains(dimension_scores_high, gains=scaled_gains)
    for key in DIMENSION_KEYS:
        assert adjusted_dim_scores_high[key] == pytest.approx(10.0)

    dimension_scores_low = _sample_dims(scale=2.0)
    adjusted_dim_scores_low = _apply_dimension_gains(dimension_scores_low, gains=scaled_gains)
    for key in DIMENSION_KEYS:
        assert adjusted_dim_scores_low[key] == pytest.approx(4.0)


def test_apply_bias_shifts_and_clips_negative() -> None:
    bias_gains = {
        key: {"gain": 1.0, "bias": -3.0} for key in DIMENSION_KEYS
    }
    dimension_scores_low = _sample_dims(scale=2.0)
    adjusted_low = _apply_dimension_gains(dimension_scores_low, gains=bias_gains)
    for key in DIMENSION_KEYS:
        assert adjusted_low[key] == pytest.approx(0.0), (
            "bias of -3.0 on raw=2.0 should clip to 0.0 not -1.0"
        )


def test_aggregate_rank_score_with_identity_matches_raw_weighted_sum() -> None:
    # The contract: applying identity gains (gain=1, bias=0 per dimension)
    # must reduce `_aggregate_rank_score` to the pure weighted sum, clipped
    # to [0, 10] and rounded to 2 decimals. This is independent of whatever
    # is currently sitting in config/weights.json — that's why we build the
    # expected value from scratch instead of calling _aggregate_rank_score
    # with gains=None (which picks up the module-level DIMENSION_GAINS).
    identity_gains = default_dimension_gains(DEFAULT_DIMENSION_WEIGHTS)
    dimension_scores_mixed = {
        "phd_researcher": 4.0,
        "sf_location_fit": 10.0,
        "technical_background": 7.0,
        "education_prestige": 2.0,
        "founder_experience": 9.0,
    }
    score_with_identity = _aggregate_rank_score(
        dimension_scores_mixed,
        weights=DEFAULT_DIMENSION_WEIGHTS,
        gains=identity_gains,
    )
    expected_weighted_sum = sum(
        DEFAULT_DIMENSION_WEIGHTS[dimension_key] * dimension_scores_mixed[dimension_key]
        for dimension_key in DEFAULT_DIMENSION_WEIGHTS
    )
    expected_weighted_sum = round(max(0.0, min(10.0, expected_weighted_sum)), 2)
    assert score_with_identity == pytest.approx(expected_weighted_sum)


def test_load_gains_missing_file_falls_back_to_identity(weights_file_sandbox: Path) -> None:
    assert not weights_file_path().exists()
    loaded_gains = load_gains(DEFAULT_DIMENSION_WEIGHTS)
    for key in DIMENSION_KEYS:
        assert loaded_gains[key] == {"gain": 1.0, "bias": 0.0}


def test_load_gains_v1_file_without_gains_block_is_identity(
    weights_file_sandbox: Path,
) -> None:
    v1_payload = {
        "version": "v1",
        "weights": {
            "technical_background": 0.30,
            "founder_experience": 0.25,
            "phd_researcher": 0.15,
            "education_prestige": 0.15,
            "sf_location_fit": 0.15,
        },
    }
    weights_file_path().write_text(json.dumps(v1_payload), encoding="utf-8")
    loaded_gains = load_gains(DEFAULT_DIMENSION_WEIGHTS)
    for key in DIMENSION_KEYS:
        assert loaded_gains[key] == {"gain": 1.0, "bias": 0.0}


def test_save_and_load_gains_round_trip(weights_file_sandbox: Path) -> None:
    fitted_gains = {
        "technical_background": {"gain": 1.2, "bias": -0.5},
        "founder_experience": {"gain": 0.8, "bias": 1.0},
        "phd_researcher": {"gain": 1.5, "bias": 0.0},
        "education_prestige": {"gain": 1.0, "bias": 0.25},
        "sf_location_fit": {"gain": 0.9, "bias": -0.1},
    }
    save_weights(
        weights=DEFAULT_DIMENSION_WEIGHTS,
        defaults=DEFAULT_DIMENSION_WEIGHTS,
        version="v2",
        n_labels=17,
        mae_before=2.1,
        mae_after=1.4,
        labeler="tester",
        gains=fitted_gains,
    )
    reloaded_gains = load_gains(DEFAULT_DIMENSION_WEIGHTS)
    for key, expected_entry in fitted_gains.items():
        assert reloaded_gains[key]["gain"] == pytest.approx(expected_entry["gain"])
        assert reloaded_gains[key]["bias"] == pytest.approx(expected_entry["bias"])


def test_save_gains_rejects_out_of_range(weights_file_sandbox: Path) -> None:
    too_large_gain = {
        "technical_background": {"gain": GAIN_MAX + 1.0, "bias": 0.0},
        "founder_experience": {"gain": 1.0, "bias": 0.0},
        "phd_researcher": {"gain": 1.0, "bias": 0.0},
        "education_prestige": {"gain": 1.0, "bias": 0.0},
        "sf_location_fit": {"gain": 1.0, "bias": 0.0},
    }
    with pytest.raises(AssertionError):
        save_weights(
            weights=DEFAULT_DIMENSION_WEIGHTS,
            defaults=DEFAULT_DIMENSION_WEIGHTS,
            version="v2",
            n_labels=1,
            gains=too_large_gain,
        )

    too_negative_bias = {
        "technical_background": {"gain": 1.0, "bias": BIAS_MIN - 1.0},
        "founder_experience": {"gain": 1.0, "bias": 0.0},
        "phd_researcher": {"gain": 1.0, "bias": 0.0},
        "education_prestige": {"gain": 1.0, "bias": 0.0},
        "sf_location_fit": {"gain": 1.0, "bias": 0.0},
    }
    with pytest.raises(AssertionError):
        save_weights(
            weights=DEFAULT_DIMENSION_WEIGHTS,
            defaults=DEFAULT_DIMENSION_WEIGHTS,
            version="v2",
            n_labels=1,
            gains=too_negative_bias,
        )


def test_load_gains_malformed_falls_back_safely(weights_file_sandbox: Path) -> None:
    malformed_payload = {
        "version": "v2",
        "weights": dict(DEFAULT_DIMENSION_WEIGHTS),
        "gains": "not a dict",
    }
    weights_file_path().write_text(json.dumps(malformed_payload), encoding="utf-8")
    loaded_gains = load_gains(DEFAULT_DIMENSION_WEIGHTS)
    for key in DIMENSION_KEYS:
        assert loaded_gains[key] == {"gain": 1.0, "bias": 0.0}


def test_bias_bounds_are_sane() -> None:
    # Basic sanity so a future refactor can't set BIAS_MAX < BIAS_MIN.
    assert BIAS_MIN < BIAS_MAX
    assert GAIN_MIN < GAIN_MAX
    assert GAIN_MIN > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
