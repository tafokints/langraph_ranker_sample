"""Disk-backed load/save for DIMENSION_WEIGHTS used by the recruiter graph.

The rubric calibration loop writes a tuned weights JSON to `config/weights.json`
after fitting against human labels. At import time `langgraph_app` reads that
file (via `load_weights`) and uses it in place of the hardcoded defaults. If
the file is missing or malformed we fall back to the defaults so the app keeps
running with byte-identical behavior to the pre-calibration code.

Defaults are passed in explicitly (rather than imported from `langgraph_app`)
to avoid a circular import between this module and `langgraph_app`.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

WEIGHTS_SUM_TOLERANCE = 1e-3
WEIGHTS_FILE_RELATIVE_PATH = "config/weights.json"
WEIGHTS_BACKUP_RELATIVE_PATH = "config/weights.prev.json"

# Per-dimension gain/bias bounds. The calibrator fits an affine transform
# `adjusted = gain * raw + bias` per dimension. We constrain the gain to
# [GAIN_MIN, GAIN_MAX] so a single outlier can't produce a 50x multiplier,
# and bias to [BIAS_MIN, BIAS_MAX] so a misfit can't yank a dimension to the
# clip floor/ceiling. Identity transform (gain=1, bias=0) is back-compat.
GAIN_MIN = 0.25
GAIN_MAX = 3.0
BIAS_MIN = -5.0
BIAS_MAX = 5.0


def default_dimension_gains(dimension_keys: Any) -> Dict[str, Dict[str, float]]:
    """Identity transform per dimension: gain=1.0, bias=0.0.

    Accepts any iterable of dimension keys (e.g. the DIMENSION_KEYS tuple)
    and returns a freshly allocated dict so callers can mutate safely.
    """
    return {str(key): {"gain": 1.0, "bias": 0.0} for key in dimension_keys}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def weights_file_path() -> Path:
    return _project_root() / WEIGHTS_FILE_RELATIVE_PATH


def weights_backup_path() -> Path:
    return _project_root() / WEIGHTS_BACKUP_RELATIVE_PATH


def _validate_weights_dict(candidate_weights: Dict[str, float], expected_keys: Dict[str, float]) -> None:
    """Assert weights are non-negative, cover every expected key, and sum to ~1."""
    missing_keys = [key for key in expected_keys if key not in candidate_weights]
    assert not missing_keys, f"weights.json missing dimension keys: {missing_keys}"
    unexpected_keys = [key for key in candidate_weights if key not in expected_keys]
    assert not unexpected_keys, f"weights.json has unexpected keys: {unexpected_keys}"

    for key, value in candidate_weights.items():
        assert isinstance(value, (int, float)), f"weight '{key}' must be numeric, got {type(value).__name__}"
        assert value >= 0.0, f"weight '{key}' must be >= 0, got {value}"

    total_weight = sum(float(value) for value in candidate_weights.values())
    assert math.isclose(total_weight, 1.0, abs_tol=WEIGHTS_SUM_TOLERANCE), (
        f"weights.json weights sum to {total_weight:.4f}, expected 1.0 (+/- {WEIGHTS_SUM_TOLERANCE})"
    )


def load_weights(defaults: Dict[str, float]) -> Dict[str, float]:
    """Read `config/weights.json` and return the weight dict.

    Falls back to `defaults` (a caller-owned dict, e.g. DEFAULT_DIMENSION_WEIGHTS)
    when the file is missing, unreadable, malformed, or contains weights that
    fail validation. Never raises; calibration is opt-in and the pipeline must
    keep running without it.
    """
    weights_path = weights_file_path()
    if not weights_path.exists():
        return dict(defaults)

    try:
        with weights_path.open("r", encoding="utf-8") as weights_file_handle:
            payload = json.load(weights_file_handle)
    except (OSError, json.JSONDecodeError):
        return dict(defaults)

    raw_weights = payload.get("weights") if isinstance(payload, dict) else None
    if not isinstance(raw_weights, dict):
        return dict(defaults)

    try:
        coerced_weights = {str(key): float(value) for key, value in raw_weights.items()}
        _validate_weights_dict(coerced_weights, defaults)
    except (AssertionError, TypeError, ValueError):
        return dict(defaults)

    return coerced_weights


def _validate_gains_dict(
    candidate_gains: Dict[str, Dict[str, float]],
    expected_keys: Dict[str, float],
) -> None:
    """Assert gains cover every dimension and fall within bounded ranges."""
    missing_keys = [key for key in expected_keys if key not in candidate_gains]
    assert not missing_keys, f"weights.json gains missing dimension keys: {missing_keys}"
    unexpected_keys = [key for key in candidate_gains if key not in expected_keys]
    assert not unexpected_keys, f"weights.json gains has unexpected keys: {unexpected_keys}"
    for dimension_key, gain_entry in candidate_gains.items():
        assert isinstance(gain_entry, dict), (
            f"gain for '{dimension_key}' must be a dict, got {type(gain_entry).__name__}"
        )
        gain_value = gain_entry.get("gain")
        bias_value = gain_entry.get("bias")
        assert isinstance(gain_value, (int, float)), (
            f"gain.{dimension_key}.gain must be numeric, got {type(gain_value).__name__}"
        )
        assert isinstance(bias_value, (int, float)), (
            f"gain.{dimension_key}.bias must be numeric, got {type(bias_value).__name__}"
        )
        assert GAIN_MIN <= float(gain_value) <= GAIN_MAX, (
            f"gain.{dimension_key}.gain={gain_value} outside [{GAIN_MIN}, {GAIN_MAX}]"
        )
        assert BIAS_MIN <= float(bias_value) <= BIAS_MAX, (
            f"gain.{dimension_key}.bias={bias_value} outside [{BIAS_MIN}, {BIAS_MAX}]"
        )


def load_gains(defaults: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Read per-dimension gain/bias from `config/weights.json` if present.

    Falls back to identity (gain=1.0, bias=0.0) for every dimension key when
    the file is missing, malformed, or has no `gains` block. Never raises, so
    pre-v2 weights files remain usable unchanged.
    """
    weights_path = weights_file_path()
    if not weights_path.exists():
        return default_dimension_gains(defaults)

    try:
        with weights_path.open("r", encoding="utf-8") as weights_file_handle:
            payload = json.load(weights_file_handle)
    except (OSError, json.JSONDecodeError):
        return default_dimension_gains(defaults)

    raw_gains = payload.get("gains") if isinstance(payload, dict) else None
    if not isinstance(raw_gains, dict):
        return default_dimension_gains(defaults)

    try:
        coerced_gains: Dict[str, Dict[str, float]] = {}
        for dimension_key, gain_entry in raw_gains.items():
            if not isinstance(gain_entry, dict):
                raise TypeError(f"gain for '{dimension_key}' is not a dict")
            coerced_gains[str(dimension_key)] = {
                "gain": float(gain_entry.get("gain", 1.0)),
                "bias": float(gain_entry.get("bias", 0.0)),
            }
        _validate_gains_dict(coerced_gains, defaults)
    except (AssertionError, TypeError, ValueError):
        return default_dimension_gains(defaults)

    return coerced_gains


def save_weights(
    weights: Dict[str, float],
    defaults: Dict[str, float],
    version: str,
    n_labels: int,
    mae_before: Optional[float] = None,
    mae_after: Optional[float] = None,
    labeler: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    gains: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """Write a tuned weights JSON with provenance metadata.

    The prior file, if any, is copied to `config/weights.prev.json` so the last
    revision is always recoverable without git. Raises AssertionError if the
    provided weights / gains fail validation (caller must fix before persisting).
    When `gains` is None, identity gains are written so pre-v2 consumers still
    see a valid affine block.
    """
    _validate_weights_dict(weights, defaults)

    resolved_gains = gains if gains is not None else default_dimension_gains(defaults)
    _validate_gains_dict(resolved_gains, defaults)

    weights_path = weights_file_path()
    backup_path = weights_backup_path()
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        previous_payload = weights_path.read_bytes()
        backup_path.write_bytes(previous_payload)

    payload: Dict[str, Any] = {
        "version": version,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_labels": int(n_labels),
        "mae_before": mae_before,
        "mae_after": mae_after,
        "labeler": labeler,
        "weights": {str(key): round(float(value), 6) for key, value in weights.items()},
        "gains": {
            str(key): {
                "gain": round(float(entry["gain"]), 6),
                "bias": round(float(entry["bias"]), 6),
            }
            for key, entry in resolved_gains.items()
        },
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata

    with weights_path.open("w", encoding="utf-8") as weights_file_handle:
        json.dump(payload, weights_file_handle, indent=2, sort_keys=False)
        weights_file_handle.write("\n")

    return weights_path


def current_version() -> Optional[str]:
    """Return the `version` field of the current weights file, if present."""
    weights_path = weights_file_path()
    if not weights_path.exists():
        return None
    try:
        with weights_path.open("r", encoding="utf-8") as weights_file_handle:
            payload = json.load(weights_file_handle)
    except (OSError, json.JSONDecodeError):
        return None
    version_value = payload.get("version") if isinstance(payload, dict) else None
    return str(version_value) if isinstance(version_value, str) else None


def next_version() -> str:
    """Produce the next 'vN' string based on the current file's version."""
    existing_version = current_version()
    if not existing_version or not existing_version.lower().startswith("v"):
        return "v1"
    numeric_suffix = existing_version[1:]
    try:
        next_numeric = int(numeric_suffix) + 1
    except ValueError:
        return "v1"
    return f"v{next_numeric}"
