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


def save_weights(
    weights: Dict[str, float],
    defaults: Dict[str, float],
    version: str,
    n_labels: int,
    mae_before: Optional[float] = None,
    mae_after: Optional[float] = None,
    labeler: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a tuned weights JSON with provenance metadata.

    The prior file, if any, is copied to `config/weights.prev.json` so the last
    revision is always recoverable without git. Raises AssertionError if the
    provided weights fail validation (caller must fix before persisting).
    """
    _validate_weights_dict(weights, defaults)

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
