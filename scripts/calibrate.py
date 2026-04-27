"""Fit DIMENSION_WEIGHTS against human labels.

Reads rows from `recruiter_rubric_labels`, re-computes the heuristic per-dimension
scores for each labeled profile (from the current code revision), and emits:

1. A per-dimension calibration report (MAE, signed bias, Spearman, N, std devs,
   flags for dimensions that look miscalibrated).
2. A weighted-least-squares fit of `DIMENSION_WEIGHTS` against the human overall
   score, constrained to the probability simplex (w_i >= 0, sum_i w_i = 1).
3. A markdown report written to `reports/calibration_<YYYY-MM-DD_HHMM>.md`.
4. Unless `--dry-run`, a new `config/weights.json` with `overall_mae_before`
   (current weights) and `overall_mae_after` (fitted weights) for provenance.

Usage:

    python scripts/calibrate.py                  # fit using all labels
    python scripts/calibrate.py --labeler me     # only labels from 'me'
    python scripts/calibrate.py --dry-run        # report only; do not write weights
    python scripts/calibrate.py --min-labels 20  # require >=20 labels before fitting
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

from src.labels_store import load_labels
from src.langgraph_app import (
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSION_GAINS,
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    DIMENSION_WEIGHTS,
    _aggregate_rank_score,
    _apply_dimension_gains,
    _deterministic_rank,
)
from src.retriever import _open_connection as open_profile_connection
from src.weights_loader import (
    BIAS_MAX,
    BIAS_MIN,
    GAIN_MAX,
    GAIN_MIN,
    archive_weights_snapshot,
    default_dimension_gains,
    find_most_recent_archive,
    load_gains,
    load_weights,
    next_version,
    save_weights,
    weights_file_path,
)

DEFAULT_MIN_LABELS = 15
BIAS_FLAG_THRESHOLD = 1.5
MAE_FLAG_THRESHOLD = 2.0
LOW_STD_WARNING_THRESHOLD = 0.5
REPORT_DIR_NAME = "reports"


PROFILE_FETCH_SQL = """
SELECT
    profile_id,
    name,
    headline,
    location,
    about_text,
    skills_count,
    experience_count,
    education_count,
    experience_json,
    education_json
FROM linkedin_api_profiles_parsed
WHERE profile_id = %s
"""


@dataclass
class LabeledSample:
    profile_id: str
    labeler: str
    human_dim_scores: Dict[str, float]
    human_overall: float
    heuristic_dim_scores: Dict[str, float]
    heuristic_overall_current_weights: float


def _fetch_profile(database_connection: Any, profile_id: str) -> Optional[Dict[str, Any]]:
    with database_connection.cursor() as database_cursor:
        database_cursor.execute(PROFILE_FETCH_SQL, (profile_id,))
        row = database_cursor.fetchone()
    if not row:
        return None
    return {
        "profile_id": row[0],
        "name": row[1] or "",
        "headline": row[2] or "",
        "location": row[3] or "",
        "about_text": row[4] or "",
        "skills_count": int(row[5] or 0),
        "experience_count": int(row[6] or 0),
        "education_count": int(row[7] or 0),
        "experience_json": row[8] or "",
        "education_json": row[9] or "",
        "relevance_score": 0,
    }


def _collect_samples(labeler: Optional[str]) -> Tuple[List[LabeledSample], List[str]]:
    """Return (samples, warnings). Missing profiles / malformed labels become warnings."""
    label_rows = load_labels(labeler=labeler, latest_per_profile_labeler=True)
    warnings_emitted: List[str] = []
    samples: List[LabeledSample] = []

    if not label_rows:
        return samples, warnings_emitted

    with open_profile_connection() as database_connection:
        for label_row in label_rows:
            profile_id_value = str(label_row.get("profile_id") or "").strip()
            if not profile_id_value:
                warnings_emitted.append("Skipping label with empty profile_id.")
                continue

            raw_human_dims = label_row.get("dim_scores") or {}
            if not isinstance(raw_human_dims, dict):
                warnings_emitted.append(f"Skipping label {profile_id_value}: dim_scores not a dict.")
                continue

            missing_dimension_keys = [key for key in DIMENSION_KEYS if key not in raw_human_dims]
            if missing_dimension_keys:
                warnings_emitted.append(
                    f"Skipping label {profile_id_value}: missing dims {missing_dimension_keys}."
                )
                continue

            try:
                human_dim_scores = {
                    key: float(raw_human_dims[key]) for key in DIMENSION_KEYS
                }
                human_overall_score = float(label_row.get("overall_score") or 0.0)
            except (TypeError, ValueError):
                warnings_emitted.append(f"Skipping label {profile_id_value}: non-numeric scores.")
                continue

            profile_record = _fetch_profile(database_connection, profile_id_value)
            if profile_record is None:
                warnings_emitted.append(
                    f"Skipping label {profile_id_value}: profile not found in linkedin_api_profiles_parsed."
                )
                continue

            ranked_baseline = _deterministic_rank(profile_record)
            heuristic_dim_scores = dict(ranked_baseline.get("dimension_scores") or {})
            heuristic_overall_current = _aggregate_rank_score(heuristic_dim_scores)

            samples.append(
                LabeledSample(
                    profile_id=profile_id_value,
                    labeler=str(label_row.get("labeler") or ""),
                    human_dim_scores=human_dim_scores,
                    human_overall=human_overall_score,
                    heuristic_dim_scores=heuristic_dim_scores,
                    heuristic_overall_current_weights=heuristic_overall_current,
                )
            )

    return samples, warnings_emitted


def _per_dimension_metrics(samples: Sequence[LabeledSample]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for dimension_key in DIMENSION_KEYS:
        heuristic_values = np.array(
            [sample.heuristic_dim_scores.get(dimension_key, 0.0) for sample in samples],
            dtype=float,
        )
        human_values = np.array(
            [sample.human_dim_scores.get(dimension_key, 0.0) for sample in samples],
            dtype=float,
        )
        differences = heuristic_values - human_values
        mean_absolute_error = float(np.mean(np.abs(differences))) if differences.size else 0.0
        signed_bias = float(np.mean(differences)) if differences.size else 0.0
        heuristic_std = float(np.std(heuristic_values, ddof=0)) if heuristic_values.size else 0.0
        human_std = float(np.std(human_values, ddof=0)) if human_values.size else 0.0

        if heuristic_values.size >= 2 and heuristic_std > 0 and human_std > 0:
            spearman_result = spearmanr(heuristic_values, human_values)
            spearman_rho = float(spearman_result.correlation) if spearman_result.correlation is not None else float("nan")
        else:
            spearman_rho = float("nan")

        metrics[dimension_key] = {
            "n": int(len(samples)),
            "mae": mean_absolute_error,
            "bias": signed_bias,
            "spearman": spearman_rho,
            "heuristic_std": heuristic_std,
            "human_std": human_std,
        }
    return metrics


def _project_onto_probability_simplex(vector_values: np.ndarray) -> np.ndarray:
    """Closed-form Euclidean projection onto {w in R^n : w_i >= 0, sum w_i = 1}.

    Reference: Duchi et al. 2008, 'Efficient Projections onto the l1-Ball'.
    """
    n = vector_values.size
    if n == 0:
        return vector_values
    sorted_descending = np.sort(vector_values)[::-1]
    cumulative_sum = np.cumsum(sorted_descending)
    rho_candidates = sorted_descending - (cumulative_sum - 1.0) / np.arange(1, n + 1)
    positive_mask = rho_candidates > 0
    if not np.any(positive_mask):
        uniform_projection = np.ones_like(vector_values) / float(n)
        return uniform_projection
    rho_index = np.max(np.where(positive_mask)[0])
    tau_value = (cumulative_sum[rho_index] - 1.0) / float(rho_index + 1)
    projected_values = np.maximum(vector_values - tau_value, 0.0)
    # Numerical cleanup: if everything was clipped to zero (shouldn't happen), uniform.
    total_mass = float(np.sum(projected_values))
    if total_mass <= 0.0:
        return np.ones_like(vector_values) / float(n)
    return projected_values / total_mass


def _fit_dimension_gains(
    samples: Sequence[LabeledSample],
) -> Dict[str, Dict[str, float]]:
    """Per-dimension OLS fit of the affine transform `gain * raw + bias`.

    For each dimension we solve argmin over (g, b) of sum_s (g * raw_s + b - human_s)^2
    independently. Degenerate columns (zero heuristic variance) fall back to
    identity so we don't produce inf/nan when every profile got the same score.
    Results are clamped to [GAIN_MIN, GAIN_MAX] and [BIAS_MIN, BIAS_MAX] so
    a single outlier can't yank a dimension off its sensible range.
    """
    fitted_gains: Dict[str, Dict[str, float]] = {}
    for dimension_key in DIMENSION_KEYS:
        raw_values = np.array(
            [sample.heuristic_dim_scores.get(dimension_key, 0.0) for sample in samples],
            dtype=float,
        )
        human_values = np.array(
            [sample.human_dim_scores.get(dimension_key, 0.0) for sample in samples],
            dtype=float,
        )
        if raw_values.size < 2 or float(np.std(raw_values)) < 1e-6:
            fitted_gains[dimension_key] = {"gain": 1.0, "bias": 0.0}
            continue
        design_matrix = np.column_stack([raw_values, np.ones_like(raw_values)])
        least_squares_solution, _residual_vec, _rank, _singular_values = np.linalg.lstsq(
            design_matrix, human_values, rcond=None
        )
        gain_raw = float(least_squares_solution[0])
        bias_raw = float(least_squares_solution[1])
        gain_clamped = max(GAIN_MIN, min(GAIN_MAX, gain_raw))
        bias_clamped = max(BIAS_MIN, min(BIAS_MAX, bias_raw))
        fitted_gains[dimension_key] = {"gain": gain_clamped, "bias": bias_clamped}
    return fitted_gains


def _fit_weights_constrained(
    samples: Sequence[LabeledSample],
    gains: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """NNLS + simplex-projection fit of weights on the (optionally adjusted) dims."""
    dimension_count = len(DIMENSION_KEYS)
    if gains is None:
        design_matrix = np.array(
            [
                [sample.heuristic_dim_scores.get(key, 0.0) for key in DIMENSION_KEYS]
                for sample in samples
            ],
            dtype=float,
        )
    else:
        design_matrix = np.array(
            [
                [
                    _apply_dimension_gains(sample.heuristic_dim_scores, gains=gains).get(
                        key, 0.0
                    )
                    for key in DIMENSION_KEYS
                ]
                for sample in samples
            ],
            dtype=float,
        )
    target_overall = np.array([sample.human_overall for sample in samples], dtype=float)

    if design_matrix.size == 0:
        return dict(DEFAULT_DIMENSION_WEIGHTS)

    raw_weights, _residual = nnls(design_matrix, target_overall)
    if not np.any(raw_weights > 0):
        # NNLS returned all-zero (rare but possible); fall back to current weights
        # and let projection handle it.
        raw_weights = np.array([DIMENSION_WEIGHTS.get(key, 1.0 / dimension_count) for key in DIMENSION_KEYS])

    projected_weights = _project_onto_probability_simplex(raw_weights)
    return {dimension_key: float(projected_weights[idx]) for idx, dimension_key in enumerate(DIMENSION_KEYS)}


def _bias_vs_weight_diagnostic(
    samples: Sequence[LabeledSample],
    fitted_gains: Dict[str, Dict[str, float]],
    weights_only_mae: float,
) -> List[Dict[str, float]]:
    """For each dimension, compare 'fix bias on this dim + refit weights' vs
    'refit weights only, all gains identity'.

    If a dimension's bias-fix MAE is materially lower than the weights-only
    baseline, the scorer has a systematic offset that re-weighting alone
    cannot absorb — the right fix is in the scorer (token lists, proximity
    windows) rather than in `config/weights.json`. If the two MAEs are
    close, re-weighting already covers it.

    We don't touch `gain`; only `bias`. This isolates the additive-offset
    hypothesis from the linear-rescale hypothesis.
    """
    diagnostic_rows: List[Dict[str, float]] = []
    for dimension_key in DIMENSION_KEYS:
        bias_only_gains: Dict[str, Dict[str, float]] = {
            other_key: {"gain": 1.0, "bias": 0.0} for other_key in DIMENSION_KEYS
        }
        bias_only_gains[dimension_key] = {
            "gain": 1.0,
            "bias": float(fitted_gains.get(dimension_key, {}).get("bias", 0.0)),
        }
        refit_weights = _fit_weights_constrained(samples, gains=bias_only_gains)
        bias_fix_mae = _compute_overall_mae(samples, refit_weights, gains=bias_only_gains)
        diagnostic_rows.append(
            {
                "dimension": dimension_key,
                "bias_applied": float(bias_only_gains[dimension_key]["bias"]),
                "bias_fix_mae": float(bias_fix_mae),
                "weights_only_mae": float(weights_only_mae),
                "delta": float(weights_only_mae - bias_fix_mae),
            }
        )
    return diagnostic_rows


def _format_bias_vs_weight_table(
    diagnostic_rows: Sequence[Dict[str, float]],
) -> str:
    header = (
        "| Dimension | Bias applied (Δ) | Bias-fix MAE | Weights-only MAE | "
        "Weights-only - Bias-fix | Recommendation |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for diagnostic_row in diagnostic_rows:
        delta_value = float(diagnostic_row.get("delta", 0.0))
        if delta_value > 0.05:
            recommendation = "fix scorer (bias helps)"
        elif delta_value < -0.05:
            recommendation = "leave scorer (bias hurts)"
        else:
            recommendation = "re-weighting covers it"
        rows.append(
            f"| {DIMENSION_LABELS.get(str(diagnostic_row['dimension']), str(diagnostic_row['dimension']))} "
            f"| {float(diagnostic_row['bias_applied']):+.3f} "
            f"| {float(diagnostic_row['bias_fix_mae']):.3f} "
            f"| {float(diagnostic_row['weights_only_mae']):.3f} "
            f"| {delta_value:+.3f} "
            f"| {recommendation} |"
        )
    return header + "\n".join(rows)


def _per_labeler_summary(
    samples: Sequence[LabeledSample],
    weights: Dict[str, float],
    gains: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Per-labeler diagnostic: how does each labeler compare to the fitted model?

    Splits samples by labeler handle and computes N, overall MAE (using the
    supplied weights + gains — i.e. the fitted ones), per-dim mean absolute
    deviation from the heuristic, and the labeler's mean overall score. This
    surfaces a labeler who's systematically more generous/harsh than the rest.
    """
    labeler_groups: Dict[str, List[LabeledSample]] = {}
    for sample in samples:
        labeler_groups.setdefault(sample.labeler, []).append(sample)
    summary_rows: List[Dict[str, Any]] = []
    for labeler_handle, labeler_samples in sorted(labeler_groups.items()):
        labeler_overall_mae = _compute_overall_mae(labeler_samples, weights, gains=gains)
        dim_mean_abs_dev: Dict[str, float] = {}
        for dimension_key in DIMENSION_KEYS:
            differences = [
                sample.heuristic_dim_scores.get(dimension_key, 0.0)
                - sample.human_dim_scores.get(dimension_key, 0.0)
                for sample in labeler_samples
            ]
            dim_mean_abs_dev[dimension_key] = (
                float(np.mean(np.abs(differences))) if differences else 0.0
            )
        human_overall_values = [float(sample.human_overall) for sample in labeler_samples]
        summary_rows.append(
            {
                "labeler": labeler_handle,
                "n": len(labeler_samples),
                "overall_mae": labeler_overall_mae,
                "mean_human_overall": (
                    float(np.mean(human_overall_values)) if human_overall_values else 0.0
                ),
                "dim_mae": dim_mean_abs_dev,
            }
        )
    return summary_rows


def _format_per_labeler_table(summary_rows: Sequence[Dict[str, Any]]) -> str:
    header_cells = ["| Labeler | N | Overall MAE | Mean human overall"]
    separator_cells = ["|---|---|---|---"]
    for dimension_key in DIMENSION_KEYS:
        header_cells.append(f"| Δ {DIMENSION_LABELS.get(dimension_key, dimension_key)}")
        separator_cells.append("|---")
    header_cells.append(" |")
    separator_cells.append("|")
    lines: List[str] = ["".join(header_cells), "".join(separator_cells)]
    for row in summary_rows:
        cells = [
            f"| {row['labeler']} ",
            f"| {int(row['n'])} ",
            f"| {float(row['overall_mae']):.3f} ",
            f"| {float(row['mean_human_overall']):.2f} ",
        ]
        for dimension_key in DIMENSION_KEYS:
            cells.append(f"| {float(row['dim_mae'].get(dimension_key, 0.0)):.2f} ")
        cells.append("|")
        lines.append("".join(cells))
    return "\n".join(lines)


def _inter_labeler_disagreement_matrix(
    samples: Sequence[LabeledSample],
) -> Optional[List[List[Any]]]:
    """Return an NxN disagreement matrix (list-of-lists suitable for Markdown).

    Cell (A, B) is the mean |overall_A - overall_B| over profiles both
    labeled. Diagonal is always 0. Returns None when there is only one
    labeler — there's no pair to compare, so the matrix is vacuous.
    """
    labeler_to_profiles: Dict[str, Dict[str, float]] = {}
    for sample in samples:
        labeler_to_profiles.setdefault(sample.labeler, {})[sample.profile_id] = float(
            sample.human_overall
        )
    labeler_handles = sorted(labeler_to_profiles.keys())
    if len(labeler_handles) < 2:
        return None

    matrix: List[List[Any]] = []
    header_row: List[Any] = [""]
    header_row.extend(labeler_handles)
    matrix.append(header_row)
    for row_labeler in labeler_handles:
        row_cells: List[Any] = [row_labeler]
        for column_labeler in labeler_handles:
            if row_labeler == column_labeler:
                row_cells.append("0.00")
                continue
            shared_profile_ids = set(labeler_to_profiles[row_labeler].keys()) & set(
                labeler_to_profiles[column_labeler].keys()
            )
            if not shared_profile_ids:
                row_cells.append("n/a")
                continue
            differences = [
                abs(
                    labeler_to_profiles[row_labeler][profile_id_value]
                    - labeler_to_profiles[column_labeler][profile_id_value]
                )
                for profile_id_value in shared_profile_ids
            ]
            row_cells.append(f"{float(np.mean(differences)):.2f} (n={len(differences)})")
        matrix.append(row_cells)
    return matrix


def _format_matrix_as_markdown(matrix: Sequence[Sequence[Any]]) -> str:
    if not matrix:
        return ""
    header_cells = matrix[0]
    header_line = "| " + " | ".join(str(cell) for cell in header_cells) + " |"
    separator_line = "|" + "|".join(["---"] * len(header_cells)) + "|"
    body_lines: List[str] = []
    for row in matrix[1:]:
        body_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


def _compute_overall_mae(
    samples: Sequence[LabeledSample],
    weights: Dict[str, float],
    gains: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """MAE of predicted vs human overall. If `gains` is supplied the raw
    dimension scores are passed through the affine transform first, matching
    what the live pipeline will do with the same (weights, gains) pair."""
    if not samples:
        return 0.0
    total_absolute_error = 0.0
    for sample in samples:
        if gains is None:
            dim_scores_for_sum = sample.heuristic_dim_scores
        else:
            dim_scores_for_sum = _apply_dimension_gains(
                sample.heuristic_dim_scores, gains=gains
            )
        predicted_overall = sum(
            float(weights.get(key, 0.0)) * float(dim_scores_for_sum.get(key, 0.0))
            for key in DIMENSION_KEYS
        )
        predicted_overall = max(0.0, min(10.0, predicted_overall))
        total_absolute_error += abs(predicted_overall - sample.human_overall)
    return total_absolute_error / len(samples)


def _format_gains_block(label: str, gains: Dict[str, Dict[str, float]]) -> str:
    lines = [f"**{label}**"]
    for dimension_key in DIMENSION_KEYS:
        dimension_gain_entry = gains.get(dimension_key, {"gain": 1.0, "bias": 0.0})
        lines.append(
            f"- {DIMENSION_LABELS.get(dimension_key, dimension_key)}: "
            f"gain={float(dimension_gain_entry['gain']):.3f}, "
            f"bias={float(dimension_gain_entry['bias']):+.3f}"
        )
    return "\n".join(lines)


def _format_dimension_metrics_table(metrics: Dict[str, Dict[str, float]]) -> str:
    header = (
        "| Dimension | N | MAE | Bias (heur - human) | Spearman | Heur std | Human std | Flag |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for dimension_key in DIMENSION_KEYS:
        dimension_metrics = metrics.get(dimension_key, {})
        n_value = int(dimension_metrics.get("n", 0))
        mae_value = float(dimension_metrics.get("mae", 0.0))
        bias_value = float(dimension_metrics.get("bias", 0.0))
        spearman_value = float(dimension_metrics.get("spearman", float("nan")))
        heuristic_std_value = float(dimension_metrics.get("heuristic_std", 0.0))
        human_std_value = float(dimension_metrics.get("human_std", 0.0))

        flag_parts: List[str] = []
        if mae_value > MAE_FLAG_THRESHOLD:
            flag_parts.append(f"MAE>{MAE_FLAG_THRESHOLD}")
        if abs(bias_value) > BIAS_FLAG_THRESHOLD:
            direction = "over" if bias_value > 0 else "under"
            flag_parts.append(f"{direction}scores")
        if heuristic_std_value < LOW_STD_WARNING_THRESHOLD:
            flag_parts.append("heur flat")
        if human_std_value < LOW_STD_WARNING_THRESHOLD:
            flag_parts.append("human flat")

        flag_cell = ", ".join(flag_parts) if flag_parts else "-"
        spearman_cell = f"{spearman_value:+.2f}" if not math.isnan(spearman_value) else "n/a"
        rows.append(
            f"| {DIMENSION_LABELS.get(dimension_key, dimension_key)} "
            f"| {n_value} | {mae_value:.2f} | {bias_value:+.2f} | {spearman_cell} "
            f"| {heuristic_std_value:.2f} | {human_std_value:.2f} | {flag_cell} |"
        )
    return header + "\n".join(rows)


def _format_weights_block(label: str, weights: Dict[str, float]) -> str:
    lines = [f"**{label}**"]
    for dimension_key in DIMENSION_KEYS:
        weight_value = float(weights.get(dimension_key, 0.0))
        lines.append(f"- {DIMENSION_LABELS.get(dimension_key, dimension_key)}: {weight_value:.4f}")
    return "\n".join(lines)


DRIFT_SIGNIFICANT_WEIGHT_DELTA = 0.02
DRIFT_SIGNIFICANT_GAIN_DELTA = 0.05
DRIFT_SIGNIFICANT_BIAS_DELTA = 0.10


def _format_drift_section(
    previous_archive_path: Path,
    previous_weights: Dict[str, float],
    previous_gains: Dict[str, Dict[str, float]],
    new_weights: Dict[str, float],
    new_gains: Dict[str, Dict[str, float]],
) -> List[str]:
    """Format a "Drift since last fit" section comparing newly fit weights+gains
    against the most-recent archived snapshot.

    The thresholds (`DRIFT_SIGNIFICANT_*`) tag a row as a "shift" when the change
    is large enough that it would actually alter rankings. Sub-threshold deltas
    are still printed (we never hide numbers), they just don't get the marker.
    """
    section_lines: List[str] = []
    section_lines.append("## Drift since last fit")
    section_lines.append("")
    section_lines.append(
        f"Comparing the just-written weights against the previous archived "
        f"fit at `{previous_archive_path.relative_to(PROJECT_ROOT)}`. "
        f"A row is flagged when the change is large enough to plausibly move "
        f"the top-K ranking."
    )
    section_lines.append("")
    section_lines.append(
        "| Dimension | Weight prev → new (Δ) | Gain prev → new (Δ) | "
        "Bias prev → new (Δ) | Shift? |"
    )
    section_lines.append("|---|---|---|---|---|")

    for dimension_key in DIMENSION_KEYS:
        prev_weight_value = float(previous_weights.get(dimension_key, 0.0))
        new_weight_value = float(new_weights.get(dimension_key, 0.0))
        weight_delta = new_weight_value - prev_weight_value

        prev_gain_entry = previous_gains.get(dimension_key, {"gain": 1.0, "bias": 0.0})
        new_gain_entry = new_gains.get(dimension_key, {"gain": 1.0, "bias": 0.0})
        prev_gain_value = float(prev_gain_entry.get("gain", 1.0))
        new_gain_value = float(new_gain_entry.get("gain", 1.0))
        gain_delta = new_gain_value - prev_gain_value

        prev_bias_value = float(prev_gain_entry.get("bias", 0.0))
        new_bias_value = float(new_gain_entry.get("bias", 0.0))
        bias_delta = new_bias_value - prev_bias_value

        shift_markers: List[str] = []
        if abs(weight_delta) >= DRIFT_SIGNIFICANT_WEIGHT_DELTA:
            shift_markers.append("weight")
        if abs(gain_delta) >= DRIFT_SIGNIFICANT_GAIN_DELTA:
            shift_markers.append("gain")
        if abs(bias_delta) >= DRIFT_SIGNIFICANT_BIAS_DELTA:
            shift_markers.append("bias")
        shift_cell = ", ".join(shift_markers) if shift_markers else "-"

        section_lines.append(
            f"| {DIMENSION_LABELS.get(dimension_key, dimension_key)} "
            f"| {prev_weight_value:.4f} → {new_weight_value:.4f} ({weight_delta:+.4f}) "
            f"| {prev_gain_value:.3f} → {new_gain_value:.3f} ({gain_delta:+.3f}) "
            f"| {prev_bias_value:+.3f} → {new_bias_value:+.3f} ({bias_delta:+.3f}) "
            f"| {shift_cell} |"
        )
    section_lines.append("")
    return section_lines


def _write_report(
    report_lines: List[str],
    report_timestamp: datetime,
) -> Path:
    report_directory = PROJECT_ROOT / REPORT_DIR_NAME
    report_directory.mkdir(parents=True, exist_ok=True)
    filename = f"calibration_{report_timestamp.strftime('%Y-%m-%d_%H%M')}.md"
    report_path = report_directory / filename
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", default=None, help="Restrict to a single labeler handle.")
    parser.add_argument(
        "--min-labels",
        type=int,
        default=DEFAULT_MIN_LABELS,
        help=f"Minimum labels required before writing new weights (default {DEFAULT_MIN_LABELS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report but do not write config/weights.json.",
    )
    parser.add_argument(
        "--fit-gains",
        dest="fit_gains",
        action="store_true",
        default=True,
        help=(
            "Fit per-dimension affine gains (gain, bias) before fitting weights. "
            "This is on by default; use --no-fit-gains to disable."
        ),
    )
    parser.add_argument(
        "--no-fit-gains",
        dest="fit_gains",
        action="store_false",
        help="Disable per-dimension gain fit; only fit global weights.",
    )
    parser.add_argument(
        "--per-labeler",
        action="store_true",
        help=(
            "Add per-labeler MAE + inter-labeler disagreement matrix to the "
            "report. Useful when multiple humans have rated overlapping "
            "candidates and you want to see whether they mean the same thing "
            "by each dimension."
        ),
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_cli_args()
    report_timestamp = datetime.now()

    samples, collection_warnings = _collect_samples(labeler=cli_args.labeler)
    n_samples = len(samples)

    report_lines: List[str] = []
    report_lines.append(f"# Calibration report — {report_timestamp.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")
    report_lines.append(f"- Labeler filter: `{cli_args.labeler or 'all'}`")
    report_lines.append(f"- Labels used (latest per profile+labeler): **{n_samples}**")
    report_lines.append(f"- Min labels required: {cli_args.min_labels}")
    report_lines.append(f"- Dry run: {cli_args.dry_run}")
    report_lines.append("")

    if collection_warnings:
        report_lines.append("## Collection warnings")
        report_lines.extend([f"- {warning_message}" for warning_message in collection_warnings])
        report_lines.append("")

    if n_samples == 0:
        report_lines.append("No usable labels available; nothing to calibrate.")
        report_path = _write_report(report_lines, report_timestamp)
        print(f"No labels found. Report: {report_path}")
        return 0

    metrics = _per_dimension_metrics(samples)
    report_lines.append("## Per-dimension calibration")
    report_lines.append("")
    report_lines.append(_format_dimension_metrics_table(metrics))
    report_lines.append("")
    report_lines.append(
        f"Flags trigger at |bias| > {BIAS_FLAG_THRESHOLD} or MAE > {MAE_FLAG_THRESHOLD}; "
        f"std < {LOW_STD_WARNING_THRESHOLD} suggests insufficient variation."
    )
    report_lines.append("")

    mae_before_fit = _compute_overall_mae(samples, DIMENSION_WEIGHTS, gains=DIMENSION_GAINS)
    report_lines.append("## Overall score alignment")
    report_lines.append("")
    report_lines.append(f"- Overall MAE with current weights+gains: **{mae_before_fit:.3f}**")

    if n_samples < cli_args.min_labels:
        report_lines.append("")
        report_lines.append(
            f"Skipping weight fit: have {n_samples} labels, need >= {cli_args.min_labels}. "
            f"Use `--min-labels` to override."
        )
        report_lines.append("")
        report_lines.append(_format_weights_block("Current weights (unchanged)", DIMENSION_WEIGHTS))
        report_lines.append("")
        report_lines.append(_format_gains_block("Current gains (unchanged)", DIMENSION_GAINS))
        report_path = _write_report(report_lines, report_timestamp)
        print(
            f"Need >={cli_args.min_labels} labels to fit weights (have {n_samples}). "
            f"Report: {report_path}"
        )
        return 0

    if cli_args.fit_gains:
        fitted_gains = _fit_dimension_gains(samples)
        weights_only_fitted = _fit_weights_constrained(samples, gains=None)
        mae_weights_only = _compute_overall_mae(
            samples, weights_only_fitted, gains=None
        )
        fitted_weights = _fit_weights_constrained(samples, gains=fitted_gains)
        mae_after_fit = _compute_overall_mae(samples, fitted_weights, gains=fitted_gains)
    else:
        fitted_gains = default_dimension_gains(DEFAULT_DIMENSION_WEIGHTS)
        mae_weights_only = None
        weights_only_fitted = _fit_weights_constrained(samples, gains=None)
        fitted_weights = weights_only_fitted
        mae_after_fit = _compute_overall_mae(samples, fitted_weights, gains=None)

    report_lines.append(f"- Overall MAE with fitted weights+gains: **{mae_after_fit:.3f}**")
    if mae_weights_only is not None:
        report_lines.append(
            f"- Ablation (fit weights only, gains fixed at identity): **{mae_weights_only:.3f}**"
        )
    improvement_delta = mae_before_fit - mae_after_fit
    report_lines.append(f"- Improvement: **{improvement_delta:+.3f}** (positive = better)")
    report_lines.append("")
    report_lines.append(_format_weights_block("Current weights (before fit)", DIMENSION_WEIGHTS))
    report_lines.append("")
    report_lines.append(_format_weights_block("Fitted weights (after fit)", fitted_weights))
    report_lines.append("")
    report_lines.append(_format_gains_block("Current gains (before fit)", DIMENSION_GAINS))
    report_lines.append("")
    report_lines.append(_format_gains_block("Fitted gains (after fit)", fitted_gains))
    report_lines.append("")

    if mae_weights_only is not None:
        report_lines.append("## Bias vs weight diagnostic")
        report_lines.append("")
        report_lines.append(
            "For each dimension we ask: if I *only* shift this dimension's "
            "baseline (gain=1, bias=fitted_bias) and then refit weights on "
            "top of that, how does the MAE compare to the \"just refit "
            "weights, gains=identity\" baseline? A positive Δ means the "
            "systematic offset is real and the scorer itself should be "
            "tightened. Near-zero Δ means re-weighting already absorbs it."
        )
        report_lines.append("")
        diagnostic_rows = _bias_vs_weight_diagnostic(
            samples, fitted_gains, mae_weights_only
        )
        report_lines.append(_format_bias_vs_weight_table(diagnostic_rows))
        report_lines.append("")

    if cli_args.per_labeler:
        report_lines.append("## Per-labeler diagnostics")
        report_lines.append("")
        per_labeler_rows = _per_labeler_summary(
            samples, weights=fitted_weights, gains=fitted_gains
        )
        report_lines.append(
            "Overall MAE is computed against the *fitted* weights and gains. "
            "Per-dim columns show mean |heuristic - human| for that labeler. "
            "A row with much higher overall MAE than the others usually means "
            "that labeler interprets the rubric differently, not that the "
            "model got worse for them."
        )
        report_lines.append("")
        report_lines.append(_format_per_labeler_table(per_labeler_rows))
        report_lines.append("")
        disagreement_matrix = _inter_labeler_disagreement_matrix(samples)
        if disagreement_matrix is not None:
            report_lines.append("### Inter-labeler overall disagreement")
            report_lines.append("")
            report_lines.append(
                "Mean |overall_A - overall_B| on profiles both labelers rated. "
                "`n/a` means the pair has no shared profiles."
            )
            report_lines.append("")
            report_lines.append(_format_matrix_as_markdown(disagreement_matrix))
            report_lines.append("")
        else:
            report_lines.append(
                "Only one labeler present — skipping inter-labeler disagreement matrix."
            )
            report_lines.append("")

    if cli_args.dry_run:
        report_lines.append("Dry run: `config/weights.json` NOT written.")
        report_path = _write_report(report_lines, report_timestamp)
        print(
            f"Dry run. MAE {mae_before_fit:.3f} -> {mae_after_fit:.3f} (delta {improvement_delta:+.3f}). "
            f"Report: {report_path}"
        )
        return 0

    previous_archive_path = find_most_recent_archive()
    previous_archive_weights: Optional[Dict[str, float]] = None
    previous_archive_gains: Optional[Dict[str, Dict[str, float]]] = None
    if previous_archive_path is not None:
        previous_archive_weights = load_weights(
            DEFAULT_DIMENSION_WEIGHTS, path=previous_archive_path
        )
        previous_archive_gains = load_gains(
            DEFAULT_DIMENSION_WEIGHTS, path=previous_archive_path
        )

    new_version = next_version()
    save_weights(
        weights=fitted_weights,
        defaults=DEFAULT_DIMENSION_WEIGHTS,
        version=new_version,
        n_labels=n_samples,
        mae_before=round(mae_before_fit, 4),
        mae_after=round(mae_after_fit, 4),
        labeler=cli_args.labeler,
        gains=fitted_gains,
    )
    archive_path = archive_weights_snapshot(timestamp=report_timestamp)
    report_lines.append(f"Wrote `{weights_file_path()}` (version `{new_version}`).")
    if archive_path is not None:
        report_lines.append(
            f"Archived snapshot to `{archive_path.relative_to(PROJECT_ROOT)}`."
        )
    report_lines.append("")

    if (
        previous_archive_path is not None
        and previous_archive_weights is not None
        and previous_archive_gains is not None
    ):
        report_lines.extend(
            _format_drift_section(
                previous_archive_path=previous_archive_path,
                previous_weights=previous_archive_weights,
                previous_gains=previous_archive_gains,
                new_weights=fitted_weights,
                new_gains=fitted_gains,
            )
        )
    else:
        report_lines.append(
            "No prior archived weights snapshot found; skipping drift section. "
            "Future fits will include a Drift since last fit table."
        )
        report_lines.append("")

    report_path = _write_report(report_lines, report_timestamp)

    print(
        f"Fit {n_samples} labels. "
        f"MAE {mae_before_fit:.3f} -> {mae_after_fit:.3f} "
        f"(delta {improvement_delta:+.3f}). "
        f"Wrote weights version {new_version}. "
        f"Report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
