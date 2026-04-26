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
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    DIMENSION_WEIGHTS,
    _aggregate_rank_score,
    _deterministic_rank,
)
from src.retriever import _open_connection as open_profile_connection
from src.weights_loader import next_version, save_weights, weights_file_path

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


def _fit_weights_constrained(samples: Sequence[LabeledSample]) -> Dict[str, float]:
    dimension_count = len(DIMENSION_KEYS)
    design_matrix = np.array(
        [[sample.heuristic_dim_scores.get(key, 0.0) for key in DIMENSION_KEYS] for sample in samples],
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


def _compute_overall_mae(samples: Sequence[LabeledSample], weights: Dict[str, float]) -> float:
    if not samples:
        return 0.0
    total_absolute_error = 0.0
    for sample in samples:
        predicted_overall = sum(
            float(weights.get(key, 0.0)) * float(sample.heuristic_dim_scores.get(key, 0.0))
            for key in DIMENSION_KEYS
        )
        predicted_overall = max(0.0, min(10.0, predicted_overall))
        total_absolute_error += abs(predicted_overall - sample.human_overall)
    return total_absolute_error / len(samples)


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

    mae_before_fit = _compute_overall_mae(samples, DIMENSION_WEIGHTS)
    report_lines.append("## Overall score alignment")
    report_lines.append("")
    report_lines.append(f"- Overall MAE with current weights: **{mae_before_fit:.3f}**")

    if n_samples < cli_args.min_labels:
        report_lines.append("")
        report_lines.append(
            f"Skipping weight fit: have {n_samples} labels, need >= {cli_args.min_labels}. "
            f"Use `--min-labels` to override."
        )
        report_lines.append("")
        report_lines.append(_format_weights_block("Current weights (unchanged)", DIMENSION_WEIGHTS))
        report_path = _write_report(report_lines, report_timestamp)
        print(
            f"Need >={cli_args.min_labels} labels to fit weights (have {n_samples}). "
            f"Report: {report_path}"
        )
        return 0

    fitted_weights = _fit_weights_constrained(samples)
    mae_after_fit = _compute_overall_mae(samples, fitted_weights)

    report_lines.append(f"- Overall MAE with fitted weights: **{mae_after_fit:.3f}**")
    improvement_delta = mae_before_fit - mae_after_fit
    report_lines.append(f"- Improvement: **{improvement_delta:+.3f}** (positive = better)")
    report_lines.append("")
    report_lines.append(_format_weights_block("Current weights (before fit)", DIMENSION_WEIGHTS))
    report_lines.append("")
    report_lines.append(_format_weights_block("Fitted weights (after fit)", fitted_weights))
    report_lines.append("")

    if cli_args.dry_run:
        report_lines.append("Dry run: `config/weights.json` NOT written.")
        report_path = _write_report(report_lines, report_timestamp)
        print(
            f"Dry run. MAE {mae_before_fit:.3f} -> {mae_after_fit:.3f} (delta {improvement_delta:+.3f}). "
            f"Report: {report_path}"
        )
        return 0

    new_version = next_version()
    save_weights(
        weights=fitted_weights,
        defaults=DEFAULT_DIMENSION_WEIGHTS,
        version=new_version,
        n_labels=n_samples,
        mae_before=round(mae_before_fit, 4),
        mae_after=round(mae_after_fit, 4),
        labeler=cli_args.labeler,
    )
    report_lines.append(f"Wrote `{weights_file_path()}` (version `{new_version}`).")
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
