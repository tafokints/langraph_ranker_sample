"""Generate demo seed labels for the rubric calibration loop.

Runs the three smoke-test prompts, takes the top-N candidates from each,
applies a deterministic "perceptual adjustment" on top of the heuristic
dimension scores (no LLM, no human) to simulate plausible recruiter
disagreements, and emits both:

1. A SQL fixture at `fixtures/seed_labels.sql` with INSERTs for the
   `recruiter_rubric_labels` table so a fresh clone can reproduce the
   calibration run deterministically.
2. A JSON inventory at `fixtures/seed_labels.json` listing every
   labeled (profile_id, dim_scores, overall_score) tuple. Useful for
   eyeballing before loading.

The perceptual-adjustment biases encode the most common disagreements a
human reviewer raises in practice:

- `phd_researcher`: humans discount postdoc/candidate mentions more than
  the heuristic does; scores drift down when the heuristic was > 5.
- `education_prestige`: humans weight graduation signals harder; scores
  drift down by ~1 point.
- `technical_background`: humans reward depth over breadth; scores drift
  up by ~0.6 when the heuristic is between 4 and 8.
- `founder_experience`: humans notice even mild founder/startup context;
  scores drift up by ~0.8 when heuristic > 2.
- `sf_location_fit`: mostly agreed with the heuristic, small jitter.

This script is deterministic (seeded RNG) so the SQL fixture is
reproducible across runs. It is NOT a replacement for real labels; it
exists so reviewers can demo the calibration loop end-to-end without
needing humans in the loop.

Usage:
    python scripts/generate_seed_labels.py               # write SQL + JSON fixture, no DB writes
    python scripts/generate_seed_labels.py --write-db    # also insert labels into MySQL
    python scripts/generate_seed_labels.py --labeler me  # label under a specific handle
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.langgraph_app import (
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSION_KEYS,
    run_recruiter_search,
)
from src.labels_store import LABELS_TABLE_NAME, save_label

FIXTURES_DIR = PROJECT_ROOT / "fixtures"
SQL_OUTPUT_PATH = FIXTURES_DIR / "seed_labels.sql"
JSON_OUTPUT_PATH = FIXTURES_DIR / "seed_labels.json"
DEFAULT_LABELER = "seed-demo"
DEFAULT_TOP_N_PER_PROMPT = 7
RANDOM_SEED = 20260425  # change only to regenerate the fixture

SEED_PROMPTS = [
    {
        "label": "role-focused",
        "question": "Senior technical recruiter focused on hiring ML engineers in the US",
        "top_k": 8,
        "min_experience": 3,
    },
    {
        "label": "skill-focused",
        "question": "Unity game developer with C# experience and shipped mobile titles",
        "top_k": 8,
        "min_experience": 2,
    },
    {
        "label": "location-focused",
        "question": "Product designer based in the Bay Area with fintech background",
        "top_k": 8,
        "min_experience": 2,
    },
]


def _clip_score(raw_score: float) -> float:
    return max(0.0, min(10.0, round(float(raw_score), 2)))


def _perceptual_adjust(
    heuristic_dim_scores: Dict[str, float],
    rng: random.Random,
) -> Dict[str, float]:
    """Apply systematic human-biases + jitter to a set of heuristic dim scores.

    Biases match the five disagreements called out in the module docstring.
    Jitter is Gaussian with small sigma so samples stay plausible. All
    outputs are clipped to [0, 10].
    """
    adjusted: Dict[str, float] = {}

    raw_phd = float(heuristic_dim_scores.get("phd_researcher", 0.0))
    phd_shift = -1.5 if raw_phd > 5.0 else -0.3
    adjusted["phd_researcher"] = _clip_score(raw_phd + phd_shift + rng.gauss(0.0, 0.4))

    raw_sf = float(heuristic_dim_scores.get("sf_location_fit", 0.0))
    adjusted["sf_location_fit"] = _clip_score(raw_sf + rng.gauss(0.0, 0.35))

    raw_technical = float(heuristic_dim_scores.get("technical_background", 0.0))
    technical_shift = 0.6 if 4.0 <= raw_technical <= 8.0 else 0.1
    adjusted["technical_background"] = _clip_score(
        raw_technical + technical_shift + rng.gauss(0.0, 0.5)
    )

    raw_edu = float(heuristic_dim_scores.get("education_prestige", 0.0))
    adjusted["education_prestige"] = _clip_score(raw_edu - 1.0 + rng.gauss(0.0, 0.45))

    raw_founder = float(heuristic_dim_scores.get("founder_experience", 0.0))
    founder_shift = 0.8 if raw_founder > 2.0 else 0.2
    adjusted["founder_experience"] = _clip_score(
        raw_founder + founder_shift + rng.gauss(0.0, 0.4)
    )

    return adjusted


def _perceptual_overall(
    adjusted_dim_scores: Dict[str, float],
    rng: random.Random,
) -> float:
    """Compute a plausible human overall score.

    Humans don't perfectly average the dimensions; they place slightly more
    weight on the strongest dimension and add a small halo/penalty. This
    gives the calibration loop something non-trivial to fit.
    """
    weighted_sum = sum(
        float(DEFAULT_DIMENSION_WEIGHTS[key]) * float(adjusted_dim_scores[key])
        for key in DIMENSION_KEYS
    )
    top_dimension_score = max(adjusted_dim_scores.values())
    halo_contribution = 0.15 * top_dimension_score
    gaussian_jitter = rng.gauss(0.0, 0.35)
    return _clip_score(weighted_sum + halo_contribution + gaussian_jitter - 0.6)


def _collect_prompt_candidates(top_n_per_prompt: int) -> List[Dict[str, Any]]:
    """Run each seed prompt through the recruiter graph and return
    candidate records with the heuristic dim scores we'll adjust from."""
    collected: List[Dict[str, Any]] = []
    for prompt_spec in SEED_PROMPTS:
        print(
            f"[seed-labels] running prompt '{prompt_spec['label']}' "
            f"(top_k={prompt_spec['top_k']}) ..."
        )
        run_result = run_recruiter_search(
            question_text=prompt_spec["question"],
            top_k=prompt_spec["top_k"],
            min_experience_entries=prompt_spec["min_experience"],
        )
        ranked_candidates_all = run_result.get("ranked_candidates") or []
        for candidate_record in ranked_candidates_all[:top_n_per_prompt]:
            raw_dim_scores = dict(candidate_record.get("dimension_scores") or {})
            collected.append(
                {
                    "profile_id": candidate_record.get("profile_id") or "",
                    "prompt_label": prompt_spec["label"],
                    "heuristic_dim_scores": {
                        key: float(raw_dim_scores.get(key, 0.0)) for key in DIMENSION_KEYS
                    },
                    "heuristic_overall": float(candidate_record.get("rank_score") or 0.0),
                }
            )
    return collected


def _build_label_rows(
    prompt_candidates: List[Dict[str, Any]],
    labeler: str,
) -> List[Dict[str, Any]]:
    """Deterministic seed RNG so the fixture bytes are reproducible."""
    rng = random.Random(RANDOM_SEED)
    label_rows: List[Dict[str, Any]] = []
    base_time = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    for offset_index, candidate_record in enumerate(prompt_candidates):
        heuristic_dim_scores = candidate_record["heuristic_dim_scores"]
        adjusted_dim_scores = _perceptual_adjust(heuristic_dim_scores, rng=rng)
        adjusted_overall_score = _perceptual_overall(adjusted_dim_scores, rng=rng)
        label_rows.append(
            {
                "profile_id": candidate_record["profile_id"],
                "labeler": labeler,
                "prompt_label": candidate_record["prompt_label"],
                "heuristic_dim_scores": heuristic_dim_scores,
                "heuristic_overall": candidate_record["heuristic_overall"],
                "dim_scores": adjusted_dim_scores,
                "overall_score": adjusted_overall_score,
                "note": (
                    f"[seed-demo] adjusted-from-heuristic; prompt="
                    f"{candidate_record['prompt_label']}"
                ),
                "created_at": (base_time + timedelta(seconds=offset_index)).replace(
                    tzinfo=None
                ),
            }
        )
    return label_rows


def _sql_escape_string(raw_value: str) -> str:
    return (raw_value or "").replace("\\", "\\\\").replace("'", "''")


def _format_insert_sql(label_rows: List[Dict[str, Any]]) -> str:
    """Emit an idempotent SQL fixture.

    Strategy: DELETE any prior labels from this labeler (across fixture
    profiles) before INSERT so loading the fixture repeatedly doesn't pile
    up duplicate rows under different timestamps. This keeps the
    `latest_per_profile_labeler=True` calibrator behavior honest.
    """
    fixture_lines: List[str] = []
    fixture_lines.append(
        "-- Seed labels for the rubric calibration loop. Generated by\n"
        "-- scripts/generate_seed_labels.py. Idempotent: deletes any prior\n"
        "-- rows for the same (labeler, profile_id) pairs before INSERT so\n"
        "-- rerunning the loader does not duplicate.\n"
    )
    if not label_rows:
        return "".join(fixture_lines) + "-- (no rows generated)\n"

    fixture_labeler = label_rows[0]["labeler"]
    fixture_profile_ids = ", ".join(
        f"'{_sql_escape_string(row['profile_id'])}'" for row in label_rows
    )
    fixture_lines.append(
        f"DELETE FROM {LABELS_TABLE_NAME} "
        f"WHERE labeler = '{_sql_escape_string(fixture_labeler)}' "
        f"AND profile_id IN ({fixture_profile_ids});\n\n"
    )

    fixture_lines.append(
        f"INSERT INTO {LABELS_TABLE_NAME} "
        "(profile_id, labeler, dim_scores, overall_score, note, created_at) VALUES\n"
    )
    value_clauses: List[str] = []
    for label_row in label_rows:
        dim_json = json.dumps(label_row["dim_scores"], sort_keys=True)
        created_at_text = label_row["created_at"].strftime("%Y-%m-%d %H:%M:%S.000")
        value_clauses.append(
            f"  ('{_sql_escape_string(label_row['profile_id'])}', "
            f"'{_sql_escape_string(label_row['labeler'])}', "
            f"'{_sql_escape_string(dim_json)}', "
            f"{float(label_row['overall_score']):.2f}, "
            f"'{_sql_escape_string(label_row['note'])}', "
            f"'{created_at_text}')"
        )
    fixture_lines.append(",\n".join(value_clauses))
    fixture_lines.append(";\n")
    return "".join(fixture_lines)


def _write_sql_fixture(label_rows: List[Dict[str, Any]]) -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    SQL_OUTPUT_PATH.write_text(_format_insert_sql(label_rows), encoding="utf-8")
    return SQL_OUTPUT_PATH


def _write_json_inventory(label_rows: List[Dict[str, Any]]) -> Path:
    serializable_rows = [
        {
            "profile_id": row["profile_id"],
            "labeler": row["labeler"],
            "prompt_label": row["prompt_label"],
            "heuristic_dim_scores": row["heuristic_dim_scores"],
            "heuristic_overall": row["heuristic_overall"],
            "dim_scores": row["dim_scores"],
            "overall_score": row["overall_score"],
            "note": row["note"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in label_rows
    ]
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(serializable_rows, indent=2, sort_keys=False), encoding="utf-8"
    )
    return JSON_OUTPUT_PATH


def _insert_rows_into_db(label_rows: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Write rows into MySQL via the normal `save_label` path. Returns
    (rows_inserted, warnings)."""
    rows_inserted = 0
    warnings: List[str] = []
    for row_record in label_rows:
        try:
            save_label(
                profile_id=row_record["profile_id"],
                labeler=row_record["labeler"],
                dim_scores=row_record["dim_scores"],
                overall_score=row_record["overall_score"],
                note=row_record["note"],
                required_dimension_keys=list(DIMENSION_KEYS),
            )
            rows_inserted += 1
        except Exception as save_error:  # noqa: BLE001 - surface per-row error
            warnings.append(
                f"failed to insert {row_record['profile_id']} "
                f"for labeler={row_record['labeler']}: {save_error}"
            )
    return rows_inserted, warnings


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeler",
        default=DEFAULT_LABELER,
        help=f"Labeler handle to write labels under (default: {DEFAULT_LABELER}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N_PER_PROMPT,
        help=(
            "Labels per prompt (default {}). Total fixture size is "
            "roughly 3 * top-n.".format(DEFAULT_TOP_N_PER_PROMPT)
        ),
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Also insert the generated labels into MySQL via save_label().",
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_cli_args()

    prompt_candidates = _collect_prompt_candidates(top_n_per_prompt=cli_args.top_n)
    if not prompt_candidates:
        print("[seed-labels] no candidates collected; nothing to write.")
        return 1

    label_rows = _build_label_rows(prompt_candidates, labeler=cli_args.labeler)
    sql_path = _write_sql_fixture(label_rows)
    json_path = _write_json_inventory(label_rows)
    print(
        f"[seed-labels] wrote {len(label_rows)} rows: "
        f"{sql_path.relative_to(PROJECT_ROOT)} and {json_path.relative_to(PROJECT_ROOT)}"
    )

    if cli_args.write_db:
        inserted_count, warning_messages = _insert_rows_into_db(label_rows)
        print(f"[seed-labels] inserted {inserted_count}/{len(label_rows)} rows into MySQL.")
        for warning_message in warning_messages:
            print(f"[seed-labels]   WARN: {warning_message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
