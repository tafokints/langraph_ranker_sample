"""A/B compare two weights configurations on a fixed prompt list.

For each prompt: runs the recruiter graph twice (once per weights file via
`run_recruiter_search(weights_override=...)`), takes the top-K profile_ids
from each, and prints a markdown table showing the side-by-side ordering
plus their Jaccard similarity.

Usage:

    python scripts/ab_compare_weights.py \
        --weights-a config/weights.json \
        --weights-b config/weights.history/2026-04-19.json \
        --prompts fixtures/smoke_prompts.yml \
        --top-k 3

Notes:
  - Each (prompt, config) pair is a real LangGraph run, so this hits the
    LLM and the DB. Budget accordingly: 10 prompts x 2 configs ~= 20 runs.
  - Weights file must conform to the schema written by
    `scripts/calibrate.py` (a `weights` dict + optional `gains` dict;
    see `src/weights_loader.load_weights`).
  - If `--prompts` is omitted, falls back to the same three prompts the
    smoke test uses, so the script is runnable without any extra config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.langgraph_app import DEFAULT_DIMENSION_WEIGHTS, run_recruiter_search
from src.weights_loader import load_gains, load_weights

DEFAULT_TOP_K = 3
DEFAULT_PROMPTS_TOP_K = 6
DEFAULT_PROMPTS_MIN_EXPERIENCE = 2

# Fallback prompts used when --prompts is omitted. Mirrors `scripts/smoke_test.py`
# so a brand-new clone of the repo can run an A/B compare immediately.
FALLBACK_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "role-focused",
        "query": "Senior technical recruiter focused on hiring ML engineers in the US",
        "top_k": DEFAULT_PROMPTS_TOP_K,
        "min_experience": 3,
    },
    {
        "name": "skill-focused",
        "query": "Unity game developer with C# experience and shipped mobile titles",
        "top_k": DEFAULT_PROMPTS_TOP_K,
        "min_experience": DEFAULT_PROMPTS_MIN_EXPERIENCE,
    },
    {
        "name": "location-focused",
        "query": "Product designer based in the Bay Area with fintech background",
        "top_k": DEFAULT_PROMPTS_TOP_K,
        "min_experience": DEFAULT_PROMPTS_MIN_EXPERIENCE,
    },
]


def _load_prompts(prompts_path: Optional[Path]) -> List[Dict[str, Any]]:
    if prompts_path is None:
        return list(FALLBACK_PROMPTS)
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as import_error:
        raise SystemExit(
            "PyYAML is required to read --prompts. "
            "Install it with: pip install pyyaml"
        ) from import_error
    with prompts_path.open("r", encoding="utf-8") as prompts_file_handle:
        raw_payload = yaml.safe_load(prompts_file_handle)
    if not isinstance(raw_payload, list):
        raise ValueError(
            f"{prompts_path} must contain a YAML list of prompts; got {type(raw_payload).__name__}"
        )
    cleaned_prompts: List[Dict[str, Any]] = []
    for prompt_index, raw_prompt in enumerate(raw_payload):
        if not isinstance(raw_prompt, dict):
            raise ValueError(f"prompts[{prompt_index}] must be a mapping")
        if "query" not in raw_prompt:
            raise ValueError(f"prompts[{prompt_index}] missing required 'query' field")
        cleaned_prompts.append(
            {
                "name": str(raw_prompt.get("name") or f"prompt_{prompt_index}"),
                "query": str(raw_prompt["query"]),
                "top_k": int(raw_prompt.get("top_k", DEFAULT_PROMPTS_TOP_K)),
                "min_experience": int(
                    raw_prompt.get("min_experience", DEFAULT_PROMPTS_MIN_EXPERIENCE)
                ),
            }
        )
    return cleaned_prompts


def _resolve_weights_path(raw_path: str) -> Path:
    candidate_path = Path(raw_path)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path
    return candidate_path


def _load_config(weights_path: Path) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    config_weights = load_weights(DEFAULT_DIMENSION_WEIGHTS, path=weights_path)
    config_gains = load_gains(DEFAULT_DIMENSION_WEIGHTS, path=weights_path)
    return config_weights, config_gains


def _top_k_profile_ids(result: Dict[str, Any], top_k_value: int) -> List[str]:
    ranked_candidates = result.get("ranked_candidates") or []
    profile_ids: List[str] = []
    for candidate in ranked_candidates[:top_k_value]:
        candidate_id = str(candidate.get("profile_id") or "").strip()
        if candidate_id:
            profile_ids.append(candidate_id)
    return profile_ids


def _jaccard(profile_ids_a: List[str], profile_ids_b: List[str]) -> float:
    """Jaccard on top-K *as sets* (ordering doesn't count). 0 = disjoint, 1 = identical members."""
    set_a = set(profile_ids_a)
    set_b = set(profile_ids_b)
    if not set_a and not set_b:
        return 1.0
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 1.0
    return len(set_a & set_b) / float(union_size)


def _format_ids(profile_ids: List[str]) -> str:
    return ", ".join(profile_ids) if profile_ids else "(empty)"


def _run_one(
    prompt_entry: Dict[str, Any],
    weights_config: Dict[str, float],
    gains_config: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return run_recruiter_search(
        question_text=str(prompt_entry["query"]),
        top_k=int(prompt_entry["top_k"]),
        min_experience_entries=int(prompt_entry["min_experience"]),
        weights_override=weights_config,
        gains_override=gains_config,
    )


def _format_markdown_table(rows: List[Dict[str, Any]], top_k_value: int) -> str:
    header_lines = [
        f"| Prompt | A top-{top_k_value} | B top-{top_k_value} | Jaccard | Δ ordering |",
        "|---|---|---|---|---|",
    ]
    body_lines: List[str] = []
    for row in rows:
        ordering_changed = "yes" if row["ids_a"] != row["ids_b"] else "no"
        body_lines.append(
            f"| {row['name']} | {_format_ids(row['ids_a'])} | "
            f"{_format_ids(row['ids_b'])} | {row['jaccard']:.2f} | {ordering_changed} |"
        )
    return "\n".join(header_lines + body_lines)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-a",
        required=True,
        help="Path to weights JSON for config A (e.g. config/weights.json).",
    )
    parser.add_argument(
        "--weights-b",
        required=True,
        help="Path to weights JSON for config B (e.g. an archived history file).",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Optional YAML file of prompts. Falls back to the smoke test's 3 prompts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many top candidates to compare (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--label-a",
        default="A",
        help="Short display label for config A (default 'A').",
    )
    parser.add_argument(
        "--label-b",
        default="B",
        help="Short display label for config B (default 'B').",
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_cli_args()

    weights_a_path = _resolve_weights_path(cli_args.weights_a)
    weights_b_path = _resolve_weights_path(cli_args.weights_b)
    weights_a, gains_a = _load_config(weights_a_path)
    weights_b, gains_b = _load_config(weights_b_path)

    prompts_path = _resolve_weights_path(cli_args.prompts) if cli_args.prompts else None
    prompts = _load_prompts(prompts_path)

    print(f"# A/B weights compare\n")
    print(f"- Config {cli_args.label_a}: `{weights_a_path}`")
    print(f"- Config {cli_args.label_b}: `{weights_b_path}`")
    print(f"- Prompts: {len(prompts)} ({prompts_path or 'fallback (smoke prompts)'})")
    print(f"- Top-K compared: {cli_args.top_k}\n")

    rows: List[Dict[str, Any]] = []
    aggregate_jaccard = 0.0

    for prompt_index, prompt_entry in enumerate(prompts, start=1):
        prompt_name = str(prompt_entry["name"])
        try:
            result_a = _run_one(prompt_entry, weights_a, gains_a)
            result_b = _run_one(prompt_entry, weights_b, gains_b)
        except Exception as run_error:  # noqa: BLE001 - surface and continue
            print(
                f"[{prompt_index}/{len(prompts)}] FAIL prompt '{prompt_name}': "
                f"{run_error}"
            )
            continue
        ids_a = _top_k_profile_ids(result_a, cli_args.top_k)
        ids_b = _top_k_profile_ids(result_b, cli_args.top_k)
        jaccard_value = _jaccard(ids_a, ids_b)
        rows.append(
            {
                "name": prompt_name,
                "ids_a": ids_a,
                "ids_b": ids_b,
                "jaccard": jaccard_value,
            }
        )
        aggregate_jaccard += jaccard_value
        print(
            f"[{prompt_index}/{len(prompts)}] {prompt_name}: "
            f"Jaccard={jaccard_value:.2f} A={ids_a} B={ids_b}"
        )

    if not rows:
        print("\nNo successful comparisons.")
        return 1

    mean_jaccard = aggregate_jaccard / float(len(rows))
    print("\n## Results\n")
    print(_format_markdown_table(rows, cli_args.top_k))
    print(f"\nMean top-{cli_args.top_k} Jaccard across {len(rows)} prompts: **{mean_jaccard:.3f}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
