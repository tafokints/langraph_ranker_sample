"""Round-4 ablation harness for the recruiter graph.

Runs five configurations against a fixed prompt list and emits a markdown
table that quantifies how much each Round 1/2/3 feature contributes to the
final top-K ranking. The five rows are:

  1. baseline:   deterministic only (no LLM listwise rerank, no
                 enrich_low_info, no pairwise tie-break, default weights v1)
  2. + listwise: baseline + LLM listwise rerank
  3. + enrich:   baseline + listwise + enrich_low_info
  4. + pairwise: baseline + listwise + enrich + pairwise tie-break
  5. + v2 weights: row (4) but with the fitted weights+gains from
                   `config/weights.json` applied

Each row's top-K is compared against the *full pipeline* (row 5) using
top-K Jaccard. The full pipeline is the de facto reference until R4-1 has
collected enough real labels for a true human-preferred ordering — at
which point the README ablation table should be regenerated with
`--reference-from-labels` (TODO once R4-1 lands).

Usage:
    python scripts/ablation_table.py --prompts fixtures/smoke_prompts.yml --top-k 3

Notes:
  - 5 configs x N prompts is N x 5 LangGraph runs. Budget the LLM credits.
  - Configs 1-2 still hit the LLM via the pointwise scorer; only listwise/
    enrich/pairwise are flagged off. To avoid LLM entirely, unset
    OPENAI_API_KEY before running — the deterministic fallback still works
    for every config but the absolute scores will differ from production.
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
from src.weights_loader import (
    default_dimension_gains,
    load_gains,
    load_weights,
    weights_file_path,
)

DEFAULT_TOP_K = 3
DEFAULT_PROMPTS_TOP_K = 6
DEFAULT_PROMPTS_MIN_EXPERIENCE = 2

# Each config = (label, feature_flags, weights_provider, gains_provider).
# feature_flags is a dict of stage names -> True (skip). weights_provider
# returns the weight dict; gains_provider returns the per-dimension affine
# transform. Using callables lets us defer config/weights.json reads until
# the user actually runs the script (so a missing weights file fails
# cleanly with a helpful error).


def _v1_weights() -> Dict[str, float]:
    return dict(DEFAULT_DIMENSION_WEIGHTS)


def _v1_gains() -> Dict[str, Dict[str, float]]:
    return default_dimension_gains(DEFAULT_DIMENSION_WEIGHTS)


def _v2_weights() -> Dict[str, float]:
    return load_weights(DEFAULT_DIMENSION_WEIGHTS, path=weights_file_path())


def _v2_gains() -> Dict[str, Dict[str, float]]:
    return load_gains(DEFAULT_DIMENSION_WEIGHTS, path=weights_file_path())


ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {
        "label": "baseline (no rerank, no enrich, no pairwise, weights v1)",
        "feature_flags": {
            "disable_listwise_rerank": True,
            "disable_enrich_low_info": True,
            "disable_pairwise_tiebreak": True,
        },
        "weights_provider": _v1_weights,
        "gains_provider": _v1_gains,
    },
    {
        "label": "+ listwise rerank",
        "feature_flags": {
            "disable_listwise_rerank": False,
            "disable_enrich_low_info": True,
            "disable_pairwise_tiebreak": True,
        },
        "weights_provider": _v1_weights,
        "gains_provider": _v1_gains,
    },
    {
        "label": "+ enrich_low_info",
        "feature_flags": {
            "disable_listwise_rerank": False,
            "disable_enrich_low_info": False,
            "disable_pairwise_tiebreak": True,
        },
        "weights_provider": _v1_weights,
        "gains_provider": _v1_gains,
    },
    {
        "label": "+ pairwise tie-break",
        "feature_flags": {
            "disable_listwise_rerank": False,
            "disable_enrich_low_info": False,
            "disable_pairwise_tiebreak": False,
        },
        "weights_provider": _v1_weights,
        "gains_provider": _v1_gains,
    },
    {
        "label": "+ v2 weights (fitted)",
        "feature_flags": {
            "disable_listwise_rerank": False,
            "disable_enrich_low_info": False,
            "disable_pairwise_tiebreak": False,
        },
        "weights_provider": _v2_weights,
        "gains_provider": _v2_gains,
    },
]


# --- Prompt loading (mirrors scripts/ab_compare_weights.py) -----------------


def _load_prompts(prompts_path: Optional[Path]) -> List[Dict[str, Any]]:
    if prompts_path is None:
        return _fallback_prompts()
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as import_error:
        raise SystemExit(
            "PyYAML is required to read --prompts. Install it with: pip install pyyaml"
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


def _fallback_prompts() -> List[Dict[str, Any]]:
    return [
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


# --- Helpers (mirror scripts/ab_compare_weights.py for consistency) ---------


def _top_k_profile_ids(result: Dict[str, Any], top_k_value: int) -> List[str]:
    ranked_candidates = result.get("ranked_candidates") or []
    profile_ids: List[str] = []
    for candidate in ranked_candidates[:top_k_value]:
        candidate_id = str(candidate.get("profile_id") or "").strip()
        if candidate_id:
            profile_ids.append(candidate_id)
    return profile_ids


def _jaccard(profile_ids_a: List[str], profile_ids_b: List[str]) -> float:
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


# --- Main run loop -----------------------------------------------------------


def _run_one_config(
    prompt_entry: Dict[str, Any],
    config_entry: Dict[str, Any],
) -> Dict[str, Any]:
    return run_recruiter_search(
        question_text=str(prompt_entry["query"]),
        top_k=int(prompt_entry["top_k"]),
        min_experience_entries=int(prompt_entry["min_experience"]),
        weights_override=config_entry["weights_provider"](),
        gains_override=config_entry["gains_provider"](),
        feature_flags=dict(config_entry["feature_flags"]),
    )


def _format_summary_table(
    aggregate_jaccard_by_config_label: Dict[str, float],
    successful_prompt_count: int,
    top_k_value: int,
) -> str:
    lines: List[str] = [
        f"## Round 4 ablation summary (top-{top_k_value} Jaccard vs full pipeline)",
        "",
        "| # | configuration | mean Jaccard vs full pipeline |",
        "|---|---|---|",
    ]
    for index, config_entry in enumerate(ABLATION_CONFIGS, start=1):
        config_label = config_entry["label"]
        mean_jaccard_value = aggregate_jaccard_by_config_label.get(config_label, 0.0)
        if successful_prompt_count > 0:
            mean_jaccard_value = mean_jaccard_value / float(successful_prompt_count)
        else:
            mean_jaccard_value = 0.0
        lines.append(
            f"| {index} | {config_label} | {mean_jaccard_value:.2f} |"
        )
    lines.append("")
    lines.append(
        f"_n = {successful_prompt_count} prompts; reference = full pipeline (row 5)._"
    )
    return "\n".join(lines)


def _format_per_prompt_table(
    per_prompt_results: List[Dict[str, Any]],
    top_k_value: int,
) -> str:
    header_cells = ["| prompt"]
    separator_cells = ["|---"]
    for config_entry in ABLATION_CONFIGS:
        header_cells.append(f"| top-{top_k_value} {config_entry['label']}")
        separator_cells.append("|---")
    header_cells.append(" |")
    separator_cells.append("|")
    lines: List[str] = ["".join(header_cells), "".join(separator_cells)]
    for prompt_row in per_prompt_results:
        cells = [f"| {prompt_row['prompt_name']}"]
        for config_entry in ABLATION_CONFIGS:
            ids_value = prompt_row["ids_by_config"].get(config_entry["label"]) or []
            cells.append(f"| {_format_ids(ids_value)}")
        cells.append(" |")
        lines.append("".join(cells))
    return "\n".join(lines)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        default=None,
        help="Optional YAML file of prompts. Falls back to the smoke fixture's first 3.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many top candidates per prompt (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help=(
            "Optionally cap the number of prompts run. Useful when the full "
            "fixture has 10+ prompts but you want a quick 3-prompt sanity check."
        ),
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_cli_args()
    prompts_path: Optional[Path] = None
    if cli_args.prompts:
        prompts_path = Path(cli_args.prompts)
        if not prompts_path.is_absolute():
            prompts_path = PROJECT_ROOT / prompts_path
    prompts = _load_prompts(prompts_path)
    if cli_args.limit_prompts > 0:
        prompts = prompts[: cli_args.limit_prompts]

    print(f"# Round 4 ablation table\n")
    print(f"- Prompts: {len(prompts)} ({prompts_path or 'fallback'})")
    print(f"- Configs: {len(ABLATION_CONFIGS)}")
    print(f"- Top-K compared: {cli_args.top_k}\n")

    full_pipeline_label = ABLATION_CONFIGS[-1]["label"]
    aggregate_jaccard_by_config_label: Dict[str, float] = {
        config_entry["label"]: 0.0 for config_entry in ABLATION_CONFIGS
    }
    successful_prompt_count = 0
    per_prompt_results: List[Dict[str, Any]] = []

    for prompt_index, prompt_entry in enumerate(prompts, start=1):
        prompt_name = str(prompt_entry["name"])
        print(f"[{prompt_index}/{len(prompts)}] {prompt_name}: running {len(ABLATION_CONFIGS)} configs ...")
        ids_by_config: Dict[str, List[str]] = {}
        prompt_succeeded = True
        for config_entry in ABLATION_CONFIGS:
            try:
                run_result = _run_one_config(prompt_entry, config_entry)
            except Exception as run_error:  # noqa: BLE001 - surface and continue
                print(
                    f"  FAIL config '{config_entry['label']}' on prompt "
                    f"'{prompt_name}': {run_error}"
                )
                prompt_succeeded = False
                continue
            ids_value = _top_k_profile_ids(run_result, cli_args.top_k)
            ids_by_config[config_entry["label"]] = ids_value
        if not prompt_succeeded or full_pipeline_label not in ids_by_config:
            print(f"  skipping prompt '{prompt_name}' (incomplete run set)")
            continue
        successful_prompt_count += 1
        full_pipeline_ids = ids_by_config[full_pipeline_label]
        for config_entry in ABLATION_CONFIGS:
            this_config_ids = ids_by_config.get(config_entry["label"]) or []
            jaccard_against_full = _jaccard(this_config_ids, full_pipeline_ids)
            aggregate_jaccard_by_config_label[config_entry["label"]] += jaccard_against_full
        per_prompt_results.append(
            {
                "prompt_name": prompt_name,
                "ids_by_config": ids_by_config,
            }
        )

    if successful_prompt_count == 0:
        print("\nNo prompts produced a complete run set. Cannot build table.")
        return 1

    print()
    print(_format_summary_table(aggregate_jaccard_by_config_label, successful_prompt_count, cli_args.top_k))
    print()
    print("## Per-prompt top-K by configuration\n")
    print(_format_per_prompt_table(per_prompt_results, cli_args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
