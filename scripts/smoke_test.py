"""Headless smoke test for the recruiter LangGraph pipeline.

Runs three representative prompts through `run_recruiter_search` and validates
that the parsed filters, ranked candidate list, and shortlist summary are
populated. Prints a compact pass/fail report to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.langgraph_app import run_recruiter_search

REQUIRED_DIMENSION_KEYS = frozenset(
    {
        "phd_researcher",
        "sf_location_fit",
        "technical_background",
        "education_prestige",
        "founder_experience",
    }
)

TEST_PROMPTS = [
    {
        "label": "role-focused",
        "question": "Senior technical recruiter focused on hiring ML engineers in the US",
        "top_k": 6,
        "min_experience": 3,
    },
    {
        "label": "skill-focused",
        "question": "Unity game developer with C# experience and shipped mobile titles",
        "top_k": 6,
        "min_experience": 2,
    },
    {
        "label": "location-focused",
        "question": "Product designer based in the Bay Area with fintech background",
        "top_k": 6,
        "min_experience": 2,
    },
]


def _fail(label: str, reason: str) -> None:
    print(f"FAIL[{label}]: {reason}")


def _passed(label: str, result: dict) -> None:
    ranked = result.get("ranked_candidates") or []
    top_scores = [c.get("rank_score") for c in ranked[:3]]
    top_dimension_sample = (ranked[0].get("dimension_scores") if ranked else {}) or {}
    print(
        f"PASS[{label}]: candidates={len(ranked)}, top_scores={top_scores}, "
        f"top_dims={top_dimension_sample}"
    )


def main() -> int:
    failures = 0
    for prompt in TEST_PROMPTS:
        label = prompt["label"]
        try:
            result = run_recruiter_search(
                question_text=prompt["question"],
                top_k=prompt["top_k"],
                min_experience_entries=prompt["min_experience"],
            )
        except Exception as error:  # noqa: BLE001
            _fail(label, f"exception: {error}")
            failures += 1
            continue

        parsed_query = result.get("parsed_query") or {}
        if not isinstance(parsed_query, dict) or "role_keywords" not in parsed_query:
            _fail(label, "missing parsed_query shape")
            failures += 1
            continue

        ranked_candidates = result.get("ranked_candidates") or []
        if not ranked_candidates:
            _fail(label, "no ranked candidates returned")
            failures += 1
            continue

        summary_text = result.get("shortlist_summary") or ""
        if not summary_text.strip():
            _fail(label, "empty shortlist summary")
            failures += 1
            continue

        first_candidate = ranked_candidates[0]
        dimension_scores = first_candidate.get("dimension_scores") or {}
        if not REQUIRED_DIMENSION_KEYS.issubset(dimension_scores.keys()):
            _fail(
                label,
                f"missing dimension_scores keys: got {sorted(dimension_scores.keys())}",
            )
            failures += 1
            continue

        _passed(label, result)

    if failures:
        print(f"TOTAL: {failures} failure(s)")
        return 1
    print("TOTAL: all prompts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
