"""CLI entrypoint for LLM-enabled LinkedIn LangGraph Q&A."""

from __future__ import annotations

import argparse
import json
import sys

from src.langgraph_app import run_profile_question


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions over linkedin_api_profiles_parsed using LangGraph.")
    parser.add_argument("question", help="Question to ask about candidate profiles.")
    parser.add_argument("--top-k", type=int, default=8, help="How many profile candidates to retrieve (1-20).")
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print retrieved candidates as JSON for debugging.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = run_profile_question(question_text=args.question, top_k=args.top_k)
    except Exception as error:  # pragma: no cover - runtime guard for CLI
        print(f"LangGraph run failed: {error}", file=sys.stderr)
        return 1

    print("Answer:")
    print(result["answer_text"])

    if args.show_candidates:
        print("\nCandidates:")
        print(json.dumps(result["candidate_profiles"], ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
