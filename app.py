"""Streamlit portfolio demo for the LangGraph recruiter agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

from src.langgraph_app import run_recruiter_search

APP_TITLE = "LangGraph Recruiter Agent"
APP_SUBTITLE = "Query natural language -> LangGraph pipeline over 1k LinkedIn profiles"
DEFAULT_QUERY = "Senior technical recruiter with experience hiring ML engineers in the US"
DEFAULT_TOP_K = 8
MIN_TOP_K = 3
MAX_TOP_K = 15
MAX_MIN_EXPERIENCE = 15
ABOUT_PREVIEW_CHARS = 500

ARCHITECTURE_MERMAID = """
```mermaid
flowchart LR
    userQuery["User query (Streamlit)"] --> parseQuery[parse_query]
    parseQuery -->|"structured filters"| retrieveCandidates[retrieve_candidates]
    retrieveCandidates -->|"top-K rows (MySQL)"| rankCandidates[rank_candidates]
    rankCandidates -->|"scored + rationales"| synthesizeShortlist[synthesize_shortlist]
    synthesizeShortlist --> uiResult["Shortlist + citations"]
```
""".strip()


def _configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":mag:",
        layout="wide",
    )


def _render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Search")
        query_text = st.text_area(
            "Role brief",
            value=DEFAULT_QUERY,
            height=140,
            help="Describe the candidate profile you are looking for.",
        )
        top_k_value = st.slider(
            "Top K candidates",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            help="How many candidates the retriever returns before ranking.",
        )
        min_experience_value = st.slider(
            "Min experience entries",
            min_value=0,
            max_value=MAX_MIN_EXPERIENCE,
            value=0,
            help="Filter out profiles with fewer experience records than this.",
        )
        run_button = st.button("Run recruiter agent", type="primary")

        st.divider()
        st.subheader("Architecture")
        st.markdown(ARCHITECTURE_MERMAID)
        st.caption("Prototype retrieval: lexical SQL filter + LLM ranking. Not production RAG.")

    return {
        "query_text": query_text,
        "top_k": int(top_k_value),
        "min_experience": int(min_experience_value),
        "run_button": run_button,
    }


def _render_parsed_filters(parsed_query: Dict[str, Any]) -> None:
    st.subheader("Parsed filters")
    st.json(parsed_query, expanded=False)


def _truncate_about_for_ui(about_text: str) -> str:
    if not about_text:
        return ""
    if len(about_text) <= ABOUT_PREVIEW_CHARS:
        return about_text
    return about_text[:ABOUT_PREVIEW_CHARS].rstrip() + "..."


def _render_candidate_cards(ranked_candidates: List[Dict[str, Any]]) -> None:
    st.subheader(f"Ranked candidates ({len(ranked_candidates)})")
    if not ranked_candidates:
        st.info("No candidates returned for the current filters.")
        return

    for rank_index, candidate in enumerate(ranked_candidates, start=1):
        header_name = candidate.get("name") or candidate.get("profile_id") or "Unknown"
        rank_score = candidate.get("rank_score", 0)
        with st.container(border=True):
            header_columns = st.columns([4, 1])
            with header_columns[0]:
                st.markdown(f"**{rank_index}. {header_name}**")
                headline = candidate.get("headline") or ""
                location = candidate.get("location") or ""
                subtitle_bits = [bit for bit in (headline, location) if bit]
                if subtitle_bits:
                    st.caption(" | ".join(subtitle_bits))
            with header_columns[1]:
                st.metric("Score", f"{rank_score}/10")

            chip_columns = st.columns(3)
            chip_columns[0].markdown(f"Skills: **{candidate.get('skills_count', 0)}**")
            chip_columns[1].markdown(f"Experience: **{candidate.get('experience_count', 0)}**")
            chip_columns[2].markdown(f"Education: **{candidate.get('education_count', 0)}**")

            match_reasons = candidate.get("match_reasons") or []
            if match_reasons:
                st.markdown("**Why matched**")
                for reason in match_reasons:
                    st.markdown(f"- {reason}")

            risks = candidate.get("risks") or []
            if risks:
                st.markdown("**Risks**")
                for risk in risks:
                    st.markdown(f"- {risk}")

            about_preview = _truncate_about_for_ui(candidate.get("about_text", ""))
            if about_preview:
                with st.expander("About (preview)"):
                    st.write(about_preview)

            st.caption(f"Source: `{candidate.get('profile_id', '')}`")


def _render_shortlist(summary_text: str) -> None:
    st.subheader("Shortlist summary")
    if summary_text:
        st.write(summary_text)
    else:
        st.info("Run a query to see the shortlist summary.")


def _render_errors(error_messages: List[str]) -> None:
    if not error_messages:
        return
    with st.expander("Run warnings", expanded=False):
        for message in error_messages:
            st.warning(message)


def main() -> None:
    _configure_page()
    _render_header()

    sidebar_values = _render_sidebar()
    query_text = sidebar_values["query_text"].strip()

    if not sidebar_values["run_button"]:
        st.info("Enter a role brief on the left and click **Run recruiter agent**.")
        return

    if not query_text:
        st.warning("Please enter a role brief before running the agent.")
        return

    with st.status("Running LangGraph pipeline...", expanded=False) as run_status:
        try:
            result = run_recruiter_search(
                question_text=query_text,
                top_k=sidebar_values["top_k"],
                min_experience_entries=sidebar_values["min_experience"],
            )
            run_status.update(label="Pipeline complete", state="complete")
        except Exception as run_error:  # noqa: BLE001 - surface to user
            run_status.update(label="Pipeline failed", state="error")
            st.error(f"LangGraph run failed: {run_error}")
            return

    top_columns = st.columns([1, 1])
    with top_columns[0]:
        _render_parsed_filters(result.get("parsed_query", {}))
    with top_columns[1]:
        _render_shortlist(result.get("shortlist_summary", ""))

    _render_candidate_cards(result.get("ranked_candidates", []))
    _render_errors(result.get("error_messages", []))

    with st.expander("Raw result JSON (debug)"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
