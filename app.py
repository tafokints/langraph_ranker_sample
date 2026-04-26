"""Streamlit portfolio demo for the LangGraph recruiter agent."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st

from src.labels_store import count_labels, save_label
from src.langgraph_app import DIMENSION_KEYS, run_recruiter_search

APP_TITLE = "LangGraph Recruiter Agent"
APP_SUBTITLE = "Query natural language -> LangGraph pipeline over 1k LinkedIn profiles"
DEFAULT_QUERY = "Senior technical recruiter with experience hiring ML engineers in the US"
DEFAULT_TOP_K = 8
MIN_TOP_K = 3
MAX_TOP_K = 15
MAX_MIN_EXPERIENCE = 15
ABOUT_PREVIEW_CHARS = 500

# Four starter chips mapped to the four rubric dimensions we want to exercise.
# Keeps the demo dogfood-able without the reviewer having to invent a query.
STARTER_PROMPTS: List[Dict[str, str]] = [
    {
        "label": "Role",
        "prompt": "Senior staff ML engineer with recommender-systems experience in the Bay Area.",
    },
    {
        "label": "Skill",
        "prompt": "Backend engineer strong in Rust and distributed systems, US-based.",
    },
    {
        "label": "Location",
        "prompt": (
            "Full-stack engineer based in San Francisco or willing to relocate; "
            "open to early-stage startup risk."
        ),
    },
    {
        "label": "Founder",
        "prompt": (
            "Technical co-founder candidate with a PhD and prior founder or "
            "first-engineer experience at a YC-backed company."
        ),
    },
]

ROLE_BRIEF_KEY = "role_brief_input"
LAST_RESULT_KEY = "last_result"
LAST_QUERY_KEY = "last_query"
LAST_TOP_K_KEY = "last_top_k"
LAST_MIN_EXP_KEY = "last_min_experience"

DIMENSION_DISPLAY_ORDER = (
    ("technical_background", "Technical", 0.30),
    ("founder_experience", "Founder", 0.25),
    ("phd_researcher", "PhD / Research", 0.15),
    ("education_prestige", "Education", 0.15),
    ("sf_location_fit", "SF fit", 0.15),
)
RUBRIC_HELP = (
    "Overall score = weighted aggregate of 5 dimensions. "
    "Weights are loaded from config/weights.json (fit by scripts/calibrate.py) "
    "or the hardcoded defaults if that file is missing."
)
DEFAULT_LABELER_NAME = "me"
SCORE_SLIDER_STEP = 0.5

ARCHITECTURE_MERMAID = """
```mermaid
flowchart LR
    userQuery["User query (Streamlit)"] --> parseQuery[parse_query]
    parseQuery -->|"with_structured_output(ParsedQueryModel)"| retrieveCandidates[retrieve_candidates]
    retrieveCandidates -->|"any about<200"| enrichLowInfo[enrich_low_info]
    retrieveCandidates -->|"skip"| rankCandidates[rank_candidates]
    enrichLowInfo --> rankCandidates
    rankCandidates -->|"pointwise + listwise rerank"| synthesizeShortlist[synthesize_shortlist]
    synthesizeShortlist --> uiResult["Shortlist + citations"]
```
""".strip()


def _tracing_enabled() -> bool:
    """True when either LANGSMITH_TRACING or LANGCHAIN_TRACING_V2 is set truthy."""
    for env_var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if os.environ.get(env_var, "").strip().lower() in ("true", "1", "yes"):
            return True
    return False


def _configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":mag:",
        layout="wide",
    )


def _render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def _render_starter_prompts() -> None:
    """Four single-click starter chips. Clicking a chip sets the role brief
    via `st.session_state`; the text_area below picks it up via its `key`.
    """
    st.caption("Starter prompts — click one to pre-fill the role brief:")
    chip_columns = st.columns(len(STARTER_PROMPTS))
    for chip_column, prompt_entry in zip(chip_columns, STARTER_PROMPTS):
        if chip_column.button(
            prompt_entry["label"],
            key=f"chip_{prompt_entry['label'].lower()}",
            use_container_width=True,
            help=prompt_entry["prompt"],
        ):
            st.session_state[ROLE_BRIEF_KEY] = prompt_entry["prompt"]


def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Search")
        _render_starter_prompts()
        if ROLE_BRIEF_KEY not in st.session_state:
            st.session_state[ROLE_BRIEF_KEY] = DEFAULT_QUERY
        query_text = st.text_area(
            "Role brief",
            key=ROLE_BRIEF_KEY,
            height=140,
            help="Describe the candidate profile you are looking for.",
        )
        top_k_value = st.slider(
            "Top K candidates",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=st.session_state.get(LAST_TOP_K_KEY, DEFAULT_TOP_K),
            help="How many candidates the retriever returns before ranking.",
        )
        min_experience_value = st.slider(
            "Min experience entries",
            min_value=0,
            max_value=MAX_MIN_EXPERIENCE,
            value=st.session_state.get(LAST_MIN_EXP_KEY, 0),
            help="Filter out profiles with fewer experience records than this.",
        )
        run_button = st.button("Run recruiter agent", type="primary")

        st.divider()
        st.subheader("Calibration")
        labeler_name = st.text_input(
            "Labeler",
            value=DEFAULT_LABELER_NAME,
            help="Your name/handle; attached to every rating you save.",
        )
        total_labels_collected = count_labels()
        labels_by_user = count_labels(labeler=labeler_name) if labeler_name.strip() else 0
        st.caption(
            f"Labels collected: **{total_labels_collected}** total"
            f" · **{labels_by_user}** by `{labeler_name or '-'}`"
        )
        st.caption(
            "Rate candidates below to teach the rubric what 'good' looks like; "
            "then run `python scripts/calibrate.py` to refit the weights."
        )

        st.divider()
        st.subheader("Tracing")
        if _tracing_enabled():
            st.caption(
                "LangSmith tracing is **on** (env var detected). "
                "Each run is a trace tree with one child run per node."
            )
            latest_trace_url = st.session_state.get("latest_trace_url")
            if latest_trace_url:
                st.markdown(f"[Open latest run in LangSmith]({latest_trace_url})")
            else:
                st.caption("Run a query to populate the trace link.")
        else:
            st.caption(
                "LangSmith tracing is **off**. "
                "Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` to enable."
            )

        last_token_usage = st.session_state.get("latest_token_usage")
        if last_token_usage:
            st.caption(
                f"Last run: **{int(last_token_usage.get('llm_calls', 0))}** LLM calls · "
                f"**{int(last_token_usage.get('total_tokens', 0))}** tokens · "
                f"~**${float(last_token_usage.get('estimated_cost_usd', 0.0)):.5f}** "
                f"({last_token_usage.get('model', '?')})"
            )

    return {
        "query_text": query_text,
        "top_k": int(top_k_value),
        "min_experience": int(min_experience_value),
        "labeler": labeler_name.strip() or DEFAULT_LABELER_NAME,
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


def _render_rating_form(
    candidate: Dict[str, Any],
    labeler_name: str,
    default_expanded: bool = False,
) -> None:
    """Collapsed per-candidate rating form (5 per-dim + 1 overall + note).

    Sliders default to the system's current scores so the labeler can override
    only the dimensions they disagree with. All widget keys are namespaced by
    profile_id so multiple cards on the page don't collide. The rank-1 card
    opens by default so the "this is how you teach the rubric" loop is
    visible without scrolling.
    """
    profile_id_value = str(candidate.get("profile_id", "")).strip()
    if not profile_id_value:
        return

    heuristic_dimension_scores = candidate.get("dimension_scores") or {}
    heuristic_overall_score = float(candidate.get("rank_score") or 5.0)

    with st.expander("Rate this candidate", expanded=default_expanded):
        st.caption(
            "Score each dimension 0-10 as a non-technical reviewer would; "
            "then score the overall fit. Defaults are the system's current scores."
        )

        rating_columns = st.columns(len(DIMENSION_DISPLAY_ORDER))
        submitted_dimension_scores: Dict[str, float] = {}
        for rating_column, (dimension_key, dimension_label, _weight) in zip(
            rating_columns, DIMENSION_DISPLAY_ORDER
        ):
            try:
                default_value = float(heuristic_dimension_scores.get(dimension_key, 5.0) or 5.0)
            except (TypeError, ValueError):
                default_value = 5.0
            default_value = max(0.0, min(10.0, default_value))
            submitted_dimension_scores[dimension_key] = rating_column.slider(
                dimension_label,
                min_value=0.0,
                max_value=10.0,
                value=default_value,
                step=SCORE_SLIDER_STEP,
                key=f"rate_{profile_id_value}_{dimension_key}",
            )

        overall_input_value = st.slider(
            "Overall fit",
            min_value=0.0,
            max_value=10.0,
            value=max(0.0, min(10.0, heuristic_overall_score)),
            step=SCORE_SLIDER_STEP,
            key=f"rate_{profile_id_value}_overall",
        )
        note_input_value = st.text_input(
            "Note (optional)",
            key=f"rate_{profile_id_value}_note",
            help="Short free-text context, e.g. 'strong founder signal from YC W21'.",
        )
        save_clicked = st.button(
            "Save rating",
            key=f"rate_{profile_id_value}_save",
            type="secondary",
        )

        if save_clicked:
            try:
                save_label(
                    profile_id=profile_id_value,
                    labeler=labeler_name,
                    dim_scores=submitted_dimension_scores,
                    overall_score=float(overall_input_value),
                    note=note_input_value,
                    required_dimension_keys=list(DIMENSION_KEYS),
                )
                st.success(f"Saved rating for {profile_id_value} (labeler: {labeler_name}).")
            except Exception as save_error:  # noqa: BLE001 - show DB / validation errors inline
                st.error(f"Failed to save rating: {save_error}")


def _render_dimension_breakdown(candidate: Dict[str, Any]) -> None:
    dimension_scores = candidate.get("dimension_scores") or {}
    dimension_reasons = candidate.get("dimension_reasons") or {}
    breakdown_columns = st.columns(len(DIMENSION_DISPLAY_ORDER))
    for column, (dimension_key, dimension_label, dimension_weight) in zip(
        breakdown_columns, DIMENSION_DISPLAY_ORDER
    ):
        raw_value = dimension_scores.get(dimension_key, 0.0) or 0.0
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        clamped_value = max(0.0, min(10.0, numeric_value))
        column.markdown(f"**{dimension_label}**  \n<sub>w={int(dimension_weight * 100)}%</sub>", unsafe_allow_html=True)
        column.progress(clamped_value / 10.0, text=f"{clamped_value:.1f}/10")
        reason_text = dimension_reasons.get(dimension_key, "")
        if reason_text:
            column.caption(reason_text)


def _render_candidate_cards(ranked_candidates: List[Dict[str, Any]], labeler_name: str) -> None:
    st.subheader(f"Ranked candidates ({len(ranked_candidates)})")
    st.caption(RUBRIC_HELP)
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
                st.metric("Score", f"{rank_score}/10", help=RUBRIC_HELP)

            _render_dimension_breakdown(candidate)

            st.caption(
                f"{candidate.get('skills_count', 0)} skills"
                f" · {candidate.get('experience_count', 0)} experience"
                f" · {candidate.get('education_count', 0)} education"
            )

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

            _render_rating_form(
                candidate,
                labeler_name,
                default_expanded=(rank_index == 1),
            )

            st.caption(f"Source: `{candidate.get('profile_id', '')}`")


def _render_shortlist(summary_text: str) -> None:
    st.subheader("Shortlist summary")
    if summary_text:
        st.write(summary_text)
    else:
        st.info("Run a query to see the shortlist summary.")


def _render_copy_profile_ids(ranked_candidates: List[Dict[str, Any]]) -> None:
    """Render ranked profile_ids in an `st.code` block so Streamlit's
    built-in copy-to-clipboard widget is usable without a custom JS escape
    hatch. One ID per line keeps it pasteable into any tool.
    """
    if not ranked_candidates:
        return
    profile_id_values = [
        str(candidate.get("profile_id", "")).strip()
        for candidate in ranked_candidates
        if str(candidate.get("profile_id", "")).strip()
    ]
    if not profile_id_values:
        return
    with st.expander(f"Copy {len(profile_id_values)} shortlisted profile_ids"):
        st.caption("Click the copy icon in the top-right of the code block.")
        st.code("\n".join(profile_id_values), language="text")


def _render_run_metrics(token_usage: Dict[str, Any], trace_url: str) -> None:
    if not token_usage and not trace_url:
        return
    metric_columns = st.columns(4)
    metric_columns[0].metric("LLM calls", int(token_usage.get("llm_calls", 0)))
    metric_columns[1].metric("Tokens", int(token_usage.get("total_tokens", 0)))
    metric_columns[2].metric(
        "Est. cost (USD)",
        f"${float(token_usage.get('estimated_cost_usd', 0.0)):.5f}",
    )
    with metric_columns[3]:
        st.metric("Trace", "open" if trace_url else "—")
        if trace_url:
            st.markdown(f"[Open in LangSmith]({trace_url})")


def _render_errors(error_messages: List[str]) -> None:
    if not error_messages:
        return
    with st.expander("Run warnings", expanded=False):
        for message in error_messages:
            st.warning(message)


def _render_architecture_panel() -> None:
    """Top-of-page architecture view. Keeps the graph shape visible on the
    primary surface so reviewers don't have to open the sidebar to see it.
    """
    with st.expander("Architecture (LangGraph pipeline)", expanded=False):
        st.markdown(ARCHITECTURE_MERMAID)
        st.caption(
            "Hybrid retrieval: lexical SQL ∪ FAISS embedding recall. "
            "Two-stage ranking: pointwise scoring + listwise rerank. "
            "Near-ties (<0.3) go through a pairwise LLM tie-break."
        )


def main() -> None:
    _configure_page()
    _render_header()

    sidebar_values = _render_sidebar()
    query_text = sidebar_values["query_text"].strip()
    top_k_value = sidebar_values["top_k"]
    min_experience_value = sidebar_values["min_experience"]
    labeler_value = sidebar_values["labeler"]

    _render_architecture_panel()

    if sidebar_values["run_button"]:
        if not query_text:
            st.warning("Please enter a role brief before running the agent.")
            return
        with st.status("Running LangGraph pipeline...", expanded=False) as run_status:
            try:
                result = run_recruiter_search(
                    question_text=query_text,
                    top_k=top_k_value,
                    min_experience_entries=min_experience_value,
                )
                run_status.update(label="Pipeline complete", state="complete")
            except Exception as run_error:  # noqa: BLE001 - surface to user
                run_status.update(label="Pipeline failed", state="error")
                st.error(f"LangGraph run failed: {run_error}")
                return

        trace_url_value = result.get("trace_url")
        if trace_url_value:
            st.session_state["latest_trace_url"] = trace_url_value
            st.sidebar.success(f"[LangSmith trace]({trace_url_value})")

        token_usage_value = result.get("token_usage") or {}
        if token_usage_value:
            st.session_state["latest_token_usage"] = token_usage_value

        st.session_state[LAST_RESULT_KEY] = result
        st.session_state[LAST_QUERY_KEY] = query_text
        st.session_state[LAST_TOP_K_KEY] = top_k_value
        st.session_state[LAST_MIN_EXP_KEY] = min_experience_value

    cached_result: Optional[Dict[str, Any]] = st.session_state.get(LAST_RESULT_KEY)
    if cached_result is None:
        st.info(
            "Pick a starter prompt or enter a role brief on the left and "
            "click **Run recruiter agent**."
        )
        return

    cached_query_text = st.session_state.get(LAST_QUERY_KEY, "")
    if cached_query_text:
        st.caption(f"Showing last run: `{cached_query_text}`")

    _render_run_metrics(
        cached_result.get("token_usage") or {},
        cached_result.get("trace_url", ""),
    )

    top_columns = st.columns([1, 1])
    with top_columns[0]:
        _render_parsed_filters(cached_result.get("parsed_query", {}))
    with top_columns[1]:
        _render_shortlist(cached_result.get("shortlist_summary", ""))

    ranked_candidates_value = cached_result.get("ranked_candidates", [])
    _render_copy_profile_ids(ranked_candidates_value)
    _render_candidate_cards(ranked_candidates_value, labeler_value)
    _render_errors(cached_result.get("error_messages", []))

    with st.expander("Raw result JSON (debug)"):
        st.code(json.dumps(cached_result, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
