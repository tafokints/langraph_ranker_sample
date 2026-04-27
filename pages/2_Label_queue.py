"""Streamlit page: rate unlabeled candidates one at a time.

Companion page to `app.py`. The cards page rates *ranked* candidates from a
search; this page just walks through unlabeled profiles so the human can
build label volume fast (target: ~50/week to make the calibration loop
real instead of synthetic).

Uses the same `save_label` / dimension keys as the cards page so a label
saved here is indistinguishable from one saved during a search session.
Defaults each per-dimension slider to the *deterministic* heuristic score
for that profile (no LLM call) so the labeler can override only the
dimensions they disagree with — same UX as the cards page.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from src.labels_store import (
    count_labels,
    fetch_unlabeled_candidates,
    save_label,
)
from src.langgraph_app import (
    DIMENSION_KEYS,
    _compute_dimension_scores,
    _aggregate_rank_score,
)

PAGE_TITLE = "Label queue"
DEFAULT_LABELER_NAME = "me"
DEFAULT_BATCH_SIZE = 20
MIN_BATCH_SIZE = 5
MAX_BATCH_SIZE = 50
SCORE_SLIDER_STEP = 0.5

# Dimension display order matches the cards page so the UX is identical
# whether the human is rating from a search or from the queue.
DIMENSION_DISPLAY_ORDER = (
    ("technical_background", "Technical"),
    ("founder_experience", "Founder"),
    ("phd_researcher", "PhD / Research"),
    ("education_prestige", "Education"),
    ("sf_location_fit", "SF fit"),
)

QUEUE_KEY = "label_queue"
QUEUE_INDEX_KEY = "label_queue_index"
QUEUE_LABELER_KEY = "label_queue_labeler"
QUEUE_PREFER_DIVERSE_KEY = "label_queue_prefer_diverse"
QUEUE_BATCH_SIZE_KEY = "label_queue_batch_size"
ABOUT_PREVIEW_CHARS = 700


def _configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":pencil2:",
        layout="wide",
    )


def _truncate_about(about_text: str) -> str:
    if not about_text:
        return ""
    if len(about_text) <= ABOUT_PREVIEW_CHARS:
        return about_text
    return about_text[:ABOUT_PREVIEW_CHARS].rstrip() + "..."


def _heuristic_defaults_for(profile: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic scorer outputs as slider defaults. No LLM call."""
    dim_scores, _reasons = _compute_dimension_scores(profile)
    return {key: float(value) for key, value in dim_scores.items()}


def _heuristic_overall(profile_dim_scores: Dict[str, float]) -> float:
    """Heuristic aggregate (with current weights/gains) so the overall
    slider has a reasonable default too."""
    return float(_aggregate_rank_score(profile_dim_scores))


def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Label queue")
        st.caption(
            "One-at-a-time rating page. The queue pulls profiles you "
            "haven't labeled yet."
        )

        if QUEUE_LABELER_KEY not in st.session_state:
            st.session_state[QUEUE_LABELER_KEY] = DEFAULT_LABELER_NAME
        labeler_name = st.text_input(
            "Labeler",
            key=QUEUE_LABELER_KEY,
            help="Your name/handle; attached to every rating you save.",
        )

        if QUEUE_BATCH_SIZE_KEY not in st.session_state:
            st.session_state[QUEUE_BATCH_SIZE_KEY] = DEFAULT_BATCH_SIZE
        batch_size_value = st.slider(
            "Queue size",
            min_value=MIN_BATCH_SIZE,
            max_value=MAX_BATCH_SIZE,
            value=st.session_state[QUEUE_BATCH_SIZE_KEY],
            help="How many unlabeled profiles to fetch in one batch.",
            key=QUEUE_BATCH_SIZE_KEY,
        )

        if QUEUE_PREFER_DIVERSE_KEY not in st.session_state:
            st.session_state[QUEUE_PREFER_DIVERSE_KEY] = True
        prefer_diverse_value = st.checkbox(
            "Prefer rich profiles",
            key=QUEUE_PREFER_DIVERSE_KEY,
            help=(
                "Order the queue by about_text length / experience count. "
                "Labels on richer profiles move the calibrator more than "
                "labels on thin/empty profiles."
            ),
        )

        load_clicked = st.button(
            "Load / refresh queue",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        cleaned_labeler = (labeler_name or "").strip() or DEFAULT_LABELER_NAME
        labels_for_user = count_labels(labeler=cleaned_labeler)
        st.caption(
            f"Labels by `{cleaned_labeler}`: **{labels_for_user}**. "
            "Target: 50/week for a useful calibration run."
        )

    return {
        "labeler": (labeler_name or "").strip() or DEFAULT_LABELER_NAME,
        "batch_size": int(batch_size_value),
        "prefer_diverse": bool(prefer_diverse_value),
        "load_clicked": bool(load_clicked),
    }


def _load_queue(labeler: str, batch_size: int, prefer_diverse: bool) -> List[Dict[str, Any]]:
    try:
        return fetch_unlabeled_candidates(
            labeler=labeler,
            limit=batch_size,
            prefer_diverse=prefer_diverse,
        )
    except Exception as fetch_error:  # noqa: BLE001 - surface DB errors inline
        st.error(f"Failed to load queue: {fetch_error}")
        return []


def _render_profile_snippet(profile: Dict[str, Any]) -> None:
    name_value = profile.get("name") or profile.get("profile_id") or "Unknown"
    headline_value = profile.get("headline") or ""
    location_value = profile.get("location") or "(location unknown)"
    profile_id_value = profile.get("profile_id") or ""

    st.markdown(f"### {name_value}")
    st.caption(
        f"`{profile_id_value}` · {headline_value or 'no headline'} · {location_value}"
    )
    skills_count_value = int(profile.get("skills_count") or 0)
    experience_count_value = int(profile.get("experience_count") or 0)
    education_count_value = int(profile.get("education_count") or 0)
    st.caption(
        f"Skills: {skills_count_value} · Experience entries: "
        f"{experience_count_value} · Education entries: {education_count_value}"
    )

    about_value = profile.get("about_text") or ""
    if about_value:
        with st.expander("About text", expanded=True):
            st.write(_truncate_about(about_value))
    else:
        st.caption("(no about_text on this profile)")


def _render_rating_form(
    profile: Dict[str, Any],
    labeler_name: str,
    on_saved: Any,
) -> None:
    profile_id_value = str(profile.get("profile_id") or "").strip()
    if not profile_id_value:
        st.error("Profile is missing profile_id; skipping.")
        return

    heuristic_dim_scores = _heuristic_defaults_for(profile)
    heuristic_overall_value = _heuristic_overall(heuristic_dim_scores)

    rating_columns = st.columns(len(DIMENSION_DISPLAY_ORDER))
    submitted_dim_scores: Dict[str, float] = {}
    for rating_column, (dimension_key, dimension_label) in zip(
        rating_columns, DIMENSION_DISPLAY_ORDER
    ):
        default_value = max(
            0.0,
            min(10.0, float(heuristic_dim_scores.get(dimension_key, 5.0) or 5.0)),
        )
        submitted_dim_scores[dimension_key] = rating_column.slider(
            dimension_label,
            min_value=0.0,
            max_value=10.0,
            value=default_value,
            step=SCORE_SLIDER_STEP,
            key=f"queue_{profile_id_value}_{dimension_key}",
        )

    overall_input_value = st.slider(
        "Overall fit (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=max(0.0, min(10.0, heuristic_overall_value)),
        step=SCORE_SLIDER_STEP,
        key=f"queue_{profile_id_value}_overall",
    )
    note_input_value = st.text_input(
        "Note (optional)",
        key=f"queue_{profile_id_value}_note",
        help="Short context, e.g. 'looks like a strong founder signal but mostly enterprise sales'.",
    )

    button_columns = st.columns([1, 1, 4])
    save_clicked = button_columns[0].button(
        "Save & next",
        key=f"queue_{profile_id_value}_save",
        type="primary",
    )
    skip_clicked = button_columns[1].button(
        "Skip",
        key=f"queue_{profile_id_value}_skip",
        type="secondary",
    )

    if save_clicked:
        try:
            save_label(
                profile_id=profile_id_value,
                labeler=labeler_name,
                dim_scores=submitted_dim_scores,
                overall_score=float(overall_input_value),
                note=note_input_value,
                required_dimension_keys=list(DIMENSION_KEYS),
            )
        except Exception as save_error:  # noqa: BLE001 - inline error
            st.error(f"Failed to save rating: {save_error}")
            return
        st.success(f"Saved rating for {profile_id_value}.")
        on_saved()
    elif skip_clicked:
        on_saved()


def _advance_index() -> None:
    current_index_value = int(st.session_state.get(QUEUE_INDEX_KEY, 0))
    st.session_state[QUEUE_INDEX_KEY] = current_index_value + 1


def _render_progress(current_index: int, total: int) -> None:
    if total <= 0:
        return
    capped_index = min(current_index, total)
    completion_fraction = capped_index / float(total)
    st.progress(
        completion_fraction,
        text=f"Candidate {min(capped_index + 1, total)} of {total} (saved/skipped: {capped_index})",
    )


def main() -> None:
    _configure_page()
    st.title(PAGE_TITLE)
    st.caption(
        "Walk through unlabeled candidates and rate them. Each saved label "
        "contributes to the next `python scripts/calibrate.py` fit. Use the "
        "main page (`app.py`) when you want to label *ranked* search results "
        "instead."
    )

    sidebar_values = _render_sidebar()

    if sidebar_values["load_clicked"]:
        st.session_state[QUEUE_KEY] = _load_queue(
            labeler=sidebar_values["labeler"],
            batch_size=sidebar_values["batch_size"],
            prefer_diverse=sidebar_values["prefer_diverse"],
        )
        st.session_state[QUEUE_INDEX_KEY] = 0

    queue_items: List[Dict[str, Any]] = st.session_state.get(QUEUE_KEY, [])
    if not queue_items:
        st.info(
            "Click **Load / refresh queue** in the sidebar to fetch a batch "
            "of unlabeled candidates."
        )
        return

    current_index_value = int(st.session_state.get(QUEUE_INDEX_KEY, 0))
    total_items = len(queue_items)
    _render_progress(current_index_value, total_items)

    if current_index_value >= total_items:
        st.success(
            f"Queue complete - rated/skipped {total_items} candidate(s). "
            "Reload the queue to fetch a new batch."
        )
        return

    current_profile: Dict[str, Any] = queue_items[current_index_value]
    with st.container(border=True):
        _render_profile_snippet(current_profile)
        st.divider()
        st.caption(
            "Score each dimension 0-10. Defaults are the deterministic "
            "heuristic; only move the dimensions you disagree with."
        )
        _render_rating_form(
            profile=current_profile,
            labeler_name=sidebar_values["labeler"],
            on_saved=_advance_index,
        )


if __name__ == "__main__":
    main()
