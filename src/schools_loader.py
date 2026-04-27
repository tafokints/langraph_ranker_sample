"""Disk-backed lookup for the structured schools corpus used by
`_score_education_prestige`.

Replaces the keyword-only TOP_SCHOOL_TOKENS / STRONG_SCHOOL_TOKENS lists in
`src/langgraph_app.py` with a JSON config of `{name, aliases[], tier, country}`
records plus tier base scores and program modifiers (executive education,
online certificate, doctoral, graduate, undergraduate). Loading happens once
per process; mutating the returned config dict mutates the shared cache, so
callers should treat it as read-only.

The corpus lives at `config/schools.json`. If the file is missing or
malformed we fall back to an empty corpus and the scorer simply returns 0
for any candidate without other prestige signals — same fail-open contract
as `weights_loader`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHOOLS_FILE_RELATIVE_PATH = "config/schools.json"
DEFAULT_TIER_BASE_SCORES: Dict[int, float] = {1: 7.0, 2: 5.0, 3: 3.5}
DEFAULT_PROGRAM_MODIFIER_WINDOW = 80
DEFAULT_PROGRAM_SIGNAL = 1.0


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def schools_file_path() -> Path:
    return _project_root() / SCHOOLS_FILE_RELATIVE_PATH


def _build_alias_index(
    schools: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return a list of (lowercased_alias, school_record) tuples sorted by
    descending alias length. Longest-match-wins keeps "MIT Sloan" from being
    eaten by the bare "mit" alias when both are in the corpus.
    """
    indexed_pairs: List[Tuple[str, Dict[str, Any]]] = []
    for school_record in schools:
        for alias_value in school_record.get("aliases", []) or []:
            if not isinstance(alias_value, str):
                continue
            cleaned_alias = alias_value.strip().lower()
            if not cleaned_alias:
                continue
            indexed_pairs.append((cleaned_alias, school_record))
    indexed_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return indexed_pairs


def _coerce_tier_base_scores(raw_value: Any) -> Dict[int, float]:
    """Accept either int-keyed or string-keyed JSON and produce {int: float}."""
    if not isinstance(raw_value, dict):
        return dict(DEFAULT_TIER_BASE_SCORES)
    coerced: Dict[int, float] = {}
    for tier_key, score_value in raw_value.items():
        try:
            coerced[int(tier_key)] = float(score_value)
        except (TypeError, ValueError):
            continue
    if not coerced:
        return dict(DEFAULT_TIER_BASE_SCORES)
    return coerced


_CACHED_CONFIG: Optional[Dict[str, Any]] = None


def load_schools_config(
    path: Optional[Path] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Read `config/schools.json` and return a normalized dict.

    The returned dict has shape:
    ```
    {
        "schools": [...record...],
        "alias_index": [(alias_lower, record), ...],   # sorted longest-first
        "tier_base_scores": {1: 7.0, 2: 5.0, 3: 3.5},
        "program_modifiers": [...record...],
        "program_modifier_window": int,
        "default_program_signal": float,
    }
    ```

    Pass `force_reload=True` from tests; production callers should let the
    in-memory cache do its job. Callers must not mutate the result.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload and path is None:
        return _CACHED_CONFIG

    resolved_path = path if path is not None else schools_file_path()
    fallback_config: Dict[str, Any] = {
        "schools": [],
        "alias_index": [],
        "tier_base_scores": dict(DEFAULT_TIER_BASE_SCORES),
        "program_modifiers": [],
        "program_modifier_window": DEFAULT_PROGRAM_MODIFIER_WINDOW,
        "default_program_signal": DEFAULT_PROGRAM_SIGNAL,
    }

    if not resolved_path.exists():
        if path is None:
            _CACHED_CONFIG = fallback_config
        return fallback_config

    try:
        with resolved_path.open("r", encoding="utf-8") as schools_file_handle:
            payload = json.load(schools_file_handle)
    except (OSError, json.JSONDecodeError):
        if path is None:
            _CACHED_CONFIG = fallback_config
        return fallback_config

    if not isinstance(payload, dict):
        if path is None:
            _CACHED_CONFIG = fallback_config
        return fallback_config

    raw_schools = payload.get("schools")
    if not isinstance(raw_schools, list):
        raw_schools = []
    cleaned_schools: List[Dict[str, Any]] = []
    for record in raw_schools:
        if not isinstance(record, dict):
            continue
        try:
            tier_value = int(record.get("tier", 0))
        except (TypeError, ValueError):
            continue
        if tier_value <= 0:
            continue
        cleaned_record: Dict[str, Any] = {
            "name": str(record.get("name") or ""),
            "aliases": [
                alias_value
                for alias_value in (record.get("aliases") or [])
                if isinstance(alias_value, str) and alias_value.strip()
            ],
            "tier": tier_value,
            "country": str(record.get("country") or ""),
        }
        if not cleaned_record["aliases"]:
            continue
        cleaned_schools.append(cleaned_record)

    raw_modifiers = payload.get("program_modifiers")
    if not isinstance(raw_modifiers, list):
        raw_modifiers = []
    cleaned_modifiers: List[Dict[str, Any]] = []
    for modifier_record in raw_modifiers:
        if not isinstance(modifier_record, dict):
            continue
        try:
            modifier_value = float(modifier_record.get("modifier", 1.0))
        except (TypeError, ValueError):
            continue
        modifier_tokens = [
            token.lower()
            for token in (modifier_record.get("tokens") or [])
            if isinstance(token, str) and token.strip()
        ]
        if not modifier_tokens:
            continue
        cleaned_modifiers.append(
            {
                "label": str(modifier_record.get("label") or ""),
                "tokens": modifier_tokens,
                "modifier": modifier_value,
            }
        )

    program_modifier_window_value = payload.get("program_modifier_window", DEFAULT_PROGRAM_MODIFIER_WINDOW)
    try:
        program_modifier_window_value = int(program_modifier_window_value)
        if program_modifier_window_value <= 0:
            program_modifier_window_value = DEFAULT_PROGRAM_MODIFIER_WINDOW
    except (TypeError, ValueError):
        program_modifier_window_value = DEFAULT_PROGRAM_MODIFIER_WINDOW

    default_signal_value = payload.get("default_program_signal", DEFAULT_PROGRAM_SIGNAL)
    try:
        default_signal_value = float(default_signal_value)
    except (TypeError, ValueError):
        default_signal_value = DEFAULT_PROGRAM_SIGNAL

    config_object: Dict[str, Any] = {
        "schools": cleaned_schools,
        "alias_index": _build_alias_index(cleaned_schools),
        "tier_base_scores": _coerce_tier_base_scores(payload.get("tier_base_scores")),
        "program_modifiers": cleaned_modifiers,
        "program_modifier_window": program_modifier_window_value,
        "default_program_signal": default_signal_value,
    }
    if path is None:
        _CACHED_CONFIG = config_object
    return config_object


def find_school_matches(
    text_lower: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return all (record, match_offset) for schools whose alias appears in
    `text_lower`.

    Each result is `{"record": <school_record>, "offset": int, "alias": str}`.
    `text_lower` MUST already be lowercased — callers in the scorer pre-lower
    once for cheaper matching across all dimensions.

    Word-boundary checks are intentionally lenient: many education_json blobs
    are noisy ("MIT, Cambridge MA, Class of 2018"). We only enforce that
    short aliases (<=4 chars, e.g. `cmu`, `mit`, `usc`) match at a word
    boundary — long aliases like `university of california, berkeley` match
    anywhere because false positives are ~0.
    """
    resolved_config = config if config is not None else load_schools_config()
    indexed_aliases = resolved_config.get("alias_index") or []
    if not indexed_aliases or not text_lower:
        return []

    matches: List[Dict[str, Any]] = []
    seen_school_ids = set()
    for alias_lower, school_record in indexed_aliases:
        if len(alias_lower) <= 4:
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias_lower) + r"(?![a-z0-9])")
            iterator = pattern.finditer(text_lower)
            offsets = [match.start() for match in iterator]
        else:
            offsets = []
            search_index = 0
            while True:
                found_index = text_lower.find(alias_lower, search_index)
                if found_index == -1:
                    break
                offsets.append(found_index)
                search_index = found_index + len(alias_lower)
        if not offsets:
            continue
        record_id = id(school_record)
        if record_id in seen_school_ids:
            continue
        seen_school_ids.add(record_id)
        matches.append(
            {
                "record": school_record,
                "offset": offsets[0],
                "alias": alias_lower,
            }
        )
    return matches


def program_modifier_at_offset(
    text_lower: str,
    school_offset: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """Return `(modifier, label)` for the strongest program-signal token in
    the window around `school_offset`.

    "Strongest" means: lowest modifier (most penalizing) wins for executive /
    online matches, and highest modifier wins among the positive ones (PhD
    beats Masters beats Bachelors). Concretely we pick the modifier whose
    distance from 1.0 is largest. Ties broken by first match in the list,
    which matches the JSON declaration order (executive first, doctoral
    last, etc.).

    No match -> `(default_program_signal, "")` so the caller can use the
    school's tier base score unchanged.
    """
    resolved_config = config if config is not None else load_schools_config()
    program_modifiers = resolved_config.get("program_modifiers") or []
    window_size = int(
        resolved_config.get("program_modifier_window") or DEFAULT_PROGRAM_MODIFIER_WINDOW
    )
    default_signal = float(
        resolved_config.get("default_program_signal") or DEFAULT_PROGRAM_SIGNAL
    )

    if not program_modifiers or not text_lower:
        return default_signal, ""

    window_start = max(0, school_offset - window_size)
    window_end = min(len(text_lower), school_offset + window_size)
    window_text = text_lower[window_start:window_end]

    best_modifier_value: Optional[float] = None
    best_modifier_label = ""
    for modifier_record in program_modifiers:
        modifier_value = float(modifier_record.get("modifier", 1.0))
        if any(
            token in window_text for token in modifier_record.get("tokens", [])
        ):
            if (
                best_modifier_value is None
                or abs(modifier_value - 1.0) > abs(best_modifier_value - 1.0)
            ):
                best_modifier_value = modifier_value
                best_modifier_label = str(modifier_record.get("label") or "")

    if best_modifier_value is None:
        return default_signal, ""
    return best_modifier_value, best_modifier_label


def best_school_score(
    text_lower: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Score each school match by `tier_base * program_modifier` and return
    the strongest one as a structured dict, or None when no school matches.

    Returned dict shape:
    ```
    {
        "name": str,
        "tier": int,
        "country": str,
        "alias": str,
        "offset": int,
        "tier_base": float,
        "program_modifier": float,
        "program_label": str,
        "score": float,
    }
    ```

    Multiple schools at the same effective score: the first one in
    `find_school_matches` order wins (alias-index sort is longest-first,
    which is generally the more specific match).
    """
    resolved_config = config if config is not None else load_schools_config()
    matches = find_school_matches(text_lower, config=resolved_config)
    if not matches:
        return None

    tier_base_scores = resolved_config.get("tier_base_scores") or DEFAULT_TIER_BASE_SCORES
    best_result: Optional[Dict[str, Any]] = None
    for match_record in matches:
        school_record = match_record["record"]
        school_tier = int(school_record.get("tier", 0))
        tier_base = float(tier_base_scores.get(school_tier, 0.0))
        modifier_value, modifier_label = program_modifier_at_offset(
            text_lower, match_record["offset"], config=resolved_config
        )
        scored_value = tier_base * modifier_value
        if best_result is None or scored_value > float(best_result["score"]):
            best_result = {
                "name": school_record.get("name", ""),
                "tier": school_tier,
                "country": school_record.get("country", ""),
                "alias": match_record["alias"],
                "offset": match_record["offset"],
                "tier_base": tier_base,
                "program_modifier": modifier_value,
                "program_label": modifier_label,
                "score": scored_value,
            }
    return best_result


def reset_cache() -> None:
    """Drop the in-memory cache. Tests use this to swap configs between cases."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None
