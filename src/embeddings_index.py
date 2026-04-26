"""In-process FAISS embedding index over LinkedIn profile text.

What it does
------------
Builds a FAISS inner-product index over `headline + about_text` for every
row in `linkedin_api_profiles_parsed`, using OpenAI's `text-embedding-3-small`.
`semantic_search(question_text, top_k)` returns the top-K `profile_id`s most
similar to the query. The retriever in `src/retriever.py` unions those ids
with the lexical search results so candidates who phrase their background
differently (e.g. "machine learning engineer" vs "ml engineer") still reach
the ranker.

Why it's in-process
-------------------
The demo dataset is small (~1k profiles). An in-process FAISS index keeps
the dependency surface minimal (no pgvector server, no Chroma container)
and makes the retrieval pipeline fully hermetic. The embeddings are cached
to `.cache/embeddings.npz` so subsequent runs load in ~50ms instead of
re-embedding.

Cache invalidation
------------------
Delete `.cache/embeddings.npz` to force a rebuild. The cache is also
rebuilt automatically when the profile count or the embedding-model name
no longer matches what was serialized.

Graceful degradation
--------------------
Any of the following makes `semantic_search` return `[]` rather than
fail: `faiss` not installed, `langchain-openai` not installed,
`OPENAI_API_KEY` missing, cache corrupt/rebuild-failed, or MySQL unreachable.
The retriever then falls back to lexical-only recall.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

try:
    import faiss  # type: ignore[import-not-found]
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _FAISS_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
    _EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _EMBEDDINGS_AVAILABLE = False

from .retriever import _open_connection

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "embeddings.npz"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small output dim
EMBED_BATCH_SIZE = 64
MAX_PROFILES = 2000  # cap to keep cold-build + API spend bounded
MAX_CHARS_PER_DOC = 4000  # avoid blowing the token budget on giant about_texts


def _embedding_text_from_profile(profile_row: Dict[str, Any]) -> str:
    """Concatenate the fields we embed. Truncate so the API call stays cheap."""
    headline_value = str(profile_row.get("headline") or "").strip()
    about_text_value = str(profile_row.get("about_text") or "").strip()
    combined_text = f"{headline_value}\n{about_text_value}".strip()
    if len(combined_text) > MAX_CHARS_PER_DOC:
        combined_text = combined_text[:MAX_CHARS_PER_DOC]
    return combined_text


def _is_available() -> bool:
    if not _FAISS_AVAILABLE or not _EMBEDDINGS_AVAILABLE:
        return False
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False
    return True


def _fetch_all_profile_rows(limit: int = MAX_PROFILES) -> List[Dict[str, Any]]:
    """Pull the fields we embed for every profile in the DB."""
    sql_query = """
    SELECT profile_id, headline, about_text
    FROM linkedin_api_profiles_parsed
    WHERE about_text IS NOT NULL AND CHAR_LENGTH(about_text) > 0
    LIMIT %s
    """
    with _open_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            database_cursor.execute(sql_query, (limit,))
            query_rows = database_cursor.fetchall()
    return [
        {
            "profile_id": row[0],
            "headline": row[1] or "",
            "about_text": row[2] or "",
        }
        for row in query_rows
    ]


def _embed_documents_in_batches(
    embedder: Any,
    documents: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Batched embedding call that returns a float32 matrix."""
    embedded_chunks: List[np.ndarray] = []
    for batch_start in range(0, len(documents), batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch_vectors = embedder.embed_documents(batch)
        embedded_chunks.append(np.asarray(batch_vectors, dtype=np.float32))
    if not embedded_chunks:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.vstack(embedded_chunks)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization so inner-product = cosine similarity."""
    if matrix.size == 0:
        return matrix
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    return (matrix / row_norms).astype(np.float32)


def _build_cache(
    cache_path: Path = DEFAULT_CACHE_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Pull all profiles, embed them, save to cache, return (vectors, ids).

    Returns None if the build fails for any reason; the retriever then
    continues with lexical-only recall.
    """
    if not _is_available():
        return None

    try:
        profile_rows = _fetch_all_profile_rows(limit=MAX_PROFILES)
    except Exception:  # noqa: BLE001 - DB failures degrade gracefully
        return None

    if not profile_rows:
        return None

    profile_ids: List[str] = [str(row["profile_id"]) for row in profile_rows]
    documents: List[str] = [_embedding_text_from_profile(row) for row in profile_rows]

    try:
        embedder = OpenAIEmbeddings(model=model_name)
        embedding_matrix = _embed_documents_in_batches(embedder, documents)
    except Exception:  # noqa: BLE001 - network/auth issues
        return None

    if embedding_matrix.shape[0] != len(profile_ids):
        return None

    normalized_matrix = _l2_normalize(embedding_matrix)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=normalized_matrix,
        profile_ids=np.array(profile_ids, dtype=object),
        model_name=np.array([model_name], dtype=object),
    )
    return normalized_matrix, profile_ids


def _load_cache(
    cache_path: Path = DEFAULT_CACHE_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Load (vectors, ids) from disk; return None on any mismatch/corruption."""
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=True)
        cached_embeddings = cached["embeddings"].astype(np.float32)
        cached_profile_ids = [str(pid) for pid in cached["profile_ids"].tolist()]
        cached_model_name = str(cached["model_name"].tolist()[0])
    except Exception:  # noqa: BLE001 - any load failure triggers rebuild
        return None

    if cached_model_name != model_name:
        return None
    if cached_embeddings.shape[0] != len(cached_profile_ids):
        return None
    if cached_embeddings.shape[1] != EMBEDDING_DIM:
        return None
    return cached_embeddings, cached_profile_ids


def _ensure_index(
    cache_path: Path = DEFAULT_CACHE_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Tuple[Any, List[str]]]:
    """Return (faiss_index, profile_ids) or None on any failure.

    Tries cache first; only builds from scratch when cache is missing/invalid.
    """
    if not _is_available():
        return None

    cached = _load_cache(cache_path=cache_path, model_name=model_name)
    if cached is None:
        cached = _build_cache(cache_path=cache_path, model_name=model_name)
    if cached is None:
        return None

    cached_embeddings, cached_profile_ids = cached
    try:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_index.add(cached_embeddings)
    except Exception:  # noqa: BLE001
        return None
    return faiss_index, cached_profile_ids


def semantic_search(
    question_text: str,
    top_k: int = 10,
    cache_path: Path = DEFAULT_CACHE_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[str]:
    """Return top-K `profile_id`s ordered by cosine similarity to `question_text`.

    Returns `[]` when the index cannot be constructed or queried for any
    reason so callers can union with lexical recall safely.
    """
    cleaned_query = (question_text or "").strip()
    if not cleaned_query or top_k <= 0:
        return []

    ensured = _ensure_index(cache_path=cache_path, model_name=model_name)
    if ensured is None:
        return []

    faiss_index, profile_ids = ensured

    try:
        embedder = OpenAIEmbeddings(model=model_name)
        query_vector = np.asarray(
            embedder.embed_query(cleaned_query), dtype=np.float32
        ).reshape(1, -1)
    except Exception:  # noqa: BLE001 - network/auth issues
        return []

    query_vector = _l2_normalize(query_vector)
    retrieval_depth = min(top_k, len(profile_ids))
    if retrieval_depth <= 0:
        return []

    try:
        _distances, nearest_indices = faiss_index.search(query_vector, retrieval_depth)
    except Exception:  # noqa: BLE001
        return []

    return [
        profile_ids[int(idx)]
        for idx in nearest_indices[0]
        if 0 <= int(idx) < len(profile_ids)
    ]


def build_index(force: bool = False) -> bool:
    """Eager CLI-friendly hook: force (re)building the cache.

    Returns True on success, False on any failure or missing dependency.
    """
    if not _is_available():
        return False
    if force and DEFAULT_CACHE_PATH.exists():
        try:
            DEFAULT_CACHE_PATH.unlink()
        except OSError:
            return False
    built = _build_cache()
    return built is not None
