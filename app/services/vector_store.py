"""
Vector Store Service

Manages vector database operations for Comic-RAG using Supabase (pgvector)
with automatic fallback to local ChromaDB.

Uses lazy initialization to ensure fast application startup times.
"""
import logging
from typing import Optional

from app.core.config import (
    DEFAULT_TOP_K,
    VECTOR_DB_DIR,
    VECTOR_STORE_BACKEND,
)
from app.core.supabase import get_supabase_client

logger = logging.getLogger("comic_rag")

COLLECTION_NAME = "comic_pages"

# -----------------------------
# ChromaDB (Lazy Initialization / Fallback)
# -----------------------------

_chroma_client = None
_chroma_collection = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )
    return _chroma_client


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = _get_chroma_client()
        _chroma_collection = client.get_or_create_collection(
            name="comic_pages",
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


def get_collection():
    """Returns the local ChromaDB collection instance."""
    return _get_chroma_collection()


# -----------------------------
# Backend Detection & Diagnostics
# -----------------------------

def is_supabase_vector_enabled() -> bool:
    """
    Checks if Supabase vector backend is enabled and active.
    """
    if VECTOR_STORE_BACKEND == "chroma":
        return False

    client = get_supabase_client()
    if not client:
        return False

    try:
        # Quick ping to comic_page_chunks table
        res = client.table("comic_page_chunks").select("id").limit(1).execute()
        return res is not None
    except Exception as e:
        logger.debug("[VECTOR_STORE] Supabase comic_page_chunks table check: %s", str(e))
        return False


def get_active_backend_name() -> str:
    """Returns 'supabase' or 'chroma' based on current active backend."""
    if is_supabase_vector_enabled():
        return "supabase"
    return "chroma"


class _LazyCollectionProxy:
    """
    Transparent proxy that defers ChromaDB collection access until first method invocation.
    Maintains full backward compatibility.
    """
    def __getattr__(self, name):
        return getattr(_get_chroma_collection(), name)


collection = _LazyCollectionProxy()


def _to_jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


# -----------------------------
# Add / Upsert Chunks
# -----------------------------

def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]]
) -> None:
    """
    Upserts chunk documents and their embeddings into Supabase pgvector or ChromaDB.
    """
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must be the same."
        )

    # 1. Try Supabase pgvector first if enabled
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    supabase_saved = False

    if supabase_client:
        try:
            records = []
            for chunk, emb in zip(chunks, embeddings):
                meta = chunk.get("metadata", {}) or {}
                clean_meta = _to_jsonable(meta) if isinstance(meta, dict) else {}
                comic_id = str(clean_meta.get("comic_id", "")).strip()
                page_number = int(clean_meta.get("page_number", 1))
                chunk_index = int(clean_meta.get("chunk_index", 0))
                clean_emb = _to_jsonable(emb) if emb is not None else None

                records.append({
                    "id": chunk["chunk_id"],
                    "comic_id": comic_id,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "content": chunk["content"],
                    "metadata": clean_meta,
                    "embedding": clean_emb,
                })

            if records:
                # Upsert in batches of 100 to avoid payload limits
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    supabase_client.table("comic_page_chunks").upsert(batch).execute()

                supabase_saved = True
                logger.info(
                    "[VECTOR_STORE] Upserted %d chunks to Supabase pgvector (comic_page_chunks)",
                    len(records)
                )
        except Exception as e:
            logger.warning(
                "[VECTOR_STORE] Supabase upsert failed, falling back to local ChromaDB: %s",
                str(e)
            )

    # 2. Also persist to / fallback to ChromaDB
    try:
        col = _get_chroma_collection()
        col.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
    except Exception as e:
        if not supabase_saved:
            raise e
        logger.warning("[VECTOR_STORE] ChromaDB local mirror upsert warning: %s", str(e))


# -----------------------------
# Vector Search
# -----------------------------

def search_chunks(
    query_embedding: list[float],
    comic_id: str,
    top_k: int = DEFAULT_TOP_K
) -> dict:
    """
    Searches the vector store for chunks matching the query embedding within a specific comic.
    Uses cosine distance metric (similarity = 1 - cosine_distance).
    Returns Chroma-compatible dictionary format: {ids: [[]], documents: [[]], metadatas: [[]], distances: [[]]}.
    """
    if not query_embedding:
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    if not comic_id or not comic_id.strip():
        raise ValueError(
            "comic_id is required for comic-scoped search."
        )

    cleaned_id = comic_id.strip()

    # 1. Try Supabase pgvector RPC
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            res = supabase_client.rpc(
                "match_comic_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 1.0,
                    "match_count": int(top_k),
                    "p_comic_id": cleaned_id,
                }
            ).execute()

            rows = res.data if res and hasattr(res, "data") else []
            if rows is not None:
                ids = [r["id"] for r in rows]
                documents = [r.get("content", "") for r in rows]
                metadatas = [r.get("metadata", {}) for r in rows]
                distances = [float(r.get("distance", 0.0)) for r in rows]

                logger.debug(
                    "[VECTOR_STORE] Supabase search returned %d chunks for comic %s",
                    len(ids),
                    cleaned_id
                )
                return {
                    "ids": [ids],
                    "documents": [documents],
                    "metadatas": [metadatas],
                    "distances": [distances]
                }
        except Exception as e:
            logger.warning(
                "[VECTOR_STORE] Supabase match_comic_chunks failed, falling back to ChromaDB: %s",
                str(e)
            )

    # 2. Fallback to ChromaDB
    col = _get_chroma_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={
            "comic_id": cleaned_id
        }
    )
    return results


# -----------------------------
# Chunks by Page & Comic
# -----------------------------

def get_chunks_by_page(comic_id: str, page_number: int) -> list[dict]:
    """
    Retrieves all chunks belonging to a specific page of a comic.
    """
    if not comic_id or not comic_id.strip() or not page_number:
        return []

    cleaned_id = comic_id.strip()

    # 1. Try Supabase
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            res = (
                supabase_client.table("comic_page_chunks")
                .select("id, content, metadata, page_number, chunk_index")
                .eq("comic_id", cleaned_id)
                .eq("page_number", int(page_number))
                .order("chunk_index")
                .execute()
            )
            rows = res.data if res and hasattr(res, "data") else []
            if rows is not None and len(rows) > 0:
                chunks = []
                for r in rows:
                    chunks.append({
                        "chunk_id": r["id"],
                        "content": r.get("content", ""),
                        "metadata": r.get("metadata", {}) or {},
                        "distance": 0.0
                    })
                return chunks
        except Exception as e:
            logger.debug("[VECTOR_STORE] Supabase get_chunks_by_page lookup error: %s", str(e))

    # 2. Fallback to ChromaDB
    col = _get_chroma_collection()
    results = col.get(
        where={
            "$and": [
                {"comic_id": cleaned_id},
                {"page_number": int(page_number)}
            ]
        }
    )

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    chunks = []
    for i, chunk_id in enumerate(ids):
        chunks.append({
            "chunk_id": chunk_id,
            "content": documents[i] if i < len(documents) else "",
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": 0.0
        })

    chunks.sort(key=lambda c: (
        c["metadata"].get("page_number", 0),
        c["metadata"].get("chunk_index", 0),
        c["metadata"].get("chunk_id", "")
    ))
    return chunks


def get_chunks_by_comic_id(comic_id: str) -> list[str]:
    """
    Returns all chunk IDs for a given comic_id.
    """
    if not comic_id or not comic_id.strip():
        return []

    cleaned_id = comic_id.strip()

    # 1. Supabase check
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            res = (
                supabase_client.table("comic_page_chunks")
                .select("id")
                .eq("comic_id", cleaned_id)
                .execute()
            )
            rows = res.data if res and hasattr(res, "data") else []
            if rows is not None and len(rows) > 0:
                return [r["id"] for r in rows]
        except Exception as e:
            logger.debug("[VECTOR_STORE] Supabase get_chunks_by_comic_id error: %s", str(e))

    # 2. ChromaDB lookup
    col = _get_chroma_collection()
    existing = col.get(where={"comic_id": cleaned_id})
    return existing.get("ids", [])


def delete_chunks_by_ids(chunk_ids: list[str]) -> int:
    """
    Deletes chunks by their IDs from both Supabase and ChromaDB.
    """
    if not chunk_ids:
        return 0

    count = 0
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            supabase_client.table("comic_page_chunks").delete().in_("id", chunk_ids).execute()
            count = len(chunk_ids)
        except Exception as e:
            logger.warning("[VECTOR_STORE] Supabase delete_chunks_by_ids error: %s", str(e))

    try:
        col = _get_chroma_collection()
        col.delete(ids=chunk_ids)
        if count == 0:
            count = len(chunk_ids)
    except Exception as e:
        logger.warning("[VECTOR_STORE] ChromaDB delete_chunks_by_ids error: %s", str(e))

    return count


def delete_chunks_by_comic_id(comic_id: str) -> int:
    """
    Deletes all chunks belonging to the specified comic_id from Supabase and ChromaDB.
    """
    if not comic_id or not comic_id.strip():
        return 0

    cleaned_id = comic_id.strip()
    total_deleted = 0

    # 1. Delete from Supabase
    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            res = (
                supabase_client.table("comic_page_chunks")
                .delete()
                .eq("comic_id", cleaned_id)
                .execute()
            )
            deleted_rows = res.data if res and hasattr(res, "data") else []
            total_deleted = len(deleted_rows)
            logger.info("[VECTOR_STORE] Deleted %d chunks from Supabase for comic %s", total_deleted, cleaned_id)
        except Exception as e:
            logger.warning("[VECTOR_STORE] Supabase deletion error for %s: %s", cleaned_id, str(e))

    # 2. Delete from ChromaDB
    try:
        col = _get_chroma_collection()
        existing = col.get(where={"comic_id": cleaned_id})
        ids = existing.get("ids", [])
        if ids:
            col.delete(ids=ids)
            if total_deleted == 0:
                total_deleted = len(ids)
            logger.info("[VECTOR_STORE] Deleted %d chunks from ChromaDB for comic %s", len(ids), cleaned_id)
    except Exception as e:
        logger.warning("[VECTOR_STORE] ChromaDB deletion error for %s: %s", cleaned_id, str(e))

    return total_deleted


def reset_collection():
    """
    Resets the ChromaDB collection and Supabase table for testing/migration purposes.
    """
    global _chroma_collection

    supabase_client = get_supabase_client() if VECTOR_STORE_BACKEND != "chroma" else None
    if supabase_client:
        try:
            supabase_client.table("comic_page_chunks").delete().neq("id", "___nonexistent___").execute()
        except Exception as e:
            logger.warning("[VECTOR_STORE] Supabase reset error: %s", str(e))

    client = _get_chroma_client()
    try:
        client.delete_collection(name="comic_pages")
    except Exception:
        pass
    _chroma_collection = client.get_or_create_collection(
        name="comic_pages",
        metadata={"hnsw:space": "cosine"}
    )
    return _chroma_collection