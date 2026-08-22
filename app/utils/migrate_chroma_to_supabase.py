"""
Migration Utility: Local ChromaDB -> Supabase pgvector

Reads all collections and comic chunks from local ChromaDB and uploads
them into the Supabase 'comic_page_chunks' table.

Converts numpy ndarrays and scalars to JSON-serializable Python native types,
validates embedding dimensionality, and reports total row count from Supabase.
"""
import logging
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

# Configure utf-8 encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import VECTOR_DB_DIR
from app.core.supabase import get_supabase_client
from app.services.vector_store import (
    _get_chroma_collection,
    is_supabase_vector_enabled,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("chroma_to_supabase_migration")


def to_jsonable(value):
    """
    Recursively converts numpy ndarrays, numpy scalars, dicts, and lists
    into standard Python JSON-serializable types.
    """
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def migrate():
    logger.info("==========================================================")
    logger.info("STARTING CHROMADB -> SUPABASE PGVECTOR MIGRATION")
    logger.info("==========================================================")

    client = get_supabase_client()
    if not client:
        logger.error("[ERROR] Supabase client is not configured. Please check SUPABASE_URL and keys in .env.")
        return False

    # Check if target table is accessible
    if not is_supabase_vector_enabled():
        logger.warning(
            "[WARNING] Supabase 'comic_page_chunks' table is not accessible yet.\n"
            "Please ensure you executed 'supabase_pgvector_setup.sql' in your Supabase SQL Editor first!"
        )

    # Load local Chroma collection
    logger.info("Reading local ChromaDB from: %s", VECTOR_DB_DIR)
    try:
        col = _get_chroma_collection()
        data = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        logger.error("[ERROR] Failed to read local ChromaDB: %s", str(e))
        return False

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = data.get("embeddings", [])

    total_chunks = len(ids)
    logger.info("Found %d total chunk(s) in local ChromaDB.", total_chunks)

    if total_chunks == 0:
        logger.info("No chunks found in local ChromaDB to migrate.")
        return True

    has_embeddings = embeddings is not None and len(embeddings) == total_chunks

    records = []
    failed_preparation = 0

    for i, chunk_id in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        emb = embeddings[i] if has_embeddings else None

        # Convert embedding to native Python list[float]
        emb_list = to_jsonable(emb) if emb is not None else None

        if emb_list is not None:
            if not isinstance(emb_list, list):
                logger.error("[ERROR] Chunk %s embedding is not a list: %s", chunk_id, type(emb_list))
                failed_preparation += 1
                continue
            if len(emb_list) != 1024:
                logger.error("[ERROR] Invalid embedding dimension for chunk %s: %d (expected 1024)", chunk_id, len(emb_list))
                failed_preparation += 1
                continue

        clean_meta = to_jsonable(meta) if isinstance(meta, dict) else {}
        comic_id = str(clean_meta.get("comic_id", "")).strip()
        page_number = int(clean_meta.get("page_number", 1))
        chunk_index = int(clean_meta.get("chunk_index", 0))

        records.append({
            "id": chunk_id,
            "comic_id": comic_id,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content": doc,
            "metadata": clean_meta,
            "embedding": emb_list,
        })

    logger.info("Prepared %d valid record(s) for upload (Preparation failures: %d).", len(records), failed_preparation)

    # Batch upsert to Supabase
    batch_size = 50
    success_count = 0
    failed_upload_count = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            client.table("comic_page_chunks").upsert(batch).execute()
            success_count += len(batch)
            logger.info("Uploaded %d / %d chunks to Supabase...", success_count, len(records))
        except Exception as e:
            failed_upload_count += len(batch)
            logger.error("[ERROR] Batch upload failed at index %d: %s", i, str(e))

    # Query total row count in Supabase table
    supabase_row_count = "Unknown"
    try:
        count_res = client.table("comic_page_chunks").select("id", count="exact").execute()
        if hasattr(count_res, "count") and count_res.count is not None:
            supabase_row_count = count_res.count
        elif hasattr(count_res, "data") and count_res.data is not None:
            supabase_row_count = len(count_res.data)
    except Exception as e:
        logger.warning("[WARNING] Could not retrieve exact Supabase row count: %s", str(e))

    logger.info("==========================================================")
    logger.info("MIGRATION SUMMARY:")
    logger.info("  • Total local ChromaDB chunks:  %d", total_chunks)
    logger.info("  • Successfully migrated:        %d", success_count)
    logger.info("  • Failed count:                 %d", failed_preparation + failed_upload_count)
    logger.info("  • Supabase Total Row Count:     %s", str(supabase_row_count))
    logger.info("==========================================================")

    return success_count == total_chunks


if __name__ == "__main__":
    migrate()
