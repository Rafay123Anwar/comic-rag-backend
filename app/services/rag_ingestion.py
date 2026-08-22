import json
import time

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.core.logging import logger
from app.services.embedding import create_embeddings
from app.services.rag_chunker import create_rag_chunks
from app.services.rag_preprocessor import build_page_content, comic_to_rag_documents
from app.services.vector_store import (
    add_chunks,
    collection,
    delete_chunks_by_ids,
    get_chunks_by_comic_id,
)


def ingest_page_to_rag(comic_id: str, comic_name: str, source_format: str, page: dict) -> int:
    """
    Ingests a single analyzed comic page into ChromaDB immediately upon page completion.
    Returns the number of chunks added.
    """
    page_number = page.get("page_number", 1) if page else None
    if not page or page.get("status") != "success":
        logger.warning("[RAG INGESTION] ingest_page_to_rag skipped for comic_id=%s page=%s (status=%s)", comic_id, page_number, page.get("status") if page else "None")
        print(f"[RAG INGESTION] comic_id={comic_id} page_number={page_number} status={page.get('status') if page else 'None'} -> 0 chunks upserted (skipped)")
        return 0

    content = build_page_content(page)
    if not content.strip():
        logger.warning("[RAG INGESTION] ingest_page_to_rag empty content for comic_id=%s page=%s", comic_id, page_number)
        print(f"[RAG INGESTION] comic_id={comic_id} page_number={page_number} -> 0 chunks upserted (empty content)")
        return 0

    doc = {
        "content": content,
        "metadata": {
            "comic_id": comic_id,
            "comic_name": comic_name,
            "source_format": source_format,
            "page_number": page_number,
            "filename": page.get("filename"),
            "image_path": page.get("image_path")
        }
    }

    chunks = create_rag_chunks([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        print(f"[RAG INGESTION] comic_id={comic_id} page_number={page_number} -> 0 chunks created")
        return 0

    texts = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    add_chunks(chunks, embeddings)
    logger.info("[RAG INGESTION] Ingested page %s (%d chunks) into ChromaDB for comic %s", page_number, len(chunks), comic_id)
    print(f"[RAG INGESTION] SUCCESS: comic_id={comic_id} page_number={page_number} -> {len(chunks)} chunks upserted into ChromaDB")
    return len(chunks)


def ingest_comic_to_rag(comic_json_path: str) -> dict:
    """
    Reads a comic.json file, creates chunks and embeddings, and persists them into ChromaDB.
    Guarantees idempotent re-ingestion and cleans up any stale chunks for the comic.
    """
    start_time = time.perf_counter()

    # -----------------------------
    # 1. Load JSON
    # -----------------------------
    t0 = time.perf_counter()
    with open(comic_json_path, "r", encoding="utf-8") as file:
        comic_data = json.load(file)
    t_json = time.perf_counter() - t0

    # -----------------------------
    # 2. Documents (Preprocessing)
    # -----------------------------
    t0 = time.perf_counter()
    documents = comic_to_rag_documents(comic_data)
    t_preprocess = time.perf_counter() - t0

    # -----------------------------
    # 3. Chunks (Chunking)
    # -----------------------------
    t0 = time.perf_counter()
    chunks = create_rag_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    t_chunk = time.perf_counter() - t0

    comic_id = comic_data.get("comic", {}).get("id")

    # -----------------------------
    # 4. Stale Chunk Lookup
    # -----------------------------
    t0 = time.perf_counter()
    existing_ids = get_chunks_by_comic_id(comic_id)
    new_ids = [chunk["chunk_id"] for chunk in chunks]
    stale_ids = list(set(existing_ids) - set(new_ids))
    t_stale_lookup = time.perf_counter() - t0

    if not chunks:
        total_time = time.perf_counter() - start_time
        logger.info(
            "\n"
            "[PERF] ==================================================\n"
            "[PERF] RAG INGESTION PERFORMANCE\n"
            "[PERF] Documents: %s\n"
            "[PERF] Chunks: 0\n"
            "[PERF] \n"
            "[PERF] JSON load:        %.2fs\n"
            "[PERF] Preprocessing:    %.2fs\n"
            "[PERF] Chunking:         %.2fs\n"
            "[PERF] Stale lookup:     %.2fs\n"
            "[PERF] Embeddings:       0.00s\n"
            "[PERF] Vector upsert:    0.00s\n"
            "[PERF] Stale deletion:   0.00s\n"
            "[PERF] TOTAL RAG:        %.2fs\n"
            "[PERF] ==================================================",
            len(documents),
            t_json,
            t_preprocess,
            t_chunk,
            t_stale_lookup,
            total_time
        )
        return {
            "documents": len(documents),
            "chunks": 0,
            "embedding_count": 0,
            "time_seconds": round(total_time, 2)
        }

    # -----------------------------
    # 5. Embeddings
    # -----------------------------
    t0 = time.perf_counter()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    t_embed = time.perf_counter() - t0

    # -----------------------------
    # 6. Save to Vector Store (Supabase pgvector / ChromaDB Upsert)
    # -----------------------------
    t0 = time.perf_counter()
    add_chunks(chunks, embeddings)
    t_upsert = time.perf_counter() - t0

    # -----------------------------
    # 7. Stale Deletion
    # -----------------------------
    t0 = time.perf_counter()
    if stale_ids:
        delete_chunks_by_ids(stale_ids)
    t_stale_delete = time.perf_counter() - t0

    total_time = time.perf_counter() - start_time

    logger.info(
        "\n"
        "[PERF] ==================================================\n"
        "[PERF] RAG INGESTION PERFORMANCE\n"
        "[PERF] Documents: %s\n"
        "[PERF] Chunks: %s\n"
        "[PERF] \n"
        "[PERF] JSON load:        %.2fs\n"
        "[PERF] Preprocessing:    %.2fs\n"
        "[PERF] Chunking:         %.2fs\n"
        "[PERF] Stale lookup:     %.2fs\n"
        "[PERF] Embeddings:       %.2fs\n"
        "[PERF] ChromaDB upsert:  %.2fs\n"
        "[PERF] Stale deletion:   %.2fs\n"
        "[PERF] TOTAL RAG:        %.2fs\n"
        "[PERF] ==================================================",
        len(documents),
        len(chunks),
        t_json,
        t_preprocess,
        t_chunk,
        t_stale_lookup,
        t_embed,
        t_upsert,
        t_stale_delete,
        total_time
    )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_count": len(embeddings),
        "time_seconds": round(total_time, 2)
    }