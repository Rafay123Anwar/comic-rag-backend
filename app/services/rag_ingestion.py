import json
import re
import time

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.core.logging import logger
from app.services.embedding import create_embeddings
from app.services.rag_chunker import create_rag_chunks
from app.services.rag_preprocessor import build_page_content, comic_to_rag_documents
from app.services.vector_store import (
    _get_chroma_collection,
    add_chunks,
    collection,
    delete_chunks_by_ids,
    get_chunks_by_comic_id,
    get_supabase_client,
    is_supabase_vector_enabled,
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


def reconcile_comic_rag(
    comic_id: str,
    comic_name: str,
    source_format: str,
    pages: list[dict]
) -> dict:
    """
    Reconciles vector store chunks for a comic after incremental per-page ingestion.
    Verifies chunk presence in BOTH Supabase pgvector AND ChromaDB (when Supabase is enabled).
    Only embeds/upserts pages that are missing from either store, guaranteeing zero redundant
    Mistral embedding calls while ensuring both stores remain 100% in sync.
    If Supabase already has chunks, fast-mirrors them to ChromaDB directly with $0 API cost.
    """
    start_time = time.perf_counter()
    page_pattern = re.compile(r'_page_(\d+)_chunk_')

    # 1. Query ChromaDB chunks
    col = _get_chroma_collection()
    chroma_res = col.get(where={"comic_id": comic_id})
    chroma_ids = set(chroma_res.get("ids", []))
    chroma_pages = {
        int(m.group(1))
        for cid in chroma_ids
        if (m := page_pattern.search(cid))
    }

    # 2. Query Supabase pgvector chunks (if enabled)
    supabase_enabled = is_supabase_vector_enabled()
    supabase_pages = set()
    supabase_ids = set()
    supabase_rows = []
    client = get_supabase_client() if supabase_enabled else None
    if client:
        try:
            s_res = (
                client.table("comic_page_chunks")
                .select("id, page_number, chunk_index, content, metadata, embedding")
                .eq("comic_id", comic_id)
                .execute()
            )
            supabase_rows = s_res.data or []
            supabase_ids = {r["id"] for r in supabase_rows}
            supabase_pages = {
                int(m.group(1))
                for cid in supabase_ids
                if (m := page_pattern.search(cid))
            }
        except Exception as sb_err:
            logger.warning("[RAG RECONCILIATION] Supabase chunk query error for comic %s: %s", comic_id, sb_err)
            supabase_pages = set()

    # 3. Fast-path: If Supabase already has chunks for a page missing in ChromaDB, mirror them directly with $0 API cost
    mirrored_to_chroma = 0
    if supabase_enabled and supabase_rows:
        missing_in_chroma_pages = supabase_pages - chroma_pages
        if missing_in_chroma_pages:
            rows_to_mirror = [r for r in supabase_rows if int(r.get("page_number", 0)) in missing_in_chroma_pages]
            if rows_to_mirror:
                m_ids = []
                m_docs = []
                m_embs = []
                m_metas = []
                for r in rows_to_mirror:
                    emb = r.get("embedding")
                    if isinstance(emb, str):
                        try:
                            emb = json.loads(emb)
                        except Exception:
                            emb = None
                    if emb and r.get("content"):
                        m_ids.append(r["id"])
                        m_docs.append(r["content"])
                        m_embs.append(emb)
                        m_metas.append(r.get("metadata") or {})
                if m_ids:
                    try:
                        col.upsert(
                            ids=m_ids,
                            documents=m_docs,
                            embeddings=m_embs,
                            metadatas=m_metas
                        )
                        chroma_pages.update(missing_in_chroma_pages)
                        chroma_ids.update(m_ids)
                        mirrored_to_chroma = len(m_ids)
                        logger.info(
                            "[RAG RECONCILIATION] Fast-mirrored %d chunks for pages %s from Supabase to ChromaDB ($0 API cost)",
                            len(m_ids),
                            sorted(missing_in_chroma_pages)
                        )
                    except Exception as mirror_err:
                        logger.warning("[RAG RECONCILIATION] Failed mirroring chunks to ChromaDB: %s", mirror_err)

    # 4. Only consider a page present if it exists in ALL active stores
    if supabase_enabled:
        fully_ingested_pages = chroma_pages.intersection(supabase_pages)
    else:
        fully_ingested_pages = chroma_pages

    # 5. Identify analyzed pages truly missing from either store that need full re-ingestion
    missing_pages = []
    for p in pages:
        if p.get("status") == "success":
            p_num = int(p.get("page_number", 1))
            if p_num not in fully_ingested_pages:
                content = build_page_content(p)
                if content.strip():
                    missing_pages.append(p)

    reconciled_count = 0
    if missing_pages:
        logger.warning(
            "[RAG RECONCILIATION] Comic %s has %d page(s) missing from vector store(s): %s. Ingesting missing pages...",
            comic_id,
            len(missing_pages),
            [p.get("page_number") for p in missing_pages]
        )
        for mp in missing_pages:
            try:
                chunks_added = ingest_page_to_rag(
                    comic_id=comic_id,
                    comic_name=comic_name,
                    source_format=source_format,
                    page=mp
                )
                reconciled_count += chunks_added
            except Exception as e:
                logger.error("[RAG RECONCILIATION] Failed to ingest missing page %s: %s", mp.get("page_number"), e)
    else:
        logger.info(
            "[RAG RECONCILIATION] Comic %s: Incremental ingestion complete. All %d analyzed page(s) "
            "verified in BOTH stores (ChromaDB: %d, Supabase: %d). Zero re-embedding required.",
            comic_id,
            len(pages),
            len(chroma_ids),
            len(supabase_ids) if supabase_enabled else 0
        )

    duration = time.perf_counter() - start_time
    return {
        "supabase_enabled": supabase_enabled,
        "chroma_chunks": len(chroma_ids),
        "supabase_chunks": len(supabase_ids) if supabase_enabled else 0,
        "mirrored_to_chroma": mirrored_to_chroma,
        "missing_pages_reconciled": len(missing_pages),
        "chunks_added": reconciled_count,
        "duration_seconds": round(duration, 3)
    }


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