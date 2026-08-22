"""
RAG QA Service

Orchestrates query normalization, vector retrieval, deterministic context assembly,
LLM answer generation with conversation memory, and answer/source consistency validation.
"""
import json
import re
from pathlib import Path

from app.core.config import COMICS_DIR, DEFAULT_TOP_K
from app.core.logging import logger
from app.services.llm import generate_answer
from app.services.query_normalizer import normalize_query
from app.services.rag_preprocessor import build_page_content
from app.services.retriever import (
    DEFAULT_DISTANCE_THRESHOLD,
    retrieve_chunks,
)
from app.services.vector_store import get_chunks_by_page

FALLBACK_ANSWER = "I could not find relevant information in the comic."
PROCESSING_PAGE_ANSWER = "This page is currently being analyzed. Please wait a moment..."

FOLLOW_UP_INDICATORS = {
    "he", "she", "it", "they", "him", "her", "his", "their", "them",
    "usne", "uske", "uski", "uska", "unhone", "unka", "unki", "unke",
    "who", "why", "what", "where", "how", "next", "then", "that", "this", "also", "did"
}

PAGE_SCOPED_REGEX = re.compile(
    r"\b(this|is|iss|ye|yeh|current)\s+(page|scene|panel|pic|picture)\b|"
    r"\b(is|iss|ye|yeh)\s+page\b|"
    r"\bpage\s*\d+\b|"
    r"\b(what('s|\s+is)?\s+happening|what\s+happened|what\s+happend)\b|"
    r"\b(kya\s+ho\s+raha\s+hai|kya\s+hua)\b|"
    r"\b(here|yahan|idhar)\b|"
    r"\b(tell\s+me\s+about\s+this)\b|"
    r"\b(who\s+appears|story\s+of\s+this|summary\s+of\s+this|describe\s+this)\b",
    re.IGNORECASE
)


def is_page_scoped_query(question: str, current_page: int | None) -> bool:
    """Determine if a question should be grounded strictly in the active page."""
    if not current_page:
        return False
    if PAGE_SCOPED_REGEX.search(question):
        return True
    q = question.lower()
    page_indicators = [
        "this page", "is page", "ye page", "yeh page", "current page",
        "in this", "on this", "about this", "this scene", "this panel",
        "here", "yahan", "idhar", "happend", "happened", "happening",
        "kya hua", "kya ho raha", "who appears", "story of", "summary of"
    ]
    return any(ind in q for ind in page_indicators)


def get_page_info(comic_id: str, page_number: int | None) -> tuple[str | None, dict | None]:
    """
    Reads comic.json to get the status ('processing', 'success', 'error')
    and the raw page data dict for the specified page_number.
    """
    if not comic_id or page_number is None:
        return None, None

    comic_json_path = Path(COMICS_DIR) / comic_id / "comic.json"
    if not comic_json_path.exists():
        return None, None

    try:
        with open(comic_json_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        pages = cdata.get("pages", [])
        for p in pages:
            if p.get("page_number") == page_number:
                return p.get("status", "processing"), p
        comic_meta = cdata.get("comic", {})
        return comic_meta.get("status", "processing"), None
    except Exception:
        return None, None


def validate_answer_and_sources(
    answer: str,
    sources: list[dict],
    comic_id: str,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD
) -> tuple[str, list[dict]]:
    """
    Validate answer and sources consistency after LLM generation.
    - If answer is empty or whitespace-only, fallback and drop sources.
    - If answer is exact fallback phrase, drop sources.
    - Filter any sources that exceed distance_threshold or have mismatched comic_id.
    - Preserve existing source ordering and metadata fields.
    """
    if not answer or not answer.strip():
        return FALLBACK_ANSWER, []

    trimmed_answer = answer.strip()
    if trimmed_answer == FALLBACK_ANSWER or trimmed_answer == PROCESSING_PAGE_ANSWER:
        return trimmed_answer, []

    max_dist = distance_threshold if distance_threshold is not None else DEFAULT_DISTANCE_THRESHOLD

    validated_sources = [
        s for s in sources
        if s.get("comic_id") == comic_id
        and s.get("distance") is not None
        and s["distance"] <= max_dist
    ]

    return trimmed_answer, validated_sources


def answer_question(
    question: str,
    comic_id: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
) -> dict:
    """
    Answers a question grounded in the specified comic context, optionally utilizing
    conversation history for reference resolution and current_page for page-scoped grounding.
    """
    if not comic_id or not comic_id.strip():
        raise ValueError(
            "comic_id is required for comic-scoped question answering."
        )

    comic_id = comic_id.strip()

    # -----------------------------
    # 0. Active Page Processing Check
    # -----------------------------
    page_obj = None
    if current_page is not None:
        page_status, page_obj = get_page_info(comic_id, current_page)
        if page_status == "processing":
            logger.info("[RAG QA] Page %s is currently processing for comic %s. Returning wait message.", current_page, comic_id)
            return {
                "comic_id": comic_id,
                "question": question,
                "answer": PROCESSING_PAGE_ANSWER,
                "sources": []
            }
        elif page_status == "error":
            return {
                "comic_id": comic_id,
                "question": question,
                "answer": "This page could not be analyzed due to an error.",
                "sources": []
            }

    # -----------------------------
    # 1. Normalize Query
    # -----------------------------
    original_question = question
    normalized_question = normalize_query(question)
    if not normalized_question:
        return {
            "comic_id": comic_id,
            "question": original_question,
            "answer": "Please provide a valid question.",
            "sources": []
        }

    # -----------------------------
    # 2. Build Retrieval Query
    # -----------------------------
    retrieval_query = normalized_question
    if conversation_history:
        last_user_query = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "user" and msg.get("content", "").strip():
                last_user_query = normalize_query(msg["content"])
                break

        if last_user_query:
            words = set(re.findall(r"\w+", normalized_question.lower()))
            if words.intersection(FOLLOW_UP_INDICATORS) or len(words) <= 4:
                retrieval_query = f"{last_user_query} {normalized_question}"

    # -----------------------------
    # 3. Retrieve Chunks (Semantic + Page-Scoped)
    # -----------------------------
    is_page_scoped = is_page_scoped_query(question, current_page)
    page_chunks = get_chunks_by_page(comic_id, current_page) if current_page else []

    # Fallback to direct comic.json page content if ChromaDB has not finished embedding page chunks
    if current_page and not page_chunks and page_obj and page_obj.get("status") == "success":
        content = build_page_content(page_obj)
        if content.strip():
            page_chunks = [{
                "chunk_id": f"{comic_id}_page_{current_page}_chunk_1",
                "content": content,
                "metadata": {
                    "comic_id": comic_id,
                    "page_number": current_page,
                    "chunk_index": 1
                },
                "distance": 0.0
            }]

    seen_ids = set()
    chunks = []

    if is_page_scoped:
        # Strictly scope context to active page chunks for page-specific questions
        for chunk in page_chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(chunk)
        print(f"[RAG QA DEBUG] Page-scoped query '{question}' on page {current_page} -> {len(chunks)} page chunks retrieved")
        if not chunks:
            semantic_chunks = retrieve_chunks(
                query=normalized_question,
                comic_id=comic_id,
                top_k=top_k,
                distance_threshold=distance_threshold
            )
            for chunk in semantic_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)
    else:
        # Global query: retrieve semantic chunks across comic and supplement with current page
        semantic_chunks = retrieve_chunks(
            query=retrieval_query,
            comic_id=comic_id,
            top_k=top_k,
            distance_threshold=distance_threshold
        )
        if not semantic_chunks and retrieval_query != normalized_question:
            semantic_chunks = retrieve_chunks(
                query=normalized_question,
                comic_id=comic_id,
                top_k=top_k,
                distance_threshold=distance_threshold
            )
        for chunk in semantic_chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(chunk)
        if page_chunks:
            for chunk in page_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)
        print(f"[RAG QA DEBUG] Global query '{question}' -> {len(chunks)} chunks retrieved")

    # -----------------------------
    # 4. No Results Fallback
    # -----------------------------
    if not chunks:
        return {
            "comic_id": comic_id,
            "question": original_question,
            "answer": FALLBACK_ANSWER,
            "sources": []
        }

    # -----------------------------
    # 5. Build Context
    # -----------------------------
    context_parts = []
    sources = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        page_number = metadata.get("page_number")
        content = chunk.get("content", "")
        chunk_index = metadata.get("chunk_index")
        header = f"[PAGE {page_number} | CHUNK {chunk_index}]"

        context_parts.append(f"{header}\n{content}")
        sources.append(
            {
                "comic_id": metadata.get("comic_id", comic_id),
                "page_number": page_number,
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": chunk_index,
                "distance": chunk.get("distance", 0.0)
            }
        )

    context = "\n\n--------------------\n\n".join(context_parts)

    # -----------------------------
    # 6. Generate Answer with LLM
    # -----------------------------
    if current_page is not None:
        raw_answer = generate_answer(
            question=original_question,
            context=context,
            conversation_history=conversation_history,
            current_page=current_page
        )
    else:
        raw_answer = generate_answer(
            question=original_question,
            context=context,
            conversation_history=conversation_history
        )

    # -----------------------------
    # 7. Validate Answer & Sources
    # -----------------------------
    answer, sources = validate_answer_and_sources(
        answer=raw_answer,
        sources=sources,
        comic_id=comic_id,
        distance_threshold=distance_threshold
    )

    return {
        "comic_id": comic_id,
        "question": original_question,
        "answer": answer,
        "sources": sources
    }