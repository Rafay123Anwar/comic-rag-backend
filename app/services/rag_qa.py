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
from app.services.llm import generate_answer, reformulate_query
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
    r"\b(tell\s+me\s+about\s+this\s+page|describe\s+this\s+page)\b|"
    r"\b(who\s+appears\s+on\s+this|who\s+is\s+on\s+this\s+page)\b",
    re.IGNORECASE
)

COMIC_WIDE_INDICATORS = [
    "this comic", "is comic", "iss comic", "ye comic", "yeh comic", "the comic",
    "whole comic", "entire comic", "poori comic", "complete comic", "all pages",
    "story kya hai", "kya story hai", "story batao", "kahani kya hai", "plot",
    "overall story", "what is the story", "tell me the story", "summary of comic",
    "what is this comic about", "what happened in the comic", "comic ki story",
    "story of the comic", "story of this comic", "short story", "story", "kahani",
    "batao story", "story btao", "story btaao", "story iya hai", "story bta do"
]


def is_comic_wide_story_query(question: str) -> bool:
    """Check if the question is inquiring about the overall comic, story arc, or summary."""
    q = question.lower()
    return any(ind in q for ind in COMIC_WIDE_INDICATORS)


def is_page_scoped_query(question: str, current_page: int | None) -> bool:
    """Determine if a question should be grounded strictly in the active page."""
    if not current_page:
        return False
    q = question.lower()
    has_page_keyword = any(p in q for p in [
        "this page", "is page pe", "is page par", "ye page pe", "yeh page pe",
        "on this page", "in this page", "this panel", "this scene", "who appears on this page",
        "who appears here"
    ])
    # If the user explicitly asks about the comic story or whole comic, do not page-scope unless explicit page keyword is present
    if is_comic_wide_story_query(question) and not has_page_keyword:
        return False
    if PAGE_SCOPED_REGEX.search(question) and not any(cw in q for cw in ["this comic", "is comic", "iss comic", "the comic", "whole comic", "entire comic", "poori comic"]):
        return True
    return has_page_keyword


def get_comic_overview_chunks(comic_id: str) -> list[dict]:
    """
    Builds structured story chunks across all analyzed pages of the comic,
    ensuring Mistral AI has the complete story arc (beginning, middle, climax/end).
    """
    from app.services.storage import get_comic_data
    cdata = get_comic_data(comic_id)
    if not cdata:
        return []
    pages = cdata.get("pages", [])
    if not pages:
        return []

    chunks = []
    for p in sorted(pages, key=lambda x: x.get("page_number", 0)):
        pnum = p.get("page_number", 1)
        content = build_page_content(p)
        if content.strip():
            chunks.append({
                "chunk_id": f"{comic_id}_page_{pnum}_overview",
                "content": content,
                "metadata": {
                    "comic_id": comic_id,
                    "page_number": pnum,
                    "chunk_index": 1
                },
                "distance": 0.0
            })
    return chunks


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
    current_page: int | None = None,
    standalone_query: str | None = None
) -> dict:
    """
    Answers a question grounded in the specified comic context, utilizing
    smart query reformulation for coreference resolution and current_page for page-scoped grounding.
    """
    if not comic_id or not comic_id.strip():
        raise ValueError(
            "comic_id is required for comic-scoped question answering."
        )

    comic_id = comic_id.strip()

    # -----------------------------
    # 0. Active Page Information (Non-blocking)
    # -----------------------------
    page_obj = None
    if current_page is not None:
        try:
            _, page_obj = get_page_info(comic_id, current_page)
        except Exception:
            page_obj = None

    # -----------------------------
    # 1. Smart Query Reformulation (Coreference Resolution)
    # -----------------------------
    original_question = question
    if standalone_query is not None and standalone_query.strip():
        effective_query = standalone_query.strip()
    elif conversation_history:
        lines = []
        for msg in conversation_history[-5:]:
            r = "User" if msg.get("role") == "user" else "Assistant"
            c = msg.get("content", "").strip()
            if c:
                lines.append(f"{r}: {c}")
        h_str = "\n".join(lines)
        effective_query = reformulate_query(query=question, chat_history=h_str) if h_str else question
    else:
        effective_query = question

    # -----------------------------
    # 2. Normalize Query & Retrieval Query
    # -----------------------------
    normalized_question = normalize_query(effective_query)
    if not normalized_question:
        return {
            "comic_id": comic_id,
            "question": original_question,
            "standalone_query": effective_query,
            "answer": "Please provide a valid question.",
            "sources": []
        }

    retrieval_query = normalized_question

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
        # If user is asking about the comic story / overall plot / summary, prioritize full comic overview
        is_story_query = is_comic_wide_story_query(effective_query) or is_comic_wide_story_query(original_question)
        if is_story_query:
            overview_chunks = get_comic_overview_chunks(comic_id)
            for chunk in overview_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)

        # Retrieve semantic chunks across comic using standalone reformulated query
        semantic_chunks = retrieve_chunks(
            query=retrieval_query,
            comic_id=comic_id,
            top_k=top_k,
            distance_threshold=distance_threshold
        )
        for chunk in semantic_chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(chunk)

        # Fallback to overview chunks if no semantic chunks found for global query
        if not chunks:
            overview_chunks = get_comic_overview_chunks(comic_id)
            for chunk in overview_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)

        if page_chunks and not is_story_query:
            for chunk in page_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)
        print(f"[RAG QA DEBUG] Global query '{effective_query}' -> {len(chunks)} chunks retrieved")

    # -----------------------------
    # 4. No Results Fallback (Real-Time Aware)
    # -----------------------------
    if not chunks:
        is_processing = False
        try:
            from app.models.comic import Comic
            from app.core.database import SessionLocal
            _db = SessionLocal()
            try:
                c_row = _db.query(Comic).filter(Comic.id == comic_id).first()
                if c_row and c_row.status == "processing":
                    is_processing = True
            finally:
                _db.close()
        except Exception:
            pass

        fallback_msg = (
            "This comic is currently being analyzed in real-time. No pages matching your question have finished transcription yet. Please try asking again shortly as pages complete!"
            if is_processing
            else FALLBACK_ANSWER
        )

        return {
            "comic_id": comic_id,
            "question": original_question,
            "standalone_query": effective_query,
            "answer": fallback_msg,
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
    # 6. Generate Answer with LLM (Pass standalone_query)
    # -----------------------------
    if current_page is not None:
        raw_answer = generate_answer(
            question=effective_query,
            context=context,
            conversation_history=conversation_history,
            current_page=current_page
        )
    else:
        raw_answer = generate_answer(
            question=effective_query,
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
        "standalone_query": effective_query,
        "answer": answer,
        "sources": sources
    }