"""
Conversation Routes

API endpoints for conversation lifecycle management and context-aware chat.
"""
import json
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_TOP_K, MAX_CONVERSATION_MESSAGES
from app.core.database import get_db
from app.models.user import User
from app.routes.comic import check_comic_access, validate_comic_id
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationQuestionRequest,
    ConversationQuestionResponse,
    ConversationResponse,
)
from app.services.auth import get_current_active_user
from app.services.conversation import (
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversations_by_comic_id,
    get_messages,
    validate_conversation_id,
)
from app.services.llm import clean_llm_response, stream_generate_answer_async
from app.services.query_normalizer import normalize_query
from app.services.rag_preprocessor import build_page_content
from app.services.rag_qa import (
    FOLLOW_UP_INDICATORS,
    answer_question,
    get_comic_overview_chunks,
    get_page_info,
    is_comic_wide_story_query,
    is_page_scoped_query,
)
from app.services.retriever import retrieve_chunks
from app.services.vector_store import get_chunks_by_page

router = APIRouter()


def _get_validated_conversation(
    conversation_id: str,
    user_id: str | None = None,
    db: Session | None = None
) -> dict:
    """
    Validates conversation_id and ensures the conversation exists and belongs to a comic owned by user_id.
    Raises HTTPException(400) on invalid UUID or HTTPException(404) if not found/unauthorized.
    """
    try:
        valid_id = validate_conversation_id(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    conversation = get_conversation(valid_id, user_id=user_id, db=db)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if user_id:
        check_comic_access(conversation.get("comic_id", ""), user_id=user_id, db=db)

    return conversation


# ============================================================
# Get / Find Conversations Endpoint
# ============================================================

@router.get("", response_model=list[ConversationDetailResponse], summary="List conversations or find by comic_id")
async def list_conversations(
    comic_id: str | None = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves conversations. When comic_id is provided, checks user ownership and returns all conversations for that comic.
    """
    if comic_id:
        valid_id = validate_comic_id(comic_id)
        check_comic_access(valid_id, current_user.id, db=db)
        return get_conversations_by_comic_id(valid_id, user_id=current_user.id, db=db)
    return []


# ============================================================
# Create Conversation Endpoint
# ============================================================

@router.post("", response_model=ConversationResponse, summary="Create a new conversation for a comic")
async def create_new_conversation(
    request: ConversationCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Initializes a new persistent conversation session tied to a comic owned by the user.
    """
    valid_comic_id = validate_comic_id(request.comic_id)
    check_comic_access(valid_comic_id, current_user.id, db=db)

    try:
        record = create_conversation(comic_id=valid_comic_id, user_id=current_user.id, db=db)
        return {
            "conversation_id": record["conversation_id"],
            "comic_id": record["comic_id"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to create conversation."
        )


# ============================================================
# Get Conversation Details Endpoint
# ============================================================

@router.get("/{conversation_id}", response_model=ConversationDetailResponse, summary="Get conversation history")
async def get_conversation_details(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves metadata and chronological message history for a conversation owned by the user.
    """
    conversation = _get_validated_conversation(conversation_id, current_user.id, db=db)
    return conversation


# ============================================================
# Delete Conversation Endpoint
# ============================================================

@router.delete("/{conversation_id}", summary="Delete a conversation")
async def delete_existing_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Safely removes a persistent conversation and its history.
    """
    _get_validated_conversation(conversation_id, current_user.id, db=db)
    valid_id = validate_conversation_id(conversation_id)

    deleted = delete_conversation(valid_id, user_id=current_user.id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": valid_id
    }


# ============================================================
# Context-Aware Chat Endpoint
# ============================================================

@router.post(
    "/{conversation_id}/ask",
    response_model=ConversationQuestionResponse,
    summary="Ask a question with conversation memory"
)
async def ask_in_conversation(
    conversation_id: str,
    request: ConversationQuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    conversation = _get_validated_conversation(conversation_id, current_user.id, db=db)
    comic_id = conversation.get("comic_id")

    if not request.question or not request.question.strip():
        return {
            "conversation_id": conversation["conversation_id"],
            "comic_id": comic_id,
            "question": request.question,
            "answer": "Please provide a valid question.",
            "sources": []
        }

    try:
        history = get_messages(
            conversation_id=conversation["conversation_id"],
            limit=MAX_CONVERSATION_MESSAGES,
            db=db
        )

        qa_result = answer_question(
            question=request.question,
            comic_id=comic_id,
            conversation_history=history,
            current_page=request.current_page
        )

        append_message(
            conversation_id=conversation["conversation_id"],
            role="user",
            content=request.question,
            db=db
        )

        append_message(
            conversation_id=conversation["conversation_id"],
            role="assistant",
            content=qa_result["answer"],
            sources=qa_result.get("sources", []),
            db=db
        )

        return {
            "conversation_id": conversation["conversation_id"],
            "comic_id": comic_id,
            "question": qa_result["question"],
            "answer": qa_result["answer"],
            "sources": qa_result.get("sources", [])
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while answering the question."
        )


@router.post(
    "/{conversation_id}/stream",
    summary="Stream question answering response with conversation memory (SSE)"
)
async def stream_in_conversation(
    conversation_id: str,
    request: ConversationQuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Streams the assistant's answer token-by-token using Server-Sent Events (SSE).
    Attaches sources metadata and persists user & assistant messages upon completion.
    """
    conversation = _get_validated_conversation(conversation_id, current_user.id, db=db)
    comic_id = conversation.get("comic_id")

    async def sse_event_generator():
        if not request.question or not request.question.strip():
            msg = "Please provide a valid question."
            yield f"event: token\ndata: {json.dumps({'token': msg})}\n\n"
            yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id, 'answer': msg, 'sources': []})}\n\n"
            return

        # -----------------------------
        # 0. Active Page Processing Check
        # -----------------------------
        page_obj = None
        if request.current_page is not None:
            page_status, page_obj = get_page_info(comic_id, request.current_page)
            if page_status == "processing":
                msg = "This page is currently being analyzed. Please wait a moment..."
                yield f"event: token\ndata: {json.dumps({'token': msg})}\n\n"
                yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id, 'answer': msg, 'sources': []})}\n\n"
                append_message(conversation["conversation_id"], role="user", content=request.question)
                append_message(conversation["conversation_id"], role="assistant", content=msg)
                return
            elif page_status == "error":
                msg = "This page could not be analyzed due to an error."
                yield f"event: token\ndata: {json.dumps({'token': msg})}\n\n"
                yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id, 'answer': msg, 'sources': []})}\n\n"
                append_message(conversation["conversation_id"], role="user", content=request.question)
                append_message(conversation["conversation_id"], role="assistant", content=msg)
                return

        history = get_messages(
            conversation_id=conversation["conversation_id"],
            limit=MAX_CONVERSATION_MESSAGES
        )

        normalized_q = normalize_query(request.question)
        retrieval_query = normalized_q
        if history:
            last_user_query = None
            for msg in reversed(history):
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    last_user_query = normalize_query(msg["content"])
                    break
            if last_user_query:
                words = set(re.findall(r"\w+", normalized_q.lower()))
                if words.intersection(FOLLOW_UP_INDICATORS) or len(words) <= 4:
                    retrieval_query = f"{last_user_query} {normalized_q}"

        is_page_scoped = is_page_scoped_query(request.question, request.current_page)
        page_chunks = get_chunks_by_page(comic_id, request.current_page) if request.current_page else []

        # Fallback to direct comic.json page content if ChromaDB has not finished embedding page chunks
        if request.current_page and not page_chunks and page_obj and page_obj.get("status") == "success":
            content = build_page_content(page_obj)
            if content.strip():
                page_chunks = [{
                    "chunk_id": f"{comic_id}_page_{request.current_page}_chunk_1",
                    "content": content,
                    "metadata": {
                        "comic_id": comic_id,
                        "page_number": request.current_page,
                        "chunk_index": 1
                    },
                    "distance": 0.0
                }]

        seen_ids = set()
        chunks = []

        if is_page_scoped:
            for chunk in page_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)
            if not chunks:
                semantic_chunks = retrieve_chunks(
                    query=normalized_q,
                    comic_id=comic_id,
                    top_k=DEFAULT_TOP_K
                )
                for chunk in semantic_chunks:
                    cid = chunk.get("chunk_id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        chunks.append(chunk)
        else:
            is_story_query = is_comic_wide_story_query(request.question)
            if is_story_query:
                overview_chunks = get_comic_overview_chunks(comic_id)
                for chunk in overview_chunks:
                    cid = chunk.get("chunk_id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        chunks.append(chunk)

            semantic_chunks = retrieve_chunks(
                query=retrieval_query,
                comic_id=comic_id,
                top_k=DEFAULT_TOP_K
            )
            if not semantic_chunks and retrieval_query != normalized_q:
                semantic_chunks = retrieve_chunks(
                    query=normalized_q,
                    comic_id=comic_id,
                    top_k=DEFAULT_TOP_K
                )
            for chunk in semantic_chunks:
                cid = chunk.get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append(chunk)

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

        if not chunks:
            fallback = "I could not find relevant information in the comic."
            yield f"event: token\ndata: {json.dumps({'token': fallback})}\n\n"
            yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id, 'answer': fallback, 'sources': []})}\n\n"
            append_message(conversation["conversation_id"], role="user", content=request.question)
            append_message(conversation["conversation_id"], role="assistant", content=fallback)
            return

        context_parts = []
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            page_number = metadata.get("page_number")
            content = chunk.get("content", "")
            chunk_index = metadata.get("chunk_index")
            header = f"[PAGE {page_number} | CHUNK {chunk_index}]"
            context_parts.append(f"{header}\n{content}")
            sources.append({
                "comic_id": metadata.get("comic_id", comic_id),
                "page_number": page_number,
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": chunk_index,
                "distance": chunk.get("distance", 0.0)
            })
        context = "\n\n--------------------\n\n".join(context_parts)

        # Yield sources early so UI receives citation references
        yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"

        # Stream generated tokens asynchronously in real time
        full_tokens = []
        async for token in stream_generate_answer_async(
            question=request.question,
            context=context,
            conversation_history=history,
            current_page=request.current_page
        ):
            full_tokens.append(token)
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(full_tokens).strip()
        cleaned_answer = clean_llm_response(full_answer, question=request.question)

        # Persist conversation turn in backend
        append_message(conversation["conversation_id"], role="user", content=request.question)
        append_message(conversation["conversation_id"], role="assistant", content=cleaned_answer)

        yield f"event: done\ndata: {json.dumps({'conversation_id': conversation_id, 'answer': cleaned_answer, 'sources': sources})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )