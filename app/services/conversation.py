"""
Conversation Service

Manages database-backed persistent conversation sessions, message history,
and user-comic scoping with PostgreSQL as primary store and JSON backup.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import CONVERSATIONS_DIR, MAX_CONVERSATION_MESSAGES
from app.core.database import SessionLocal
from app.models.comic import Comic
from app.models.conversation import Conversation, Message

logger = logging.getLogger("comic_rag")

ALLOWED_ROLES = {"user", "assistant"}


def _get_conversation_path(conversation_id: str) -> Path:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return CONVERSATIONS_DIR / f"{conversation_id}.json"


def validate_conversation_id(conversation_id: str) -> str:
    if not conversation_id or not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id is required and cannot be empty.")
    
    cleaned_id = conversation_id.strip()
    try:
        uuid.UUID(cleaned_id)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("Invalid conversation_id format. Expected a valid UUID.")
    
    return cleaned_id


def create_conversation(comic_id: str, user_id: str, db: Optional[Session] = None) -> dict:
    """
    Creates a new conversation record in PostgreSQL database scoped to comic_id and user_id.
    """
    if not comic_id or not isinstance(comic_id, str) or not comic_id.strip():
        raise ValueError("comic_id is required to create a conversation.")
    if not user_id:
        raise ValueError("user_id is required to create a conversation.")

    cleaned_comic_id = comic_id.strip()
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        conv_model = Conversation(
            id=conversation_id,
            comic_id=cleaned_comic_id,
            user_id=user_id,
            created_at=now,
            updated_at=now
        )
        db.add(conv_model)
        db.commit()
    except Exception as e:
        logger.warning("[CONVERSATION] DB creation error: %s", str(e))
    finally:
        if close_session:
            db.close()

    # Save local JSON backup
    record = {
        "conversation_id": conversation_id,
        "comic_id": cleaned_comic_id,
        "user_id": user_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        "messages": []
    }
    try:
        file_path = _get_conversation_path(conversation_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return record


def get_conversation(
    conversation_id: str,
    user_id: Optional[str] = None,
    db: Optional[Session] = None
) -> dict | None:
    """
    Retrieves a conversation record with all messages from database.
    """
    cleaned_id = validate_conversation_id(conversation_id)

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        query = db.query(Conversation).filter(Conversation.id == cleaned_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        conv = query.first()

        if conv:
            msgs = []
            for m in conv.messages:
                sources = None
                if m.sources_json:
                    try:
                        sources = json.loads(m.sources_json)
                    except Exception:
                        pass
                msgs.append({
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.created_at.isoformat() if m.created_at else "",
                    "sources": sources
                })

            return {
                "conversation_id": conv.id,
                "comic_id": conv.comic_id,
                "user_id": conv.user_id,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
                "messages": msgs
            }
    except Exception as e:
        logger.warning("[CONVERSATION] DB get error for %s: %s", cleaned_id, str(e))
    finally:
        if close_session:
            db.close()

    # Fallback to local JSON file
    file_path = _get_conversation_path(cleaned_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if user_id and data.get("user_id") and data.get("user_id") != user_id:
                    return None
                return data
        except Exception:
            pass
    return None


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    db: Optional[Session] = None
) -> dict:
    """
    Appends a new user or assistant message to the specified conversation in PostgreSQL DB.
    """
    cleaned_id = validate_conversation_id(conversation_id)
    if role not in ALLOWED_ROLES:
        raise ValueError(f"Invalid message role '{role}'. Must be one of: {ALLOWED_ROLES}")

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    sources_str = json.dumps(sources, ensure_ascii=False) if sources else None

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        conv = db.query(Conversation).filter(Conversation.id == cleaned_id).first()
        if conv:
            msg = Message(
                conversation_id=cleaned_id,
                role=role,
                content=str(content) if content is not None else "",
                sources_json=sources_str,
                created_at=now
            )
            conv.updated_at = now
            db.add(msg)
            db.commit()
    except Exception as e:
        logger.warning("[CONVERSATION] DB append_message error for %s: %s", cleaned_id, str(e))
    finally:
        if close_session:
            db.close()

    # Update local JSON backup file
    record = get_conversation(cleaned_id, db=db)
    if record:
        try:
            file_path = _get_conversation_path(cleaned_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return record

    raise FileNotFoundError(f"Conversation '{cleaned_id}' not found.")


def get_messages(conversation_id: str, limit: Optional[int] = None, db: Optional[Session] = None) -> list[dict]:
    record = get_conversation(conversation_id, db=db)
    if record is None:
        raise FileNotFoundError(f"Conversation '{conversation_id}' not found.")

    messages = record.get("messages", [])
    max_msgs = limit if limit is not None else MAX_CONVERSATION_MESSAGES
    if max_msgs > 0:
        return messages[-max_msgs:]
    return messages


def delete_conversation(
    conversation_id: str,
    user_id: Optional[str] = None,
    db: Optional[Session] = None
) -> bool:
    cleaned_id = validate_conversation_id(conversation_id)

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    deleted = False
    try:
        query = db.query(Conversation).filter(Conversation.id == cleaned_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        conv = query.first()
        if conv:
            db.delete(conv)
            db.commit()
            deleted = True
    except Exception as e:
        logger.warning("[CONVERSATION] DB delete error for %s: %s", cleaned_id, str(e))
    finally:
        if close_session:
            db.close()

    # Clean local file
    file_path = _get_conversation_path(cleaned_id)
    if file_path.exists():
        try:
            file_path.unlink()
            deleted = True
        except Exception:
            pass

    return deleted


def get_conversations_by_comic_id(
    comic_id: str,
    user_id: Optional[str] = None,
    db: Optional[Session] = None
) -> list[dict]:
    if not comic_id or not isinstance(comic_id, str) or not comic_id.strip():
        return []

    cleaned_id = comic_id.strip()

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        query = db.query(Conversation).filter(Conversation.comic_id == cleaned_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        convs = query.order_by(Conversation.updated_at.desc()).all()

        if convs:
            results = []
            for conv in convs:
                msgs = [
                    {"role": m.role, "content": m.content, "timestamp": m.created_at.isoformat() if m.created_at else ""}
                    for m in conv.messages
                ]
                results.append({
                    "conversation_id": conv.id,
                    "comic_id": conv.comic_id,
                    "user_id": conv.user_id,
                    "created_at": conv.created_at.isoformat() if conv.created_at else "",
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
                    "messages": msgs
                })
            return results
    except Exception as e:
        logger.warning("[CONVERSATION] DB get_by_comic_id error: %s", str(e))
    finally:
        if close_session:
            db.close()

    # Fallback to local files
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for file_path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("comic_id") == cleaned_id:
                    if user_id and data.get("user_id") and data.get("user_id") != user_id:
                        continue
                    results.append(data)
        except Exception:
            continue

    results.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return results


def get_or_create_comic_conversation(
    comic_id: str,
    user_id: str,
    db: Optional[Session] = None
) -> dict:
    existing = get_conversations_by_comic_id(comic_id, user_id=user_id, db=db)
    if existing:
        return existing[0]

    return create_conversation(comic_id=comic_id, user_id=user_id, db=db)


def delete_conversations_by_comic_id(
    comic_id: str,
    user_id: Optional[str] = None,
    db: Optional[Session] = None
) -> int:
    if not comic_id or not isinstance(comic_id, str) or not comic_id.strip():
        return 0

    cleaned_id = comic_id.strip()

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    deleted_count = 0
    try:
        query = db.query(Conversation).filter(Conversation.comic_id == cleaned_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        convs = query.all()
        for conv in convs:
            db.delete(conv)
            deleted_count += 1
        db.commit()
    except Exception as e:
        logger.warning("[CONVERSATION] DB delete by comic error: %s", str(e))
    finally:
        if close_session:
            db.close()

    # Also clean local files
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in list(CONVERSATIONS_DIR.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("comic_id") == cleaned_id:
                if user_id and data.get("user_id") and data.get("user_id") != user_id:
                    continue
                file_path.unlink(missing_ok=True)
                deleted_count += 1
        except Exception:
            continue

    return deleted_count
