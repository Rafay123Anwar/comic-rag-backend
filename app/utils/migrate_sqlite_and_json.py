"""
Migration Utility: SQLite & JSON Files -> PostgreSQL & Supabase Storage

Migrates existing local data:
1. Users from SQLite (if different DB URL) -> PostgreSQL users table.
2. Comic metadata and page analysis from storage/comics/*/comic.json -> comics & comic_pages tables.
3. Conversations & messages from storage/conversations/*.json -> conversations & messages tables.
4. Ensures every comic has a valid user_id (assigns orphaned comics to an admin/legacy user).
5. (Optional) Syncs physical assets to Supabase Storage if configured.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import COMICS_DIR, CONVERSATIONS_DIR, DATABASE_URL, STORAGE_DIR
from app.core.database import SessionLocal, init_db
from app.core.security import hash_password
from app.core.supabase import (
    ensure_bucket_exists,
    is_supabase_storage_enabled,
    upload_file_to_storage,
)
from app.models.comic import Comic, ComicPage
from app.models.conversation import Conversation, Message
from app.models.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")


def get_or_create_default_admin(db) -> User:
    """
    Retrieves or creates a default admin user to own legacy comics without an assigned user_id.
    """
    user = db.query(User).filter(User.username == "admin").first()
    if not user:
        user = User(
            id=str(uuid.uuid4()),
            username="admin",
            email="admin@example.com",
            hashed_password=hash_password("AdminPass123!"),
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("[MIGRATION] Created default admin user (%s, %s)", user.id, user.email)
    return user


def migrate_all():
    logger.info("==================================================")
    logger.info("STARTING COMIC RAG SUPABASE / POSTGRESQL MIGRATION")
    logger.info("Database URL: %s", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
    logger.info("Supabase Storage Enabled: %s", is_supabase_storage_enabled())
    logger.info("==================================================")

    init_db()
    db = SessionLocal()

    migrated_comics = 0
    migrated_pages = 0
    migrated_conversations = 0
    migrated_messages = 0

    try:
        admin_user = get_or_create_default_admin(db)

        # -----------------------------
        # 1. Migrate Comics & Pages from storage/comics/
        # -----------------------------
        if COMICS_DIR.exists():
            for comic_dir in COMICS_DIR.iterdir():
                if not comic_dir.is_dir():
                    continue

                comic_json = comic_dir / "comic.json"
                if not comic_json.exists():
                    continue

                try:
                    with open(comic_json, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    meta = data.get("comic", {})
                    comic_id = meta.get("id") or comic_dir.name
                    title = meta.get("name") or "Untitled Comic"
                    source_format = meta.get("source_format", "cbr")
                    status = meta.get("status", "completed")
                    raw_pages = data.get("pages", [])
                    total_pages = meta.get("total_pages", len(raw_pages))
                    analyzed_pages = meta.get("analyzed_pages", total_pages if status == "completed" else 0)
                    successful_pages = meta.get("successful_pages", analyzed_pages)
                    failed_pages = meta.get("failed_pages", 0)
                    full_text = data.get("comic_content", {}).get("full_text", "")

                    # Ensure non-null owner
                    owner_id = meta.get("user_id")
                    if not owner_id:
                        owner_id = admin_user.id
                    else:
                        # Ensure owner user exists in database
                        user_exists = db.query(User).filter(User.id == owner_id).first()
                        if not user_exists:
                            owner_id = admin_user.id

                    # Check if Comic already in DB
                    comic = db.query(Comic).filter(Comic.id == comic_id).first()
                    if not comic:
                        comic = Comic(
                            id=comic_id,
                            user_id=owner_id,
                            title=title,
                            source_format=source_format,
                            status=status,
                            total_pages=total_pages,
                            analyzed_pages=analyzed_pages,
                            successful_pages=successful_pages,
                            failed_pages=failed_pages,
                            full_text=full_text,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(comic)
                        migrated_comics += 1
                    else:
                        comic.user_id = owner_id
                        comic.title = title
                        comic.status = status
                        comic.total_pages = total_pages
                        comic.analyzed_pages = analyzed_pages
                        comic.successful_pages = successful_pages
                        comic.failed_pages = failed_pages
                        migrated_comics += 1

                    # Update local comic.json with assigned user_id
                    data["comic"]["user_id"] = owner_id
                    with open(comic_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    # Migrate Pages
                    existing_pages = {p.page_number: p for p in db.query(ComicPage).filter(ComicPage.comic_id == comic_id).all()}
                    for p in raw_pages:
                        pnum = int(p.get("page_number", 1))
                        fname = p.get("filename", f"page_{pnum:03d}.jpg")
                        pstatus = p.get("status", "processing")
                        analysis = p.get("analysis")
                        analysis_str = json.dumps(analysis, ensure_ascii=False) if analysis else None

                        img_storage = f"user/{owner_id}/comics/{comic_id}/pages/{fname}"
                        thumb_storage = f"user/{owner_id}/comics/{comic_id}/thumbnails/thumb_p{pnum:03d}.jpg"

                        if pnum not in existing_pages:
                            page_row = ComicPage(
                                comic_id=comic_id,
                                page_number=pnum,
                                filename=fname,
                                status=pstatus,
                                image_storage_path=img_storage,
                                thumbnail_storage_path=thumb_storage,
                                analysis_json=analysis_str,
                                created_at=datetime.now(timezone.utc)
                            )
                            db.add(page_row)
                            migrated_pages += 1
                        else:
                            page_row = existing_pages[pnum]
                            page_row.filename = fname
                            page_row.status = pstatus
                            if analysis_str:
                                page_row.analysis_json = analysis_str

                    db.commit()

                    # (Optional) Supabase Storage upload
                    if is_supabase_storage_enabled():
                        ensure_bucket_exists()
                        upload_file_to_storage(f"user/{owner_id}/comics/{comic_id}/comic.json", comic_json)
                        pages_dir = comic_dir / "pages"
                        thumb_dir = comic_dir / "thumbnails"
                        if pages_dir.exists():
                            for pfile in pages_dir.iterdir():
                                if pfile.is_file():
                                    upload_file_to_storage(f"user/{owner_id}/comics/{comic_id}/pages/{pfile.name}", pfile)
                        if thumb_dir.exists():
                            for tfile in thumb_dir.iterdir():
                                if tfile.is_file():
                                    upload_file_to_storage(f"user/{owner_id}/comics/{comic_id}/thumbnails/{tfile.name}", tfile)

                except Exception as e:
                    logger.error("[MIGRATION] Error migrating comic %s: %s", comic_dir.name, str(e))
                    db.rollback()

        # -----------------------------
        # 2. Migrate Conversations & Messages from storage/conversations/
        # -----------------------------
        if CONVERSATIONS_DIR.exists():
            for cfile in CONVERSATIONS_DIR.glob("*.json"):
                try:
                    with open(cfile, "r", encoding="utf-8") as f:
                        cdata = json.load(f)

                    conv_id = cdata.get("conversation_id") or cfile.stem
                    comic_id = cdata.get("comic_id")
                    if not comic_id:
                        continue

                    # Lookup comic owner
                    comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
                    owner_id = comic_row.user_id if comic_row else admin_user.id

                    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                    if not conv:
                        conv = Conversation(
                            id=conv_id,
                            comic_id=comic_id,
                            user_id=owner_id,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(conv)
                        migrated_conversations += 1

                    # Migrate messages
                    existing_msg_count = db.query(Message).filter(Message.conversation_id == conv_id).count()
                    raw_messages = cdata.get("messages", [])
                    if existing_msg_count == 0 and raw_messages:
                        for msg in raw_messages:
                            m_row = Message(
                                conversation_id=conv_id,
                                role=msg.get("role", "user"),
                                content=msg.get("content", ""),
                                created_at=datetime.now(timezone.utc)
                            )
                            db.add(m_row)
                            migrated_messages += 1

                    db.commit()

                except Exception as e:
                    logger.error("[MIGRATION] Error migrating conversation %s: %s", cfile.name, str(e))
                    db.rollback()

    finally:
        db.close()

    logger.info("==================================================")
    logger.info("MIGRATION COMPLETED SUCCESSFULLY")
    logger.info("Comics Migrated/Synced: %d", migrated_comics)
    logger.info("Comic Pages Migrated: %d", migrated_pages)
    logger.info("Conversations Migrated: %d", migrated_conversations)
    logger.info("Messages Migrated: %d", migrated_messages)
    logger.info("==================================================")


if __name__ == "__main__":
    migrate_all()
