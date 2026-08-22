"""
Storage & Comic Persistence Service

Coordinates database persistence (PostgreSQL as primary source of truth),
Supabase Storage synchronization for remote persistence,
and local disk caching for high-performance reading and AI indexing.
"""
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import COMICS_DIR, TEMP_DIR, UPLOADS_DIR
from app.core.database import SessionLocal
from app.core.supabase import (
    delete_storage_files,
    download_bytes_from_storage,
    ensure_bucket_exists,
    get_signed_storage_urls,
    is_supabase_storage_enabled,
    upload_file_to_storage,
)
from app.models.comic import Comic, ComicPage
from app.services.rag_preprocessor import build_complete_text

logger = logging.getLogger("comic_rag")


def build_comic_full_text(pages: list) -> str:
    """
    Combines page-level extracted text in ascending page order.
    """
    full_text_parts = []

    ordered_pages = sorted(
        pages,
        key=lambda page: int(page.get("page_number", 0))
    )

    for page in ordered_pages:
        if page.get("status") != "success":
            continue

        analysis = page.get("analysis") or {}
        text_data = analysis.get("text", {})

        if not isinstance(text_data, dict):
            text_data = {"full_text": str(text_data)}

        page_text = build_complete_text(text_data).strip()
        if not page_text:
            continue

        page_number = page.get("page_number", 1)
        full_text_parts.append(
            f"PAGE {page_number}\n{page_text}"
        )

    return "\n\n".join(full_text_parts)


def get_comic_json_data(comic_id: str) -> dict | None:
    """
    Fallback reader for local comic.json file if available.
    """
    if not comic_id:
        return None
    comic_json_path = Path(COMICS_DIR) / comic_id / "comic.json"
    if not comic_json_path.exists():
        return None
    try:
        with open(comic_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_comic_user_id(comic_id: str, db: Optional[Session] = None) -> str | None:
    """
    Retrieves the owner user_id for a comic. Checks DB first, then local comic.json.
    """
    if db is not None:
        comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
        if comic_row:
            return comic_row.user_id

    # Fallback to local session
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
        if comic_row:
            return comic_row.user_id
    except Exception:
        pass
    finally:
        if close_session:
            db.close()

    # Fallback to disk JSON
    data = get_comic_json_data(comic_id)
    if data:
        return data.get("comic", {}).get("user_id")
    return None


def save_comic_to_db(
    db: Session,
    comic_id: str,
    comic_name: str,
    source_format: str,
    pages: list,
    status: str = "completed",
    total_pages: Optional[int] = None,
    user_id: Optional[str] = None,
    original_storage_path: Optional[str] = None
) -> Comic:
    """
    Persists comic metadata and per-page records into PostgreSQL database.
    Only saves verified storage paths without guessing or fabricated values.
    """
    actual_total = total_pages if total_pages is not None else len(pages)
    successful_pages = sum(1 for p in pages if p.get("status") == "success")
    failed_pages = sum(1 for p in pages if p.get("status") == "error")
    analyzed_pages = successful_pages + failed_pages
    full_text = build_comic_full_text(pages)

    # Determine status
    if analyzed_pages >= actual_total and actual_total > 0:
        if successful_pages == 0 and failed_pages > 0:
            final_status = "failed"
        else:
            final_status = "completed"
    else:
        final_status = status

    comic = db.query(Comic).filter(Comic.id == comic_id).first()
    if not comic:
        comic = Comic(
            id=comic_id,
            user_id=user_id,
            title=comic_name,
            source_format=source_format,
            status=final_status,
            total_pages=actual_total,
            analyzed_pages=analyzed_pages,
            successful_pages=successful_pages,
            failed_pages=failed_pages,
            original_storage_path=original_storage_path,
            full_text=full_text,
        )
        db.add(comic)
    else:
        if user_id:
            comic.user_id = user_id
        comic.title = comic_name
        comic.source_format = source_format
        comic.status = final_status
        comic.total_pages = actual_total
        comic.analyzed_pages = analyzed_pages
        comic.successful_pages = successful_pages
        comic.failed_pages = failed_pages
        if original_storage_path:
            comic.original_storage_path = original_storage_path
        comic.full_text = full_text

    # Upsert ComicPage rows
    existing_pages = {p.page_number: p for p in db.query(ComicPage).filter(ComicPage.comic_id == comic_id).all()}
    for page_dict in pages:
        pnum = int(page_dict.get("page_number", 1))
        fname = page_dict.get("filename", f"page_{pnum:03d}.jpg")
        pstatus = page_dict.get("status", "processing")
        analysis_data = page_dict.get("analysis")
        analysis_str = json.dumps(analysis_data, ensure_ascii=False) if analysis_data else None

        img_storage = page_dict.get("image_storage_path") or page_dict.get("storage_path")
        thumb_storage = page_dict.get("thumbnail_storage_path")

        if pnum in existing_pages:
            db_page = existing_pages[pnum]
            db_page.filename = fname
            db_page.status = pstatus
            if analysis_str:
                db_page.analysis_json = analysis_str
            if img_storage:
                db_page.image_storage_path = img_storage
            if thumb_storage:
                db_page.thumbnail_storage_path = thumb_storage
        else:
            db_page = ComicPage(
                comic_id=comic_id,
                page_number=pnum,
                filename=fname,
                status=pstatus,
                image_storage_path=img_storage,
                thumbnail_storage_path=thumb_storage,
                analysis_json=analysis_str
            )
            db.add(db_page)

    db.commit()
    db.refresh(comic)
    return comic


def save_comic_json(
    comic_id: str,
    comic_name: str,
    source_format: str,
    pages: list,
    status: str = "completed",
    total_pages: Optional[int] = None,
    user_id: Optional[str] = None
) -> dict:
    """
    Persists structured comic metadata to database as primary store,
    and updates local comic.json backup file.
    """
    comic_dir = Path(COMICS_DIR) / comic_id
    comic_dir.mkdir(parents=True, exist_ok=True)
    output_path = comic_dir / "comic.json"

    # Retain existing user_id if not explicitly passed
    resolved_user_id = user_id
    if not resolved_user_id and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                resolved_user_id = existing_data.get("comic", {}).get("user_id")
        except Exception:
            pass

    # Save to PostgreSQL DB
    db = SessionLocal()
    try:
        save_comic_to_db(
            db=db,
            comic_id=comic_id,
            comic_name=comic_name,
            source_format=source_format,
            pages=pages,
            status=status,
            total_pages=total_pages,
            user_id=resolved_user_id
        )
    except Exception as e:
        logger.warning("[STORAGE] Database save error for comic %s: %s", comic_id, str(e))
    finally:
        db.close()

    # Save local comic.json backup
    successful_pages = sum(1 for page in pages if page.get("status") == "success")
    failed_pages = sum(1 for page in pages if page.get("status") == "error")
    analyzed_pages = successful_pages + failed_pages
    actual_total = total_pages if total_pages is not None else len(pages)
    full_text = build_comic_full_text(pages)

    comic_info = {
        "id": comic_id,
        "name": comic_name,
        "source_format": source_format,
        "total_pages": actual_total,
        "status": status,
        "analyzed_pages": analyzed_pages if status != "completed" else actual_total,
        "successful_pages": successful_pages,
        "failed_pages": failed_pages
    }
    if resolved_user_id:
        comic_info["user_id"] = resolved_user_id

    data = {
        "comic": comic_info,
        "comic_content": {
            "full_text": full_text
        },
        "pages": pages
    }

    try:
        comic_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.warning("[STORAGE] Backup comic.json save error for comic %s: %s", comic_id, str(e))

    return data


def sync_comic_assets_to_supabase(
    comic_id: str,
    user_id: str,
    pages: list,
    original_file_path: Optional[Path] = None
) -> None:
    """
    Uploads original file, extracted pages, thumbnails, and backup comic.json to Supabase Storage
    and verifies/persists storage paths to PostgreSQL.
    """
    if not is_supabase_storage_enabled() or not user_id:
        return

    from concurrent.futures import ThreadPoolExecutor
    ensure_bucket_exists()

    comic_dir = Path(COMICS_DIR) / comic_id
    pages_dir = comic_dir / "pages"
    thumb_dir = comic_dir / "thumbnails"
    json_path = comic_dir / "comic.json"

    # 1. Upload original file
    if original_file_path and original_file_path.exists():
        orig_storage = f"user/{user_id}/comics/{comic_id}/original/{original_file_path.name}"
        upload_file_to_storage(
            storage_path=orig_storage,
            local_file_path=original_file_path,
            content_type=None
        )

    # 2. Upload comic.json backup
    if json_path.exists():
        upload_file_to_storage(
            storage_path=f"user/{user_id}/comics/{comic_id}/comic.json",
            local_file_path=json_path,
            content_type="application/json"
        )

    # 3. Upload extracted pages & thumbnails with verified path updates
    page_tasks = []
    for page in pages:
        pnum = int(page.get("page_number", 1))
        fname = page.get("filename", f"page_{pnum:03d}.jpg")
        local_pfile = pages_dir / fname
        local_tfile = thumb_dir / f"thumb_p{pnum:03d}.jpg"

        p_item = {
            "page_number": pnum,
            "filename": fname,
            "pfile": local_pfile if local_pfile.exists() and local_pfile.is_file() else None,
            "tfile": local_tfile if local_tfile.exists() and local_tfile.is_file() else None,
        }
        page_tasks.append(p_item)

    uploaded_updates = {}

    def _do_page_upload(item):
        pnum = item["page_number"]
        fname = item["filename"]
        pfile = item["pfile"]
        tfile = item["tfile"]
        img_uploaded = None
        thumb_uploaded = None

        if pfile:
            img_uploaded = upload_file_to_storage(
                storage_path=f"user/{user_id}/comics/{comic_id}/pages/{fname}",
                local_file_path=pfile,
                content_type=None
            )
        if tfile:
            thumb_uploaded = upload_file_to_storage(
                storage_path=f"user/{user_id}/comics/{comic_id}/thumbnails/thumb_p{pnum:03d}.jpg",
                local_file_path=tfile,
                content_type="image/jpeg"
            )

        return pnum, img_uploaded, thumb_uploaded

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(_do_page_upload, page_tasks)
        for pnum, img_path, thumb_path in results:
            if img_path or thumb_path:
                uploaded_updates[pnum] = (img_path, thumb_path)

    # Update DB with verified paths
    if uploaded_updates:
        db = SessionLocal()
        try:
            db_pages = db.query(ComicPage).filter(ComicPage.comic_id == comic_id).all()
            for db_p in db_pages:
                if db_p.page_number in uploaded_updates:
                    img_p, thumb_p = uploaded_updates[db_p.page_number]
                    if img_p:
                        db_p.image_storage_path = img_p
                    if thumb_p:
                        db_p.thumbnail_storage_path = thumb_p
            db.commit()
        except Exception as e:
            logger.warning("[SUPABASE] Failed to persist uploaded paths to DB for comic %s: %s", comic_id, str(e))
        finally:
            db.close()

    logger.info("[SUPABASE] Synced %d page assets to Supabase Storage for comic %s", len(uploaded_updates), comic_id)


def backfill_missing_comic_storage_assets(comic_id: Optional[str] = None, user_id: Optional[str] = None) -> int:
    """
    Backfills missing Supabase Storage page images and thumbnails from local disk cache,
    and updates PostgreSQL ComicPage records with verified storage paths.
    """
    if not is_supabase_storage_enabled():
        return 0

    db = SessionLocal()
    backfilled_count = 0
    try:
        query = db.query(Comic)
        if comic_id:
            query = query.filter(Comic.id == comic_id)
        if user_id:
            query = query.filter(Comic.user_id == user_id)
        comics = query.all()

        for c in comics:
            owner_id = c.user_id
            if not owner_id:
                continue

            comic_dir = Path(COMICS_DIR) / c.id
            pages_dir = comic_dir / "pages"
            thumb_dir = comic_dir / "thumbnails"

            if not pages_dir.exists():
                continue

            for db_p in c.pages:
                fname = db_p.filename or f"page_{db_p.page_number:03d}.jpg"
                local_pfile = pages_dir / fname
                local_tfile = thumb_dir / f"thumb_p{db_p.page_number:03d}.jpg"

                if local_pfile.exists() and local_pfile.is_file():
                    uploaded = upload_file_to_storage(
                        storage_path=f"user/{owner_id}/comics/{c.id}/pages/{fname}",
                        local_file_path=local_pfile
                    )
                    if uploaded:
                        db_p.image_storage_path = uploaded
                        backfilled_count += 1

                if local_tfile.exists() and local_tfile.is_file():
                    uploaded_t = upload_file_to_storage(
                        storage_path=f"user/{owner_id}/comics/{c.id}/thumbnails/thumb_p{db_p.page_number:03d}.jpg",
                        local_file_path=local_tfile,
                        content_type="image/jpeg"
                    )
                    if uploaded_t:
                        db_p.thumbnail_storage_path = uploaded_t
                        backfilled_count += 1

            db.commit()
    except Exception as e:
        logger.warning("[STORAGE] Backfill error: %s", str(e))
    finally:
        db.close()

    return backfilled_count


def get_comic_data(comic_id: str, db: Optional[Session] = None) -> dict | None:
    """
    Retrieves full comic structured metadata including pages.
    Queries PostgreSQL DB first, falling back to local comic.json.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        comic = db.query(Comic).filter(Comic.id == comic_id).first()
        if comic:
            pages = []
            for p in comic.pages:
                analysis = None
                if p.analysis_json:
                    try:
                        analysis = json.loads(p.analysis_json)
                    except Exception:
                        pass
                pages.append({
                    "page_number": p.page_number,
                    "filename": p.filename,
                    "image_path": f"storage/comics/{comic.id}/pages/{p.filename}",
                    "thumbnail_path": f"storage/comics/{comic.id}/thumbnails/thumb_p{p.page_number:03d}.jpg",
                    "status": p.status,
                    "analysis": analysis,
                    "metadata": {
                        "page_number": p.page_number,
                        "has_text": bool(analysis and analysis.get("text", {}).get("full_text"))
                    },
                    "image_storage_path": p.image_storage_path,
                    "thumbnail_storage_path": p.thumbnail_storage_path
                })

            return {
                "comic": {
                    "id": comic.id,
                    "name": comic.title,
                    "source_format": comic.source_format,
                    "total_pages": comic.total_pages,
                    "status": comic.status,
                    "analyzed_pages": comic.analyzed_pages,
                    "successful_pages": comic.successful_pages,
                    "failed_pages": comic.failed_pages,
                    "user_id": comic.user_id,
                    "uploaded_at": comic.created_at.isoformat() if comic.created_at else None
                },
                "comic_content": {
                    "full_text": comic.full_text or ""
                },
                "pages": pages
            }
    except Exception as e:
        logger.warning("[STORAGE] DB query error for comic %s: %s", comic_id, str(e))
    finally:
        if close_session:
            db.close()

    # Fallback to local comic.json
    return get_comic_json_data(comic_id)


def list_all_comics(user_id: Optional[str] = None, db: Optional[Session] = None) -> list[dict]:
    """
    Lists all comics owned by user_id from PostgreSQL database.
    Falls back to scanning local storage directory if database is unavailable.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        query = db.query(Comic)
        if user_id is not None:
            query = query.filter(Comic.user_id == user_id)
        comic_rows = query.order_by(Comic.created_at.desc()).all()

        comic_items = []
        cover_paths = []
        for c in comic_rows:
            uid = c.user_id or user_id
            cover_path = f"user/{uid}/comics/{c.id}/thumbnails/thumb_p001.jpg" if uid else None
            if cover_path:
                cover_paths.append(cover_path)

            comic_items.append({
                "comic_id": c.id,
                "title": c.title,
                "total_pages": c.total_pages,
                "status": c.status,
                "analyzed_pages": c.analyzed_pages,
                "source_format": c.source_format,
                "uploaded_at": c.created_at.isoformat() if c.created_at else datetime.now(timezone.utc).isoformat(),
                "last_opened_at": None,
                "user_id": c.user_id,
                "cover_storage_path": cover_path,
                "cover_thumbnail_url": None,
            })

        # Batch signed URLs for covers if Supabase enabled
        if is_supabase_storage_enabled() and cover_paths:
            signed_map = get_signed_storage_urls(cover_paths, expires_in=3600)
            for item in comic_items:
                cpath = item.pop("cover_storage_path", None)
                if cpath:
                    item["cover_thumbnail_url"] = signed_map.get(cpath)
        else:
            for item in comic_items:
                item.pop("cover_storage_path", None)

        return comic_items
    except Exception as e:
        logger.warning("[STORAGE] DB list_all_comics error: %s", str(e))
    finally:
        if close_session:
            db.close()

    # Fallback to local filesystem scanning
    COMICS_DIR.mkdir(parents=True, exist_ok=True)
    comics = []
    for comic_dir in COMICS_DIR.iterdir():
        if not comic_dir.is_dir():
            continue
        comic_json_path = comic_dir / "comic.json"
        if not comic_json_path.exists():
            continue
        try:
            mtime = comic_json_path.stat().st_mtime
            uploaded_at_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            with open(comic_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            comic_meta = data.get("comic", {})
            owner_id = comic_meta.get("user_id")

            if user_id is not None and owner_id != user_id:
                continue

            comics.append({
                "comic_id": comic_meta.get("id") or comic_dir.name,
                "title": comic_meta.get("name") or "Untitled Comic",
                "total_pages": comic_meta.get("total_pages", len(data.get("pages", []))),
                "status": comic_meta.get("status", "completed"),
                "analyzed_pages": comic_meta.get("analyzed_pages", 0),
                "source_format": comic_meta.get("source_format", "cbr"),
                "uploaded_at": uploaded_at_iso,
                "last_opened_at": None,
                "user_id": owner_id
            })
        except Exception:
            continue

    comics.sort(key=lambda c: c.get("uploaded_at", ""), reverse=True)
    return comics


def delete_comic_storage(comic_id: str, user_id: Optional[str] = None, db: Optional[Session] = None) -> bool:
    """
    Deletes a comic completely:
    1. Removes database record (cascading to pages, conversations, messages).
    2. Deletes remote files from Supabase Storage.
    3. Cleans up local disk cache directories.
    """
    if not comic_id or not isinstance(comic_id, str) or not comic_id.strip():
        return False

    cleaned_id = comic_id.strip()

    # 1. Delete from PostgreSQL Database
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        comic_row = db.query(Comic).filter(Comic.id == cleaned_id).first()
        if comic_row:
            if not user_id:
                user_id = comic_row.user_id
            db.delete(comic_row)
            db.commit()
    except Exception as e:
        logger.warning("[STORAGE] DB deletion error for comic %s: %s", cleaned_id, str(e))
    finally:
        if close_session:
            db.close()

    # 2. Delete from Supabase Storage
    if is_supabase_storage_enabled() and user_id:
        try:
            storage_paths_to_delete = [
                f"user/{user_id}/comics/{cleaned_id}/comic.json",
                f"user/{user_id}/comics/{cleaned_id}/original/{cleaned_id}",
            ]
            delete_storage_files(storage_paths_to_delete)
        except Exception as e:
            logger.warning("[STORAGE] Supabase Storage deletion error for comic %s: %s", cleaned_id, str(e))

    # 3. Clean up local disk directories
    deleted_any = False
    comic_dir = Path(COMICS_DIR) / cleaned_id
    if comic_dir.exists():
        shutil.rmtree(comic_dir, ignore_errors=True)
        deleted_any = True

    if UPLOADS_DIR.exists():
        for upload_file in list(UPLOADS_DIR.glob(f"{cleaned_id}.*")):
            try:
                upload_file.unlink(missing_ok=True)
                deleted_any = True
            except Exception:
                pass

    temp_dir = Path(TEMP_DIR) / cleaned_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        deleted_any = True

    return deleted_any