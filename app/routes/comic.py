"""
Comic Routes

API endpoints for comic upload, metadata retrieval, and question-answering.
"""
import json
import mimetypes
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import (
    ALLOWED_EXTENSIONS,
    COMICS_DIR,
    MAX_AI_RETRIES,
    MAX_AI_WORKERS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_DIR,
)
from app.core.database import SessionLocal, get_db
from app.core.logging import logger
from app.core.supabase import (
    download_bytes_from_storage,
    get_signed_storage_urls,
    is_supabase_storage_enabled,
    upload_file_to_storage,
)
from app.models.comic import Comic, ComicPage
from app.models.user import User
from app.schemas.comic import (
    ComicDeleteResponse,
    ComicListItem,
    ComicStatusResponse,
    ComicUploadResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.schemas.conversation import ConversationDetailResponse
from app.services.ai_analyzer import analyze_pages
from app.services.auth import get_current_active_user
from app.services.conversation import (
    delete_conversations_by_comic_id,
    get_or_create_comic_conversation,
)
from app.services.extractor import (
    ensure_page_thumbnail,
    extract_cbr,
    extract_cbz,
    extract_image,
    extract_pdf,
)
from app.services.rag_ingestion import ingest_comic_to_rag, ingest_page_to_rag
from app.services.rag_qa import answer_question
from app.services.storage import (
    delete_comic_storage,
    get_comic_data,
    get_comic_json_data,
    get_comic_user_id,
    list_all_comics,
    save_comic_json,
    sync_comic_assets_to_supabase,
)
from app.services.vector_store import delete_chunks_by_comic_id

router = APIRouter()

# Ensure uploads directory exists on startup
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = UPLOADS_DIR

_active_comic_processing: set[str] = set()
_active_comic_lock = threading.Lock()


def run_background_comic_analysis(
    comic_id: str,
    pages: list[dict],
    comic_name: str,
    source_format: str,
    initial_pages: list[dict] | None = None,
    user_id: str | None = None
):
    """
    Executes visual AI analysis and asset delivery pipeline in the background.
    Lifecycle per page:
      AI analysis -> thumbnail generation -> image upload -> thumbnail upload -> database save -> COMPLETE -> non-blocking RAG.
    Guarantees independent exception boundaries, verified storage paths, and 100% progress completion.
    """
    with _active_comic_lock:
        if comic_id in _active_comic_processing:
            logger.warning("[BACKGROUND] Background analysis already running for comic %s. Skipping duplicate dispatch.", comic_id)
            return
        _active_comic_processing.add(comic_id)

    logger.info("[BACKGROUND] Starting AI visual analysis pipeline for comic %s (%d pages)...", comic_id, len(pages))
    t0 = time.perf_counter()

    try:
        current_pages = [dict(p) for p in (initial_pages or pages)]

        def handle_page_analyzed(page_result: dict, completed_count: int, total_count: int):
            page_num = int(page_result.get("page_number", 1))
            logger.info("[PAGE PIPELINE] START page=%d", page_num)

            try:
                # 1. AI Analysis Status
                if page_result.get("status") == "success":
                    logger.info("[PAGE PIPELINE] AI ANALYSIS SUCCESS page=%d", page_num)
                else:
                    logger.warning("[PAGE PIPELINE] AI ANALYSIS FAILED page=%d error=%s", page_num, page_result.get("error", "Unknown"))

                # 2. Thumbnail Generation
                logger.info("[PAGE PIPELINE] THUMBNAIL START page=%d", page_num)
                local_img_path = Path(page_result.get("image_path") or f"storage/comics/{comic_id}/pages/page_{page_num:03d}.jpg")
                thumb_path = ensure_page_thumbnail(comic_id, page_num, local_img_path)
                if thumb_path and thumb_path.exists():
                    logger.info("[PAGE PIPELINE] THUMBNAIL SUCCESS page=%d", page_num)
                else:
                    logger.warning("[PAGE PIPELINE] THUMBNAIL FAILED page=%d (thumbnail missing)", page_num)

                # 3. Image Upload to Supabase Storage
                img_storage_path = None
                thumb_storage_path = None
                if is_supabase_storage_enabled() and user_id:
                    if local_img_path.exists() and local_img_path.is_file():
                        logger.info("[PAGE PIPELINE] IMAGE UPLOAD START page=%d", page_num)
                        img_storage_path = upload_file_to_storage(
                            storage_path=f"user/{user_id}/comics/{comic_id}/pages/{local_img_path.name}",
                            local_file_path=local_img_path
                        )
                        if img_storage_path:
                            logger.info("[PAGE PIPELINE] IMAGE UPLOAD SUCCESS page=%d", page_num)
                        else:
                            logger.error("[PAGE PIPELINE] IMAGE UPLOAD FAILED page=%d", page_num)

                    # 4. Thumbnail Upload to Supabase Storage
                    if thumb_path and thumb_path.exists() and thumb_path.is_file():
                        logger.info("[PAGE PIPELINE] THUMBNAIL UPLOAD START page=%d", page_num)
                        thumb_storage_path = upload_file_to_storage(
                            storage_path=f"user/{user_id}/comics/{comic_id}/thumbnails/thumb_p{page_num:03d}.jpg",
                            local_file_path=thumb_path,
                            content_type="image/jpeg"
                        )
                        if thumb_storage_path:
                            logger.info("[PAGE PIPELINE] THUMBNAIL UPLOAD SUCCESS page=%d", page_num)
                        else:
                            logger.error("[PAGE PIPELINE] THUMBNAIL UPLOAD FAILED page=%d", page_num)

                # Attach verified paths to page result
                if img_storage_path:
                    page_result["image_storage_path"] = img_storage_path
                if thumb_storage_path:
                    page_result["thumbnail_storage_path"] = thumb_storage_path

                # Update page in current_pages state
                for idx, p in enumerate(current_pages):
                    if int(p.get("page_number", 1)) == page_num:
                        current_pages[idx] = page_result
                        break

                # 5. Database Save
                save_comic_json(
                    comic_id=comic_id,
                    comic_name=comic_name,
                    source_format=source_format,
                    pages=current_pages,
                    status="processing",
                    total_pages=len(pages),
                    user_id=user_id
                )
                logger.info("[PAGE PIPELINE] DB SAVE SUCCESS page=%d", page_num)

            except Exception as page_err:
                logger.exception("[PAGE PIPELINE] FAILED page=%d error=%s exception_type=%s", page_num, str(page_err), type(page_err).__name__)
                page_result["status"] = "error"
                page_result["error"] = str(page_err)
                for idx, p in enumerate(current_pages):
                    if int(p.get("page_number", 1)) == page_num:
                        current_pages[idx] = page_result
                        break

            # 6. Mark Page Processed & Complete
            processed_count = sum(1 for p in current_pages if p.get("status") in ("success", "error"))
            progress_pct = (processed_count / max(1, len(pages))) * 100
            logger.info("[PAGE PIPELINE] COMPLETE page=%d (%d/%d processed, progress=%.1f%%)", page_num, processed_count, len(pages), progress_pct)

            # 7. Non-blocking RAG Ingestion (MUST NOT block page completion)
            if page_result.get("status") == "success":
                try:
                    ingest_page_to_rag(
                        comic_id=comic_id,
                        comic_name=comic_name,
                        source_format=source_format,
                        page=page_result
                    )
                    logger.info("[PAGE PIPELINE] RAG INGESTION SUCCESS page=%d", page_num)
                except Exception as ing_err:
                    logger.warning("[PAGE PIPELINE] RAG INGESTION FAILED page=%d error=%s (non-fatal)", page_num, str(ing_err))

        # Run AI visual page analysis with worker concurrency
        analyzed_pages = analyze_pages(
            pages,
            max_workers=MAX_AI_WORKERS,
            max_retries=MAX_AI_RETRIES,
            on_page_complete=handle_page_analyzed
        )

        # 8. Final Reconciliation Step
        db = SessionLocal()
        try:
            db_pages = db.query(ComicPage).filter(ComicPage.comic_id == comic_id).all()
            succ_count = sum(1 for p in db_pages if p.status == "success")
            fail_count = sum(1 for p in db_pages if p.status == "error")
            proc_count = succ_count + fail_count
            tot_count = len(pages)

            final_status = "failed" if (succ_count == 0 and fail_count > 0) else "completed"

            # Update comic status in DB
            comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
            if comic_row:
                comic_row.status = final_status
                comic_row.analyzed_pages = proc_count
                comic_row.successful_pages = succ_count
                comic_row.failed_pages = fail_count
                db.commit()

            # Final comic.json save
            save_comic_json(
                comic_id=comic_id,
                comic_name=comic_name,
                source_format=source_format,
                pages=analyzed_pages,
                status=final_status,
                total_pages=tot_count,
                user_id=user_id
            )

            logger.info(
                "[COMIC PIPELINE] FINISHED comic=%s total=%d successful=%d failed=%d processed=%d progress=100%%",
                comic_id,
                tot_count,
                succ_count,
                fail_count,
                proc_count
            )
        except Exception as rec_err:
            logger.exception("[COMIC PIPELINE] Final reconciliation error for %s: %s", comic_id, str(rec_err))
        finally:
            db.close()

        t_total = time.perf_counter() - t0
        logger.info("[BACKGROUND] Comic %s pipeline finished in %.2fs.", comic_id, t_total)

    except Exception as e:
        logger.exception("[COMIC PIPELINE] CRITICAL FATAL ERROR for comic %s: %s", comic_id, str(e))
        try:
            save_comic_json(
                comic_id=comic_id,
                comic_name=comic_name,
                source_format=source_format,
                pages=initial_pages or pages,
                status="failed",
                total_pages=len(pages),
                user_id=user_id
            )
        except Exception:
            pass
    finally:
        with _active_comic_lock:
            _active_comic_processing.discard(comic_id)


def validate_comic_id(comic_id: str) -> str:
    """
    Validate comic_id is non-empty and formatted as a valid UUID.
    """
    if not comic_id or not comic_id.strip():
        raise HTTPException(
            status_code=400,
            detail="comic_id is required"
        )
    cleaned = comic_id.strip()
    try:
        uuid.UUID(cleaned)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid comic_id format: '{cleaned}'. Expected a valid UUID."
        )
    return cleaned


def check_comic_exists(comic_id: str, db: Session | None = None) -> Path:
    """
    Validate that a comic exists in database or local storage.
    """
    comic_dir = Path(COMICS_DIR) / comic_id
    comic_json_path = comic_dir / "comic.json"

    # Check PostgreSQL DB first
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    try:
        from app.models.comic import Comic
        comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
        if comic_row is not None:
            return comic_json_path
    except Exception:
        pass
    finally:
        if close_session:
            db.close()

    if not comic_json_path.exists() and not comic_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Comic not found"
        )
    return comic_json_path


def check_comic_access(comic_id: str, user_id: str | None = None, db: Session | None = None) -> Path:
    """
    Validate that a comic exists in storage or database and is owned by the requested user.
    """
    valid_id = validate_comic_id(comic_id)
    comic_json_path = check_comic_exists(valid_id, db=db)
    if user_id is not None:
        owner_id = get_comic_user_id(valid_id, db=db)
        if owner_id is not None and owner_id != user_id:
            raise HTTPException(
                status_code=404,
                detail="Comic not found"
            )
    return comic_json_path


def _handle_ask_question(question: str, comic_id: str, user_id: str | None = None) -> dict:
    """
    Common helper for answering a comic question with access verification.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, user_id)

    try:
        result = answer_question(
            question=question,
            comic_id=valid_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while answering the question."
        )


# ============================================================
# Upload Comic Endpoint
# ============================================================

@router.post("/upload", response_model=ComicUploadResponse, summary="Upload and ingest a comic")
async def upload_comic(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Uploads a comic file (CBR, CBZ, PDF, JPG, PNG, WEBP), extracts pages immediately,
    saves initial metadata with status='processing' scoped to current_user,
    and dispatches AI analysis to background.
    """
    t_upload_start = time.perf_counter()

    if not file or not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported comic format '{extension}'. "
                "Supported formats: CBR, CBZ, PDF, JPG, JPEG, PNG, WEBP"
            )
        )

    # Phase A: Reading uploaded file bytes
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({len(contents) / (1024 * 1024):.1f} MB) exceeds the maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB. Please upload a smaller file."
        )

    comic_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{comic_id}{extension}"

    # Phase B: Writing uploaded file to disk
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file."
        )

    # Phase C: Page extraction (fast, ~0.5 - 2s) with dual-resolution thumbnail creation
    try:
        if extension == ".cbr":
            pages = extract_cbr(str(file_path), comic_id)
        elif extension == ".cbz":
            pages = extract_cbz(str(file_path), comic_id)
        elif extension == ".pdf":
            pages = extract_pdf(str(file_path), comic_id)
        elif extension in ALLOWED_EXTENSIONS:
            pages = extract_image(str(file_path), comic_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported comic format"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Comic extraction failed. The file may be corrupted or invalid."
        )

    # Phase D: Initial page placeholders with dual resolution paths
    comic_name = Path(file.filename).stem
    initial_pages = []
    for idx, p in enumerate(pages):
        page_num = p.get("page_number", idx + 1)
        page_fn = p.get("filename", f"page_{page_num:03d}.jpg")
        page_img = p.get("image_path", "")
        page_thumb = p.get("thumbnail_path", f"storage/comics/{comic_id}/thumbnails/thumb_p{page_num:03d}.jpg")
        img_storage = f"user/{current_user.id}/comics/{comic_id}/pages/{page_fn}"
        thumb_storage = f"user/{current_user.id}/comics/{comic_id}/thumbnails/thumb_p{page_num:03d}.jpg"

        initial_pages.append({
            "page_number": page_num,
            "filename": page_fn,
            "image_path": page_img,
            "thumbnail_path": page_thumb,
            "image_storage_path": img_storage,
            "thumbnail_storage_path": thumb_storage,
            "analysis": {
                "page_summary": "",
                "panels_detected": 0,
                "text": {
                    "full_text": "",
                    "dialogue_and_narration": [],
                    "sound_effects": [],
                    "signs_and_labels": []
                },
                "visual_description": {
                    "characters": [],
                    "actions": [],
                    "environment": "",
                    "objects": [],
                    "background": "",
                    "other_details": ""
                }
            },
            "metadata": {
                "page_number": page_num,
                "has_text": False
            },
            "status": "processing"
        })

    # Synchronously save initial comic to database immediately within upload transaction
    from app.services.storage import save_comic_to_db
    save_comic_to_db(
        db=db,
        comic_id=comic_id,
        comic_name=comic_name,
        source_format=extension.replace(".", ""),
        pages=initial_pages,
        status="processing",
        total_pages=len(pages),
        user_id=current_user.id
    )

    # Also save initial backup comic.json
    save_comic_json(
        comic_id=comic_id,
        comic_name=comic_name,
        source_format=extension.replace(".", ""),
        pages=initial_pages,
        status="processing",
        total_pages=len(pages),
        user_id=current_user.id
    )

    # Immediately sync extracted page images, thumbnails, and original file to Supabase Storage in parallel
    import threading
    if current_user and current_user.id:
        threading.Thread(
            target=sync_comic_assets_to_supabase,
            args=(comic_id, current_user.id, pages, file_path),
            daemon=True
        ).start()

    # Phase E: Dispatch background analysis in detached daemon thread
    thread = threading.Thread(
        target=run_background_comic_analysis,
        args=(comic_id, pages, comic_name, extension.replace(".", ""), initial_pages, current_user.id),
        daemon=True
    )
    thread.start()

    t_total = time.perf_counter() - t_upload_start
    logger.info(
        "[PERF] Comic %s upload & extraction complete in %.2fs. Processing %d pages in background.",
        comic_id,
        t_total,
        len(pages)
    )

    return {
        "message": "Comic uploaded and processing in background",
        "comic_id": comic_id,
        "filename": file.filename,
        "format": extension,
        "total_pages": len(pages),
        "status": "processing",
        "analyzed_pages": 0,
        "successful_pages": 0,
        "failed_pages": 0,
        "json_path": f"storage/comics/{comic_id}/comic.json",
        "rag_ingested": False,
        "rag_chunks_stored": 0,
        "rag_error": None
    }


# ============================================================
# Comic Processing Status Endpoint
# ============================================================

@router.get("/{comic_id}/status", response_model=ComicStatusResponse, summary="Get comic processing status and progress")
async def get_comic_status(
    comic_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Returns the real-time processing status and analyzed pages count of a comic owned by the user.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    comic_data = get_comic_data(valid_id, db=db)
    if not comic_data:
        raise HTTPException(
            status_code=404,
            detail="Comic not found"
        )

    meta = comic_data.get("comic", {})
    pages = comic_data.get("pages", [])
    status = meta.get("status", "completed")
    successful = sum(1 for p in pages if p.get("status") == "success")
    failed = sum(1 for p in pages if p.get("status") == "error")
    analyzed = successful + failed
    total = meta.get("total_pages", len(pages))

    return {
        "comic_id": valid_id,
        "title": meta.get("name", "Untitled Comic"),
        "status": status,
        "total_pages": total,
        "analyzed_pages": analyzed if status != "completed" else total,
        "successful_pages": successful if status != "completed" else meta.get("successful_pages", successful),
        "failed_pages": failed if status != "completed" else meta.get("failed_pages", failed),
        "rag_ingested": status == "completed"
    }


# ============================================================
# Ask Question Endpoints
# ============================================================

@router.post("/{comic_id}/ask", response_model=QuestionResponse, summary="Ask question about a specific comic")
async def ask_comic_question(
    comic_id: str,
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Answers a question grounded strictly in the context of the specified comic owned by user.
    """
    return _handle_ask_question(
        question=request.question,
        comic_id=comic_id,
        user_id=current_user.id
    )


@router.post("/ask", response_model=QuestionResponse, summary="Ask question with comic_id in request body")
async def ask_question_generic(
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Answers a question where comic_id is specified in the request body.
    """
    if not request.comic_id or not request.comic_id.strip():
        raise HTTPException(
            status_code=400,
            detail="comic_id is required"
        )
    return _handle_ask_question(
        question=request.question,
        comic_id=request.comic_id,
        user_id=current_user.id
    )


# ============================================================
# Get Comic Details Endpoint
# ============================================================

@router.get("/{comic_id}", summary="Get metadata and page details for a comic")
async def get_comic_details(
    comic_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the parsed metadata and analyzed pages for a given comic_id.
    Enriches page metadata with time-limited signed Supabase Storage URLs for direct CDN delivery.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    comic_data = get_comic_data(valid_id, db=db)
    if not comic_data:
        raise HTTPException(
            status_code=404,
            detail="Comic not found"
        )

    owner_id = current_user.id
    pages = comic_data.get("pages", [])

    # 1. Collect storage paths using DB fields as source of truth
    paths_to_sign = []
    page_path_map = []  # list of (page_dict, img_storage_path, thumb_storage_path)

    for p in pages:
        pnum = int(p.get("page_number", 1))
        fname = p.get("filename") or f"page_{pnum:03d}.jpg"
        img_spath = p.get("image_storage_path") or p.get("storage_path") or f"user/{owner_id}/comics/{valid_id}/pages/{fname}"
        thumb_spath = p.get("thumbnail_storage_path") or f"user/{owner_id}/comics/{valid_id}/thumbnails/thumb_p{pnum:03d}.jpg"

        paths_to_sign.append(img_spath)
        paths_to_sign.append(thumb_spath)
        page_path_map.append((p, img_spath, thumb_spath))

    # 2. Batch request signed URLs from Supabase Storage (1 hour expiry)
    signed_urls_map: dict[str, str | None] = {}
    if is_supabase_storage_enabled() and paths_to_sign:
        try:
            signed_urls_map = get_signed_storage_urls(paths_to_sign, expires_in=3600)
        except Exception as e:
            logger.warning("[IMAGE DELIVERY] Batch signed URL generation error for %s: %s", valid_id, str(e))

    # 3. Attach signed URLs to page metadata
    for p, img_spath, thumb_spath in page_path_map:
        thumb_url = signed_urls_map.get(thumb_spath) if signed_urls_map else None
        img_url = signed_urls_map.get(img_spath) if signed_urls_map else None

        if not thumb_url:
            logger.info("[IMAGE DELIVERY] Missing storage object for thumbnail: %s", thumb_spath)
        if not img_url:
            logger.info("[IMAGE DELIVERY] Missing storage object for image: %s", img_spath)

        p["thumbnail_url"] = thumb_url
        p["image_url"] = img_url

    logger.info(
        "[IMAGE DELIVERY] Generated signed URLs for comic %s: %d pages",
        valid_id,
        len(pages)
    )

    return comic_data


# ============================================================
# Get Comic Page Image Endpoint
# ============================================================

@router.get("/{comic_id}/pages/{page_number}/image", summary="Get comic page image")
async def get_comic_page_image(
    comic_id: str,
    page_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Returns the physical comic page image for the specified comic_id and page_number.
    Serves directly from local disk cache, or downloads the exact recorded storage path from Supabase.
    """
    if page_number < 1:
        raise HTTPException(
            status_code=400,
            detail="page_number must be greater than or equal to 1"
        )

    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)
    owner_id = get_comic_user_id(valid_id, db=db) or current_user.id

    pages_dir = (Path(COMICS_DIR) / valid_id / "pages").resolve()
    pages_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch exact ComicPage record from PostgreSQL
    try:
        db_page = (
            db.query(ComicPage)
            .filter(ComicPage.comic_id == valid_id, ComicPage.page_number == page_number)
            .first()
        )
        exact_storage_key = (
            db_page.image_storage_path
            if db_page and db_page.image_storage_path
            else None
        )
        page_filename = db_page.filename if db_page and db_page.filename else f"page_{page_number:03d}.jpg"
    finally:
        db.close()

    image_file_path = pages_dir / page_filename

    # Fast path: Serve directly from local disk
    if image_file_path.exists() and image_file_path.is_file() and image_file_path.stat().st_size > 0:
        media_type, _ = mimetypes.guess_type(str(image_file_path))
        return FileResponse(
            str(image_file_path),
            media_type=media_type or "image/jpeg",
            filename=image_file_path.name,
            headers={
                "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
            }
        )

    # Secondary check: Check alternative standard names on disk without guessing loop
    for alt_name in [f"page_{page_number:03d}.jpg", f"page_{page_number:03d}.png", f"page_{page_number:03d}.webp"]:
        alt_path = pages_dir / alt_name
        if alt_path.exists() and alt_path.is_file() and alt_path.stat().st_size > 0:
            media_type, _ = mimetypes.guess_type(str(alt_path))
            return FileResponse(
                str(alt_path),
                media_type=media_type or "image/jpeg",
                filename=alt_path.name,
                headers={
                    "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
                }
            )

    # 2. If missing from local disk, download exact storage path from Supabase Storage (single direct call)
    if is_supabase_storage_enabled():
        storage_key = exact_storage_key or f"user/{owner_id}/comics/{valid_id}/pages/{page_filename}"
        downloaded_bytes = download_bytes_from_storage(storage_key)
        if downloaded_bytes:
            image_file_path.write_bytes(downloaded_bytes)
            media_type, _ = mimetypes.guess_type(str(image_file_path))
            return FileResponse(
                str(image_file_path),
                media_type=media_type or "image/jpeg",
                filename=image_file_path.name,
                headers={
                    "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
                }
            )

    raise HTTPException(
        status_code=404,
        detail=f"Page {page_number} image not found on disk or storage"
    )


# ============================================================
# Get Comic Page Thumbnail Endpoint
# ============================================================

@router.get("/{comic_id}/pages/{page_number}/thumbnail", summary="Get lightweight comic page thumbnail")
async def get_comic_page_thumbnail(
    comic_id: str,
    page_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Returns a lightweight, cached thumbnail image (~200-400px) for preview and sidebar navigation.
    Serves from disk cache, downloads exact thumbnail from Supabase, or generates on-the-fly.
    """
    if page_number < 1:
        raise HTTPException(
            status_code=400,
            detail="page_number must be greater than or equal to 1"
        )

    valid_id = validate_comic_id(comic_id)
    thumb_dir = (Path(COMICS_DIR) / valid_id / "thumbnails").resolve()
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_file = thumb_dir / f"thumb_p{page_number:03d}.jpg"

    # Fast path 1: Serve already-cached thumbnail directly (< 1ms) without DB query
    if thumb_file.exists() and thumb_file.stat().st_size > 0:
        db.close()
        return FileResponse(
            str(thumb_file),
            media_type="image/jpeg",
            filename=f"thumb_{page_number}.jpg",
            headers={
                "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
            }
        )

    try:
        check_comic_access(valid_id, current_user.id, db=db)
        owner_id = get_comic_user_id(valid_id, db=db) or current_user.id

        # 1. Fetch exact ComicPage record
        db_page = (
            db.query(ComicPage)
            .filter(ComicPage.comic_id == valid_id, ComicPage.page_number == page_number)
            .first()
        )
        exact_thumb_storage = (
            db_page.thumbnail_storage_path
            if db_page and db_page.thumbnail_storage_path
            else None
        )
        page_filename = db_page.filename if db_page and db_page.filename else f"page_{page_number:03d}.jpg"
    finally:
        db.close()

    # Fast path 2: Check if source page exists on local disk and generate thumbnail on-the-fly
    pages_dir = (Path(COMICS_DIR) / valid_id / "pages").resolve()
    source_img_path = pages_dir / page_filename

    if not source_img_path.exists():
        for alt_name in [f"page_{page_number:03d}.jpg", f"page_{page_number:03d}.png", f"page_{page_number:03d}.webp"]:
            alt_p = pages_dir / alt_name
            if alt_p.exists() and alt_p.stat().st_size > 0:
                source_img_path = alt_p
                break

    if source_img_path.exists() and source_img_path.is_file() and source_img_path.stat().st_size > 0:
        generated_thumb = ensure_page_thumbnail(valid_id, page_number, source_img_path)
        if generated_thumb and generated_thumb.exists() and generated_thumb.stat().st_size > 0:
            return FileResponse(
                str(generated_thumb),
                media_type="image/jpeg",
                filename=f"thumb_{page_number}.jpg",
                headers={
                    "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
                }
            )

    # Fast path 3: Supabase Storage single direct download for exact thumbnail path
    if is_supabase_storage_enabled():
        exact_thumb_key = exact_thumb_storage or f"user/{owner_id}/comics/{valid_id}/thumbnails/thumb_p{page_number:03d}.jpg"
        thumb_bytes = download_bytes_from_storage(exact_thumb_key)
        if thumb_bytes:
            thumb_file.write_bytes(thumb_bytes)
            return FileResponse(
                str(thumb_file),
                media_type="image/jpeg",
                filename=f"thumb_{page_number}.jpg",
                headers={
                    "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
                }
            )

    raise HTTPException(
        status_code=404,
        detail=f"Thumbnail for page {page_number} not found"
    )


# ============================================================
# List All Comics Endpoint
# ============================================================

@router.get("", response_model=list[ComicListItem], summary="List all ingested comics")
async def get_all_comics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Returns the list of all ingested comics owned by the currently authenticated user.
    """
    return list_all_comics(user_id=current_user.id, db=db)


# ============================================================
# Delete Comic Endpoint
# ============================================================

@router.delete("/{comic_id}", response_model=ComicDeleteResponse, summary="Delete a comic and all associated resources")
async def delete_comic(
    comic_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Deletes all persistent resources associated with a comic owned by current_user:
    - Comic database record (cascading to pages, conversations, messages)
    - Remote Supabase Storage files
    - Local disk cache and extracted images
    - ChromaDB vector store chunks
    Leaves all other comics unaffected.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    try:
        # 1. Clean up database records and physical files
        delete_comic_storage(valid_id, user_id=current_user.id, db=db)

        # 2. Clean up ChromaDB chunks for this comic only
        delete_chunks_by_comic_id(valid_id)

        # 3. Clean up persistent chat conversations for this comic
        delete_conversations_by_comic_id(valid_id, user_id=current_user.id, db=db)

        return {
            "message": "Comic deleted successfully",
            "comic_id": valid_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete comic %s: %s", valid_id, str(e))
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the comic."
        )


# ============================================================
# Get / Initialize Comic Conversation Endpoint
# ============================================================

@router.get("/{comic_id}/conversation", response_model=ConversationDetailResponse, summary="Get or create conversation session for a comic")
async def get_comic_conversation(
    comic_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the existing conversation record for the specified comic_id,
    or creates a new persistent conversation session if none exists yet.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    try:
        record = get_or_create_comic_conversation(valid_id, user_id=current_user.id, db=db)
        return record
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation for comic."
        )