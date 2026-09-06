"""
Comic Routes

API endpoints for comic upload, metadata retrieval, and question-answering.
"""
import asyncio
import concurrent.futures
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
    upload_comic_assets_immediately,
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
    user_id: str | None = None,
    original_file_path: Path | None = None
):
    """
    Executes the optimized concurrent background comic processing pipeline:
      1. Launch Supabase Storage asset upload AND EdenAI multimodal visual analysis CONCURRENTLY.
      2. AI analysis begins immediately at t=0 since local page images are already extracted on disk.
      3. As EdenAI analyzes each page, incremental analysis results are persisted to DB/json and ingested into RAG.
      4. As Supabase uploads page images/thumbnails and fetches signed CDN URLs, URLs are persisted to DB/json.
      5. Full comic chunking and ChromaDB / vector store ingestion runs when analysis completes.
      6. Finally, updates the database status to 'completed'.
    """
    with _active_comic_lock:
        if comic_id in _active_comic_processing:
            logger.warning("[BACKGROUND] Background analysis already running for comic %s. Skipping duplicate dispatch.", comic_id)
            return
        _active_comic_processing.add(comic_id)

    t0 = time.perf_counter()
    logger.info("[BACKGROUND] Starting concurrent comic pipeline for comic %s (%d pages)...", comic_id, len(pages))

    try:
        raw_pages = initial_pages or pages
        current_pages = [dict(p) for p in raw_pages]
        state_lock = threading.Lock()

        # Filter pages that genuinely need AI analysis (skip already successfully analyzed pages)
        pages_to_analyze = []
        for p in pages:
            p_status = p.get("status")
            p_analysis = p.get("analysis")
            has_summary = bool(p_analysis and p_analysis.get("page_summary"))
            has_text = bool(p_analysis and isinstance(p_analysis.get("text"), dict) and p_analysis["text"].get("full_text"))
            if p_status == "success" and (has_summary or has_text):
                logger.info("[BACKGROUND] Page %s already has valid AI analysis. Skipping duplicate AI call.", p.get("page_number"))
            else:
                pages_to_analyze.append(p)

        logger.info(
            "[BACKGROUND] Pipeline dispatch: comic=%s total=%d pages, to_analyze=%d pages",
            comic_id,
            len(pages),
            len(pages_to_analyze)
        )

        def handle_page_analyzed(page_result: dict, completed_count: int, total_count: int):
            page_num = int(page_result.get("page_number", 1))
            logger.info("[PAGE PIPELINE] START page=%d", page_num)

            with state_lock:
                try:
                    # Retain any storage paths and signed CDN URLs if already populated by upload task
                    for p in current_pages:
                        if int(p.get("page_number", 1)) == page_num:
                            if p.get("image_storage_path") and not page_result.get("image_storage_path"):
                                page_result["image_storage_path"] = p.get("image_storage_path")
                            if p.get("thumbnail_storage_path") and not page_result.get("thumbnail_storage_path"):
                                page_result["thumbnail_storage_path"] = p.get("thumbnail_storage_path")
                            if p.get("image_url") and not page_result.get("image_url"):
                                page_result["image_url"] = p.get("image_url")
                            if p.get("thumbnail_url") and not page_result.get("thumbnail_url"):
                                page_result["thumbnail_url"] = p.get("thumbnail_url")
                            break

                    if page_result.get("status") == "success":
                        logger.info("[PAGE PIPELINE] AI ANALYSIS SUCCESS page=%d", page_num)
                    else:
                        logger.warning("[PAGE PIPELINE] AI ANALYSIS FAILED page=%d error=%s", page_num, page_result.get("error", "Unknown"))

                    # Update page in current_pages state
                    for idx, p in enumerate(current_pages):
                        if int(p.get("page_number", 1)) == page_num:
                            current_pages[idx] = page_result
                            break

                    # Save updated page analysis to database & json
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
                    logger.exception("[PAGE PIPELINE] FAILED page=%d error=%s", page_num, str(page_err))
                    page_result["status"] = "error"
                    page_result["error"] = str(page_err)
                    for idx, p in enumerate(current_pages):
                        if int(p.get("page_number", 1)) == page_num:
                            current_pages[idx] = page_result
                            break

            # Mark Progress
            with state_lock:
                processed_count = sum(1 for p in current_pages if p.get("status") in ("success", "error"))
            progress_pct = (processed_count / max(1, len(pages))) * 100
            logger.info("[PAGE PIPELINE] COMPLETE page=%d (%d/%d processed, progress=%.1f%%)", page_num, processed_count, len(pages), progress_pct)

            # Non-blocking per-page ChromaDB / pgvector Ingestion
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

        # -------------------------------------------------------------
        # STEP 1 & 2: Concurrently execute Cloudinary Upload & EdenAI Analysis
        # -------------------------------------------------------------
        logger.info("[BACKGROUND] Step 1: Launching concurrent Cloudinary asset upload and EdenAI analysis for comic %s...", comic_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Future 1: Cloudinary storage upload (images, thumbnails, direct CDN URLs)
            upload_future = executor.submit(
                upload_comic_assets_immediately,
                comic_id=comic_id,
                user_id=user_id,
                pages=raw_pages,
                original_file_path=original_file_path,
                max_workers=8
            )

            # Future 2: EdenAI visual analysis (starts at t=0 on local image files)
            analysis_future = None
            if pages_to_analyze:
                analysis_future = executor.submit(
                    analyze_pages,
                    pages=pages_to_analyze,
                    max_workers=MAX_AI_WORKERS,
                    max_retries=MAX_AI_RETRIES,
                    on_page_complete=handle_page_analyzed
                )

            # Hook: When upload finishes, update current_pages with Cloudinary CDN URLs
            def on_upload_completed(f):
                try:
                    enriched = f.result()
                    with state_lock:
                        for ep in enriched:
                            pnum = int(ep.get("page_number", 1))
                            for idx, cp in enumerate(current_pages):
                                if int(cp.get("page_number", 1)) == pnum:
                                    if ep.get("image_url"):
                                        cp["image_url"] = ep["image_url"]
                                    if ep.get("thumbnail_url"):
                                        cp["thumbnail_url"] = ep["thumbnail_url"]
                                    if ep.get("image_storage_path"):
                                        cp["image_storage_path"] = ep["image_storage_path"]
                                    if ep.get("thumbnail_storage_path"):
                                        cp["thumbnail_storage_path"] = ep["thumbnail_storage_path"]
                                    break
                        save_comic_json(
                            comic_id=comic_id,
                            comic_name=comic_name,
                            source_format=source_format,
                            pages=current_pages,
                            status="processing",
                            total_pages=len(pages),
                            user_id=user_id
                        )
                    logger.info("[BACKGROUND] Cloudinary asset upload complete & synced to DB for comic %s", comic_id)
                except Exception as up_err:
                    logger.warning("[BACKGROUND] Cloudinary asset upload error for comic %s: %s (non-fatal)", comic_id, str(up_err))

            upload_future.add_done_callback(on_upload_completed)

            # Await completion of AI analysis
            if analysis_future:
                analyzed_pages = analysis_future.result()
                with state_lock:
                    for res in analyzed_pages:
                        pnum = int(res.get("page_number", 1))
                        for idx, p in enumerate(current_pages):
                            if int(p.get("page_number", 1)) == pnum:
                                merged = dict(p)
                                merged.update(res)
                                if p.get("image_url"):
                                    merged["image_url"] = p.get("image_url")
                                if p.get("thumbnail_url"):
                                    merged["thumbnail_url"] = p.get("thumbnail_url")
                                if p.get("image_storage_path"):
                                    merged["image_storage_path"] = p.get("image_storage_path")
                                if p.get("thumbnail_storage_path"):
                                    merged["thumbnail_storage_path"] = p.get("thumbnail_storage_path")
                                current_pages[idx] = merged
                                break

            # Ensure Supabase upload has completed as well
            try:
                upload_future.result()
            except Exception as up_exc:
                logger.warning("[BACKGROUND] Supabase upload future returned error for comic %s: %s (non-fatal)", comic_id, str(up_exc))

        # -------------------------------------------------------------
        # STEP 3: Full Comic Chunking & ChromaDB / pgvector Ingestion
        # -------------------------------------------------------------
        json_file_path = COMICS_DIR / comic_id / "comic.json"
        if json_file_path.exists():
            try:
                logger.info("[BACKGROUND] Step 3: Chunking and ingesting full comic into ChromaDB for comic %s...", comic_id)
                ingest_comic_to_rag(str(json_file_path))
            except Exception as rag_err:
                logger.warning("[BACKGROUND] Full comic RAG ingestion error for %s: %s (non-fatal)", comic_id, str(rag_err))

        # -------------------------------------------------------------
        # STEP 4: Finally, update database status to 'completed'
        # -------------------------------------------------------------
        logger.info("[BACKGROUND] Step 4: Updating database status to 'completed' for comic %s...", comic_id)
        db = SessionLocal()
        try:
            db_pages = db.query(ComicPage).filter(ComicPage.comic_id == comic_id).all()
            succ_count = sum(1 for p in db_pages if p.status == "success")
            fail_count = sum(1 for p in db_pages if p.status == "error")
            proc_count = succ_count + fail_count
            tot_count = len(pages)

            final_status = "failed" if (succ_count == 0 and fail_count > 0) else "completed"

            comic_row = db.query(Comic).filter(Comic.id == comic_id).first()
            if comic_row:
                comic_row.status = final_status
                comic_row.analyzed_pages = proc_count
                comic_row.successful_pages = succ_count
                comic_row.failed_pages = fail_count
                db.commit()

            save_comic_json(
                comic_id=comic_id,
                comic_name=comic_name,
                source_format=source_format,
                pages=current_pages,
                status=final_status,
                total_pages=tot_count,
                user_id=user_id
            )

            logger.info(
                "[COMIC PIPELINE] FINISHED comic=%s total=%d successful=%d failed=%d processed=%d status=%s",
                comic_id,
                tot_count,
                succ_count,
                fail_count,
                proc_count,
                final_status
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
            print("cbr Extracted")
            pages = extract_cbr(str(file_path), comic_id)
        elif extension == ".cbz":
            print("cbz Extracted")
            pages = extract_cbz(str(file_path), comic_id)
        elif extension == ".pdf":
            print("pdf Extracted")
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
    except Exception as e:
        logger.error(
            "[UPLOAD] Comic extraction failed for comic_id=%s file=%s (%s): %s",
            comic_id,
            file.filename,
            extension,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Comic extraction failed: {str(e)}"
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

    # Phase E: Dispatch reordered background pipeline in detached daemon thread
    # Step 1: Immediately upload images and thumbnails to Supabase Storage
    # Step 2: Save initial page records to DB with image_url and thumbnail_url populated, status='processing'
    # Step 3: Run EdenAI analysis, chunking, and ChromaDB ingestion
    # Step 4: Update database status to 'completed'
    import threading
    thread = threading.Thread(
        target=run_background_comic_analysis,
        args=(comic_id, pages, comic_name, extension.replace(".", ""), initial_pages, current_user.id, file_path),
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

    # Expire cached session objects so background thread updates to ComicPage are immediately visible
    db.expire_all()

    comic_obj = db.query(Comic).filter(Comic.id == valid_id).first()
    if not comic_obj:
        raise HTTPException(
            status_code=404,
            detail="Comic not found"
        )

    db_pages = list(comic_obj.pages)
    db_pages.sort(key=lambda p: p.page_number)
    successful = sum(1 for p in db_pages if p.status == "success")
    failed = sum(1 for p in db_pages if p.status == "error")
    analyzed = successful + failed
    total = comic_obj.total_pages or len(db_pages)
    status = comic_obj.status or "completed"

    owner_id = current_user.id
    paths_to_sign = []
    for p in db_pages:
        img_spath = p.image_storage_path or f"user/{owner_id}/comics/{valid_id}/pages/{p.filename or f'page_{p.page_number:03d}.jpg'}"
        thumb_spath = p.thumbnail_storage_path or f"user/{owner_id}/comics/{valid_id}/thumbnails/thumb_p{p.page_number:03d}.jpg"
        if (not p.thumbnail_url or p.thumbnail_url.startswith("/api/")) and thumb_spath:
            paths_to_sign.append(thumb_spath)
        if (not p.image_url or p.image_url.startswith("/api/")) and img_spath:
            paths_to_sign.append(img_spath)

    signed_urls_map = {}
    if is_supabase_storage_enabled() and paths_to_sign:
        try:
            signed_urls_map = get_signed_storage_urls(paths_to_sign, expires_in=3600)
        except Exception:
            pass

    # If any page is missing analysis_json in DB, check local comic.json as fallback
    fallback_pages_map = {}
    if any(p.status == "success" and not p.analysis_json for p in db_pages):
        try:
            local_json_file = COMICS_DIR / valid_id / "comic.json"
            if local_json_file.exists():
                with open(local_json_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for lpage in cdata.get("pages", []):
                        lp_num = lpage.get("page_number")
                        if lp_num and lpage.get("analysis"):
                            fallback_pages_map[lp_num] = lpage.get("analysis")
        except Exception:
            pass

    pages_list = []
    for p in db_pages:
        img_spath = p.image_storage_path or f"user/{owner_id}/comics/{valid_id}/pages/{p.filename or f'page_{p.page_number:03d}.jpg'}"
        thumb_spath = p.thumbnail_storage_path or f"user/{owner_id}/comics/{valid_id}/thumbnails/thumb_p{p.page_number:03d}.jpg"

        signed_thumb = signed_urls_map.get(thumb_spath) if signed_urls_map else None
        signed_img = signed_urls_map.get(img_spath) if signed_urls_map else None

        direct_thumb = p.thumbnail_url if (p.thumbnail_url and not p.thumbnail_url.startswith("/api/")) else None
        direct_img = p.image_url if (p.image_url and not p.image_url.startswith("/api/")) else None

        final_thumb = (
            direct_thumb
            or signed_thumb
            or direct_img
            or signed_img
            or p.thumbnail_url
            or f"/api/comics/{valid_id}/pages/{p.page_number}/thumbnail"
        )
        final_img = (
            direct_img
            or signed_img
            or p.image_url
            or f"/api/comics/{valid_id}/pages/{p.page_number}/image"
        )

        analysis = None
        if p.analysis_json:
            try:
                analysis = json.loads(p.analysis_json)
            except Exception:
                pass
        if analysis is None and p.page_number in fallback_pages_map:
            analysis = fallback_pages_map[p.page_number]

        pages_list.append({
            "page_number": p.page_number,
            "status": p.status,
            "thumbnail_url": final_thumb,
            "image_url": final_img,
            "analysis": analysis,
        })

    logger.info(
        "[STATUS] comic=%s status=%s pages_count=%d sample_thumb=%s",
        valid_id,
        status,
        len(pages_list),
        pages_list[0]["thumbnail_url"] if pages_list else None
    )

    return {
        "comic_id": valid_id,
        "title": comic_obj.title or "Untitled Comic",
        "status": status,
        "total_pages": total,
        "analyzed_pages": analyzed if status != "completed" else total,
        "successful_pages": successful if status != "completed" else (comic_obj.successful_pages or successful),
        "failed_pages": failed if status != "completed" else (comic_obj.failed_pages or failed),
        "rag_ingested": status == "completed",
        "pages": pages_list,
    }


# ============================================================
# Chat & Ask Question Endpoints (Unblocked Real-Time RAG)
# ============================================================

@router.post("/{comic_id}/chat", response_model=QuestionResponse, summary="Chat with comic in real-time during or after processing")
async def chat_with_comic(
    comic_id: str,
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Asynchronous chat endpoint for querying a comic in real-time.
    Unblocked during processing: searches whatever pages have been analyzed
    and ingested into ChromaDB so far without blocking the event loop.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    result = await asyncio.to_thread(
        answer_question,
        question=request.question,
        comic_id=valid_id,
        current_page=getattr(request, "current_page", None)
    )
    return result


@router.post("/{comic_id}/ask", response_model=QuestionResponse, summary="Ask question about a specific comic")
async def ask_comic_question(
    comic_id: str,
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Answers a question grounded in the context of the specified comic owned by user.
    Unblocked during processing: queries whatever pages are currently available.
    """
    valid_id = validate_comic_id(comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    result = await asyncio.to_thread(
        answer_question,
        question=request.question,
        comic_id=valid_id,
        current_page=getattr(request, "current_page", None)
    )
    return result


@router.post("/ask", response_model=QuestionResponse, summary="Ask question with comic_id in request body")
async def ask_question_generic(
    request: QuestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Answers a question where comic_id is specified in the request body.
    """
    if not request.comic_id or not request.comic_id.strip():
        raise HTTPException(
            status_code=400,
            detail="comic_id is required"
        )
    valid_id = validate_comic_id(request.comic_id)
    check_comic_access(valid_id, current_user.id, db=db)

    result = await asyncio.to_thread(
        answer_question,
        question=request.question,
        comic_id=valid_id,
        current_page=getattr(request, "current_page", None)
    )
    return result


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

    # Expire cached session objects so background thread updates to ComicPage are immediately visible
    db.expire_all()

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
        pnum = int(p.get("page_number", 1))
        thumb_url = signed_urls_map.get(thumb_spath) if signed_urls_map else None
        img_url = signed_urls_map.get(img_spath) if signed_urls_map else None

        if not thumb_url:
            thumb_url = p.get("thumbnail_url") or f"/api/comics/{valid_id}/pages/{pnum}/thumbnail"
        if not img_url:
            img_url = p.get("image_url") or f"/api/comics/{valid_id}/pages/{pnum}/image"

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