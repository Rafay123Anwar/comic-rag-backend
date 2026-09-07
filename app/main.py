import threading
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import CORS_ORIGINS, COMICS_DIR
from app.core.database import init_db, SessionLocal
from app.core.logging import logger
from app.models.comic import Comic, ComicPage
from app.routes.auth import router as auth_router
from app.routes.comic import (
    ask_question_generic,
    router as comic_router,
    run_background_comic_analysis,
)
from app.routes.conversation import router as conversation_router
from app.schemas.comic import (
    QuestionRequest,
    QuestionResponse,
)


def _recover_stuck_comics():
    """
    On startup, find any comics stuck in 'processing' status (from a previous
    server crash or restart) and re-dispatch their background analysis thread.

    This uses the local comic.json as the source of truth for page image paths
    since the DB rows may not have image_path (only image_url / storage_path).
    """
    db = SessionLocal()
    try:
        stuck_comics = (
            db.query(Comic)
            .filter(Comic.status == "processing")
            .all()
        )

        if not stuck_comics:
            logger.info("[STARTUP] No stuck comics found. All clear.")
            return

        logger.info(
            "[STARTUP] Found %d comic(s) stuck in 'processing'. Attempting recovery...",
            len(stuck_comics),
        )

        for comic in stuck_comics:
            comic_id = str(comic.id)
            comic_dir = Path(COMICS_DIR) / comic_id
            json_path = comic_dir / "comic.json"

            if not json_path.exists():
                logger.warning(
                    "[STARTUP RECOVERY] Skipping comic %s — comic.json not found at %s",
                    comic_id,
                    json_path,
                )
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    comic_data = json.load(f)
            except Exception as e:
                logger.warning(
                    "[STARTUP RECOVERY] Skipping comic %s — failed to read comic.json: %s",
                    comic_id,
                    e,
                )
                continue

            raw_pages = comic_data.get("pages", [])
            comic_name = comic_data.get("comic_name", comic.title or "Unknown")
            source_format = comic_data.get("source_format", "cbz")
            user_id = comic_data.get("user_id") or str(comic.user_id) if comic.user_id else None

            # Only recover if at least one page image exists locally
            recoverable_pages = []
            for p in raw_pages:
                img_path = p.get("image_path", "")
                if img_path and Path(img_path).exists():
                    recoverable_pages.append(p)

            if not recoverable_pages:
                logger.warning(
                    "[STARTUP RECOVERY] Skipping comic %s — no local page images found (may have been on a different machine).",
                    comic_id,
                )
                continue

            logger.info(
                "[STARTUP RECOVERY] Recovering comic %s (%s) — %d/%d pages have local images. Re-dispatching analysis...",
                comic_id,
                comic_name,
                len(recoverable_pages),
                len(raw_pages),
            )

            thread = threading.Thread(
                target=run_background_comic_analysis,
                args=(
                    comic_id,
                    raw_pages,          # full page list (image_path from json)
                    comic_name,
                    source_format,
                    None,               # initial_pages — already saved to DB
                    user_id,
                    None,               # original_file_path — not needed for recovery
                ),
                daemon=True,
            )
            thread.start()
            logger.info("[STARTUP RECOVERY] Recovery thread started for comic %s.", comic_id)

    except Exception as e:
        logger.exception("[STARTUP RECOVERY] Unexpected error during stuck-comic recovery: %s", e)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    init_db()

    # Re-dispatch analysis for any comics that were interrupted by a server restart
    recovery_thread = threading.Thread(
        target=_recover_stuck_comics,
        daemon=True,
        name="startup-comic-recovery",
    )
    recovery_thread.start()

    # Clean up any leftover extracted folders or uploaded archives from completed comics to reclaim disk space
    from app.services.storage import cleanup_orphaned_storage
    cleanup_thread = threading.Thread(
        target=cleanup_orphaned_storage,
        daemon=True,
        name="startup-storage-cleanup",
    )
    cleanup_thread.start()

    yield


app = FastAPI(
    title="Comic RAG API",
    description="Comic QA, multimodal ingestion, conversational chat memory, and authenticated user backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients (configurable via CORS_ORIGINS in .env)
raw_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
cors_origins = ["*"] if "*" in raw_origins or not raw_origins else raw_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None)
        )
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()}
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."}
    )


@app.get("/", tags=["Health"])
async def root():
    return {"message": "Comic RAG API is running", "status": "ok"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# Generic ask endpoint at root level
@app.post("/ask", response_model=QuestionResponse, tags=["Comics"], summary="Ask question about a comic")
async def root_ask_question(request: QuestionRequest):
    return await ask_question_generic(request)


app.include_router(
    auth_router,
    prefix="/auth",
    tags=["Auth"]
)

app.include_router(
    comic_router,
    prefix="/comics",
    tags=["Comics"]
)

app.include_router(
    comic_router,
    prefix="/api/comics",
    tags=["Comics"]
)

app.include_router(
    conversation_router,
    prefix="/conversations",
    tags=["Conversations"]
)