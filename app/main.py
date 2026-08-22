from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.database import init_db
from app.routes.auth import router as auth_router
from app.routes.comic import (
    ask_question_generic,
    router as comic_router,
)
from app.routes.conversation import router as conversation_router
from app.schemas.comic import (
    QuestionRequest,
    QuestionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    init_db()
    yield


app = FastAPI(
    title="Comic RAG API",
    description="Comic QA, multimodal ingestion, conversational chat memory, and authenticated user backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    conversation_router,
    prefix="/conversations",
    tags=["Conversations"]
)