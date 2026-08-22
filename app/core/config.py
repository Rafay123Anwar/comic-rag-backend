import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Mistral API Configuration
# -----------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-2512")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-2512")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed-2312")

# -----------------------------
# Storage Paths Configuration
# -----------------------------
STORAGE_PATH = os.getenv("STORAGE_PATH", "storage")
STORAGE_DIR = Path(STORAGE_PATH)
UPLOADS_DIR = STORAGE_DIR / "uploads"
COMICS_DIR = STORAGE_DIR / "comics"
CONVERSATIONS_DIR = STORAGE_DIR / "conversations"
VECTOR_DB_DIR = STORAGE_DIR / "vector_db"
TEMP_DIR = STORAGE_DIR / "temp"

# -----------------------------
# Supabase & Database Config
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv(
    "SUPABASE_STORAGE_BUCKET",
    "Comic-Rag"
)

raw_db_url = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'comic_rag.db'}")
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
DATABASE_URL = raw_db_url

# -----------------------------
# Authentication Config
# -----------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "comic-rag-super-secret-jwt-key-production-change-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# -----------------------------
# RAG & Vector Store Configuration
# -----------------------------
COLLECTION_NAME = "comic_pages"
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "auto").lower()  # "auto", "supabase", "chroma"
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("DEFAULT_DISTANCE_THRESHOLD", "0.65"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "8"))
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# -----------------------------
# Conversation Memory Config
# -----------------------------
MAX_CONVERSATION_MESSAGES = int(
    os.getenv("MAX_CONVERSATION_MESSAGES", "10")
)

# -----------------------------
# Translation & OCR Config
# -----------------------------
USE_LIBRARY_TRANSLATION = os.getenv("USE_LIBRARY_TRANSLATION", "true").lower() in ("true", "1", "yes")
ENABLE_HYBRID_OCR = os.getenv("ENABLE_HYBRID_OCR", "false").lower() in ("true", "1", "yes")

# -----------------------------
# Analysis & Extraction Config
# -----------------------------
MAX_AI_WORKERS = int(os.getenv("MAX_AI_WORKERS", "2"))
MAX_AI_RETRIES = int(os.getenv("MAX_AI_RETRIES", "5"))
SEVEN_ZIP_PATH = os.getenv(
    "SEVEN_ZIP_PATH",
    r"C:\Program Files\7-Zip\7z.exe"
)

ALLOWED_EXTENSIONS = {
    ".cbr", ".cbz", ".pdf",
    ".jpg", ".jpeg", ".png", ".webp"
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# -----------------------------
# Validation
# -----------------------------
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not configured. "
        "Please create a .env file based on .env.example."
    )
