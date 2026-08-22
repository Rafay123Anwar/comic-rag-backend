import time

from mistralai.client import Mistral

from app.core.config import EMBEDDING_MODEL, MISTRAL_API_KEY
from app.core.logging import logger

# -----------------------------
# Mistral Client
# -----------------------------

client = Mistral(
    api_key=MISTRAL_API_KEY
)


# -----------------------------
# Multiple Text Embeddings
# -----------------------------

def create_embeddings(
    texts: list[str]
) -> list[list[float]]:
    """
    Generates batch vector embeddings for a list of text strings.
    """
    if not texts:
        return []

    t0 = time.perf_counter()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=texts
    )
    duration = time.perf_counter() - t0

    logger.info(
        "[PERF] Embeddings | inputs=%s | duration=%.2fs",
        len(texts),
        duration
    )

    return [
        item.embedding
        for item in response.data
    ]


# -----------------------------
# Single Text Embedding
# -----------------------------

def generate_embedding(
    text: str
) -> list[float]:
    """
    Generates a vector embedding for a single text query or document.
    """
    if not text or not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    embeddings = create_embeddings(
        [text]
    )

    return embeddings[0]