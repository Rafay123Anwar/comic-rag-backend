"""
Retriever Service

Performs semantic vector search against ChromaDB, filters by distance threshold,
and orders retrieved chunks deterministically (page_number ASC, chunk_index ASC, chunk_id ASC).
"""
from app.core.config import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_TOP_K
from app.services.embedding import generate_embedding
from app.services.vector_store import search_chunks


def retrieve_chunks(
    query: str,
    comic_id: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD
) -> list[dict]:
    """
    Retrieve relevant comic chunks for a given query and comic_id.

    Parameters:
        query: User question / query string.
        comic_id: Comic identifier for strict isolation.
        top_k: Maximum raw chunks to retrieve from vector store (default 5).
        distance_threshold: Maximum allowable cosine distance (defaults to 0.35).

    Returns:
        Deterministic list of chunk dictionaries.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not comic_id or not comic_id.strip():
        raise ValueError("comic_id is required for comic-scoped retrieval.")

    query = query.strip()
    comic_id = comic_id.strip()

    if top_k < 1:
        raise ValueError("top_k must be greater than 0.")

    # -----------------------------
    # Query Embedding
    # -----------------------------
    query_embedding = generate_embedding(query)

    # -----------------------------
    # Vector Search
    # -----------------------------
    results = search_chunks(
        query_embedding=query_embedding,
        comic_id=comic_id,
        top_k=top_k
    )

    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    # -----------------------------
    # Build Retrieved Chunks
    # -----------------------------
    retrieved_chunks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        chunk_id = ids[index] if index < len(ids) else None

        retrieved_chunks.append(
            {
                "chunk_id": chunk_id,
                "content": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    # -----------------------------
    # Distance Threshold Filtering
    # -----------------------------
    if distance_threshold is not None:
        final_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk.get("distance") is not None
            and chunk["distance"] <= distance_threshold
        ]
    else:
        final_chunks = retrieved_chunks

    # -----------------------------
    # Deterministic Ordering
    # -----------------------------
    # Order by page_number ASC, chunk_index ASC, then chunk_id as tie-breaker
    final_chunks.sort(key=lambda c: (
        c["metadata"].get("page_number", 0),
        c["metadata"].get("chunk_index", 0),
        c["metadata"].get("chunk_id", "")
    ))

    return final_chunks