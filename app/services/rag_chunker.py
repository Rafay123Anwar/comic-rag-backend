"""
RAG Chunker Service

Splits comic page documents into deterministic, structured chunks for vector embedding.
Preserves page numbers, chunk indices, and metadata.

M-23: Replaced langchain_text_splitters.RecursiveCharacterTextSplitter with a
pure-Python stdlib-only equivalent that is behaviorally identical but has zero
import cost. The langchain import was the sole cause of ~25s chunking time.
"""
import re
import time

from app.core.logging import logger


# ---------------------------------------------------------------------------
# Pure-Python RecursiveCharacterTextSplitter (zero-dependency, stdlib-only)
#
# Faithfully reproduces the algorithm from langchain_text_splitters source:
#   - _split_text_with_regex (character.py)
#   - TextSplitter._merge_splits  (base.py)
#   - RecursiveCharacterTextSplitter._split_text (character.py)
#   - TextSplitter._join_docs (base.py)
#
# Configuration is fixed to match the project's only usage:
#   chunk_size=1000, chunk_overlap=150, is_separator_regex=False,
#   keep_separator=False, length_function=len, separators=["\n\n","\n",". "," ",""]
# ---------------------------------------------------------------------------


def _lc_split_text_with_regex(text: str, separator: str) -> list[str]:
    """
    Equivalent to langchain _split_text_with_regex(text, sep, keep_separator=False).
    """
    if separator:
        splits = re.split(separator, text)
    else:
        # empty separator → character-by-character
        splits = list(text)
    return [s for s in splits if s]


def _lc_join_docs(docs: list[str], separator: str) -> str | None:
    """
    Equivalent to langchain TextSplitter._join_docs.
    """
    text = separator.join(docs).strip()
    return text if text else None


def _lc_merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Equivalent to langchain TextSplitter._merge_splits.
    Merges short text pieces into chunks of at most chunk_size, with chunk_overlap
    characters of overlap between consecutive chunks.
    """
    sep_len = len(separator)
    docs: list[str] = []
    current_doc: list[str] = []
    total = 0

    for d in splits:
        d_len = len(d)
        # Would adding this piece overflow the chunk?
        if total + d_len + (sep_len if current_doc else 0) > chunk_size:
            if total > chunk_size:
                # Emit a warning (matching langchain behavior) but don't crash
                logger.warning(
                    "[PERF] Chunker | oversized piece: %d chars > chunk_size %d",
                    total,
                    chunk_size,
                )
            if current_doc:
                doc = _lc_join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                # Pop from front while we still exceed overlap
                while total > chunk_overlap or (
                    total + d_len + (sep_len if current_doc else 0) > chunk_size
                    and total > 0
                ):
                    total -= len(current_doc[0]) + (sep_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += d_len + (sep_len if len(current_doc) > 1 else 0)

    doc = _lc_join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


def _lc_split_text_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Equivalent to langchain RecursiveCharacterTextSplitter._split_text.
    is_separator_regex=False, keep_separator=False (project defaults).
    """
    final_chunks: list[str] = []

    # Find the first separator that is present in the text
    separator = separators[-1]
    new_separators: list[str] = []
    for i, s in enumerate(separators):
        sep_pattern = re.escape(s) if s else s
        if not s:
            # empty string separator → always matches
            separator = s
            break
        if re.search(sep_pattern, text):
            separator = s
            new_separators = separators[i + 1:]
            break

    sep_pattern = re.escape(separator) if separator else separator
    splits = _lc_split_text_with_regex(text, sep_pattern)

    good_splits: list[str] = []
    # When keep_separator=False the joining separator is the separator itself
    merge_sep = "" if not separator else separator

    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                merged = _lc_merge_splits(good_splits, merge_sep, chunk_size, chunk_overlap)
                final_chunks.extend(merged)
                good_splits = []
            if not new_separators:
                final_chunks.append(s)
            else:
                other = _lc_split_text_recursive(s, new_separators, chunk_size, chunk_overlap)
                final_chunks.extend(other)

    if good_splits:
        merged = _lc_merge_splits(good_splits, merge_sep, chunk_size, chunk_overlap)
        final_chunks.extend(merged)

    return final_chunks


def _split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Public entry point — mimics RecursiveCharacterTextSplitter.split_text()
    with the project's fixed separators.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _lc_split_text_recursive(text, separators, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Module-level cached splitter config
#
# Preserved from M-21: the cached-config sentinel avoids redundant parameter
# validation on repeated calls. The splitter itself is now free to "create"
# since it is just a dict lookup — no import, no heavy init.
# ---------------------------------------------------------------------------

_default_chunk_size: int | None = None
_default_chunk_overlap: int | None = None


def _get_default_splitter(chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Returns a callable that behaves like RecursiveCharacterTextSplitter.split_text()
    with the given chunk_size and chunk_overlap.

    Lazily caches parameters and returns a bound split function.
    Preserves fast startup times and prevents repeated parameter setup across chunking calls.
    """
    global _default_chunk_size, _default_chunk_overlap
    if _default_chunk_size is None:
        _default_chunk_size = chunk_size
        _default_chunk_overlap = chunk_overlap

    size = _default_chunk_size
    overlap = _default_chunk_overlap

    def split(text: str) -> list[str]:
        return _split_text(text, chunk_size=size, chunk_overlap=overlap)

    return split


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_rag_chunks(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Split comic page documents into deterministic text chunks.

    Args:
        documents: List of dicts, each with 'content' (str) and 'metadata' (dict).
        chunk_size: Target size in characters for each chunk.
        chunk_overlap: Overlap in characters between adjacent chunks.

    Returns:
        List of chunk dicts, each with 'chunk_id', 'content', and 'metadata'.
    """
    if not documents:
        return []

    t_start = time.perf_counter()

    # ---------------------------------
    # Reusable Splitter
    # ---------------------------------
    t0 = time.perf_counter()
    if chunk_size == 1000 and chunk_overlap == 150:
        split_fn = _get_default_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        _cs, _co = chunk_size, chunk_overlap

        def split_fn(text: str) -> list[str]:
            return _split_text(text, chunk_size=_cs, chunk_overlap=_co)

    t_splitter_init = time.perf_counter() - t0

    total_input_chars = sum(len(doc.get("content", "")) for doc in documents)

    logger.info(
        "[PERF] Chunker | documents=%d | input_chars=%d",
        len(documents),
        total_input_chars,
    )
    logger.info("[PERF] Splitter initialization: %.4fs", t_splitter_init)

    chunks: list[dict] = []

    # ---------------------------------
    # Process each page
    # ---------------------------------
    t0 = time.perf_counter()
    slowest_page = None
    slowest_duration = 0.0

    for document in documents:
        content = document.get("content", "")
        metadata = document.get("metadata", {})

        if not content.strip():
            continue

        comic_id = metadata.get("comic_id")
        page_number = metadata.get("page_number")

        # ---------------------------------
        # Split page
        # ---------------------------------
        t_doc = time.perf_counter()
        split_texts = split_fn(content)
        doc_duration = time.perf_counter() - t_doc

        if doc_duration > slowest_duration:
            slowest_duration = doc_duration
            slowest_page = (page_number, doc_duration, len(content))

        # ---------------------------------
        # Create chunks
        # ---------------------------------
        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = (
                f"{comic_id}"
                f"_page_{page_number}"
                f"_chunk_{chunk_index + 1}"
            )

            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index + 1,
                "total_page_chunks": len(split_texts),
                "chunk_id": chunk_id
            }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "metadata": chunk_metadata
                }
            )

    t_split_total = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    logger.info("[PERF] Document splitting: %.4fs", t_split_total)
    logger.info("[PERF] Total chunking: %.4fs", t_total)
    logger.info(
        "[PERF] Chunks generated: %d | avg_chars=%.0f | avg_chunks_per_doc=%.1f",
        len(chunks),
        total_input_chars / max(len(documents), 1),
        len(chunks) / max(len(documents), 1),
    )
    if slowest_page:
        logger.info(
            "[PERF] Slowest document: page=%s | duration=%.4fs | chars=%d",
            slowest_page[0],
            slowest_page[1],
            slowest_page[2],
        )

    return chunks