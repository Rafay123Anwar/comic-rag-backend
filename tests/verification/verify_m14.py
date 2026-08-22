"""
verify_m14.py -- M-14 Answer & Source Consistency Validation Verification Script

Deterministic test suite verifying the post-LLM validation layer, source integrity,
distance filtering, comic isolation, ordering preservation, and regression checks.
All console output is ASCII-only.

Usage
-----
    python -u verify_m14.py
"""

import sys
import os
import pathlib
import unittest.mock as mock

# ---------------------------------------------------------------------------
# Output Helpers (ASCII-only)
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "[PASS]" if condition else "[FAIL]"
    line = f"  {icon} {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return condition


# ---------------------------------------------------------------------------
# Import Application Modules
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

from app.services.query_normalizer import normalize_query
from app.services.retriever import (
    retrieve_chunks,
    DEFAULT_DISTANCE_THRESHOLD,
)
import app.services.rag_qa as rag_qa_mod
from app.services.rag_qa import (
    answer_question,
    validate_answer_and_sources,
    FALLBACK_ANSWER,
)
import app.services.llm as llm_mod


# ---------------------------------------------------------------------------
# Mock Data
# ---------------------------------------------------------------------------

COMIC_ID_VALID = "test-comic-001"
COMIC_ID_OTHER = "wrong-comic-999"

SAMPLE_CHUNKS = [
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        "content": "Victor speaks warmly about his mother's care and encouragement.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        },
        "distance": 0.20,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2",
        "content": "His mother told him he was destined for greatness.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 1,
            "chunk_index": 2,
            "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2",
        },
        "distance": 0.25,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1",
        "content": "Victor remembers how his mother protected him from harm.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 2,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1",
        },
        "distance": 0.30,
    },
]


# ===========================================================================
# Test A -- Normal Relevant Answer
# ===========================================================================

print("\n--- Test A: Normal relevant answer ---")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="Victor's mother always supported him."):

    res_a = answer_question("What did Victor say about his mother?", comic_id=COMIC_ID_VALID)

    check("A1: answer is preserved from LLM",
          res_a["answer"] == "Victor's mother always supported him.",
          f"answer: {res_a['answer']!r}")

    check("A2: sources are returned (non-empty)",
          len(res_a["sources"]) == 3,
          f"sources count: {len(res_a['sources'])}")

    expected_ids = [c["chunk_id"] for c in SAMPLE_CHUNKS]
    actual_ids = [s["chunk_id"] for s in res_a["sources"]]
    check("A3: source chunk_ids match retrieved chunks in exact order",
          actual_ids == expected_ids,
          f"actual: {actual_ids}")

    all_fields_present = all(
        s.get("comic_id") == COMIC_ID_VALID
        and s.get("page_number") is not None
        and s.get("chunk_id") is not None
        and s.get("chunk_index") is not None
        and s.get("distance") is not None
        for s in res_a["sources"]
    )
    check("A4: all sources contain required fields (comic_id, page_number, chunk_id, chunk_index, distance)",
          all_fields_present)

    check("A5: question is preserved in response",
          res_a["question"] == "What did Victor say about his mother?")

    check("A6: comic_id is preserved in response",
          res_a["comic_id"] == COMIC_ID_VALID)


# ===========================================================================
# Test B -- Empty LLM Answer
# ===========================================================================

print("\n--- Test B: Empty LLM answer ---")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value=""):

    res_b = answer_question("What did Victor say?", comic_id=COMIC_ID_VALID)

    check("B1: empty LLM answer falls back to standard message",
          res_b["answer"] == FALLBACK_ANSWER,
          f"answer: {res_b['answer']!r}")

    check("B2: sources list is empty on empty LLM answer",
          res_b["sources"] == [],
          f"sources: {res_b['sources']}")

    check("B3: question is preserved",
          res_b["question"] == "What did Victor say?")

    check("B4: comic_id is preserved",
          res_b["comic_id"] == COMIC_ID_VALID)


# ===========================================================================
# Test C -- Whitespace-Only LLM Answer
# ===========================================================================

print("\n--- Test C: Whitespace-only LLM answer ---")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="   \n\t  \r  "):

    res_c = answer_question("What did Victor say?", comic_id=COMIC_ID_VALID)

    check("C1: whitespace LLM answer falls back to standard message",
          res_c["answer"] == FALLBACK_ANSWER,
          f"answer: {res_c['answer']!r}")

    check("C2: sources list is empty on whitespace LLM answer",
          res_c["sources"] == [],
          f"sources: {res_c['sources']}")


# ===========================================================================
# Test D -- Exact M-13 Fallback
# ===========================================================================

print("\n--- Test D: Exact M-13 fallback ---")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="  I could not find relevant information in the comic.  "):

    res_d = answer_question("What did Victor say?", comic_id=COMIC_ID_VALID)

    check("D1: exact fallback answer returned",
          res_d["answer"] == FALLBACK_ANSWER,
          f"answer: {res_d['answer']!r}")

    check("D2: sources list is empty when LLM returns exact fallback",
          res_d["sources"] == [],
          f"sources: {res_d['sources']}")


# ===========================================================================
# Test E -- Invalid Source Distance
# ===========================================================================

print("\n--- Test E: Invalid source distance filtering ---")

CHUNKS_WITH_HIGH_DISTANCE = [
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        "content": "Valid chunk below threshold.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        },
        "distance": 0.20,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2",
        "content": "Invalid chunk above threshold.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 1,
            "chunk_index": 2,
            "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2",
        },
        "distance": 0.45,  # > 0.35
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1",
        "content": "Another valid chunk below threshold.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 2,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1",
        },
        "distance": 0.32,
    },
]

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=CHUNKS_WITH_HIGH_DISTANCE), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="Here is the validated answer."):

    res_e = answer_question("Test query", comic_id=COMIC_ID_VALID, distance_threshold=0.35)

    res_e_ids = [s["chunk_id"] for s in res_e["sources"]]
    check("E1: chunk with distance > 0.35 is filtered out of sources",
          f"{COMIC_ID_VALID}_page_1_chunk_2" not in res_e_ids,
          f"returned ids: {res_e_ids}")

    check("E2: valid chunks below threshold are preserved",
          res_e_ids == [f"{COMIC_ID_VALID}_page_1_chunk_1", f"{COMIC_ID_VALID}_page_2_chunk_1"],
          f"returned ids: {res_e_ids}")

    check("E3: all returned sources have distance <= 0.35",
          all(s["distance"] <= 0.35 for s in res_e["sources"]),
          f"distances: {[s['distance'] for s in res_e['sources']]}")


# ===========================================================================
# Test F -- Wrong Comic Source Filtering
# ===========================================================================

print("\n--- Test F: Wrong comic source filtering ---")

CHUNKS_WITH_MIXED_COMICS = [
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        "content": "Valid chunk from requested comic.",
        "metadata": {
            "comic_id": COMIC_ID_VALID,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        },
        "distance": 0.18,
    },
    {
        "chunk_id": f"{COMIC_ID_OTHER}_page_1_chunk_1",
        "content": "Leaked chunk from wrong comic.",
        "metadata": {
            "comic_id": COMIC_ID_OTHER,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{COMIC_ID_OTHER}_page_1_chunk_1",
        },
        "distance": 0.19,
    },
]

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=CHUNKS_WITH_MIXED_COMICS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="Answer with mixed sources."):

    res_f = answer_question("Test query", comic_id=COMIC_ID_VALID)

    check("F1: source belonging to wrong comic_id is filtered out",
          all(s["comic_id"] == COMIC_ID_VALID for s in res_f["sources"]),
          f"comic_ids: {[s['comic_id'] for s in res_f['sources']]}")

    check("F2: valid comic source is preserved",
          len(res_f["sources"]) == 1 and res_f["sources"][0]["comic_id"] == COMIC_ID_VALID)


# ===========================================================================
# Test G -- Source Ordering
# ===========================================================================

print("\n--- Test G: Source ordering preservation ---")

MULTI_PAGE_CHUNKS = [
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1",
        "content": "Page 1 Chunk 1",
        "metadata": {"comic_id": COMIC_ID_VALID, "page_number": 1, "chunk_index": 1, "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_1"},
        "distance": 0.30,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2",
        "content": "Page 1 Chunk 2",
        "metadata": {"comic_id": COMIC_ID_VALID, "page_number": 1, "chunk_index": 2, "chunk_id": f"{COMIC_ID_VALID}_page_1_chunk_2"},
        "distance": 0.15,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1",
        "content": "Page 2 Chunk 1",
        "metadata": {"comic_id": COMIC_ID_VALID, "page_number": 2, "chunk_index": 1, "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_1"},
        "distance": 0.28,
    },
    {
        "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_2",
        "content": "Page 2 Chunk 2",
        "metadata": {"comic_id": COMIC_ID_VALID, "page_number": 2, "chunk_index": 2, "chunk_id": f"{COMIC_ID_VALID}_page_2_chunk_2"},
        "distance": 0.10,
    },
]

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=MULTI_PAGE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", return_value="Answer for multi page."):

    res_g = answer_question("Test query", comic_id=COMIC_ID_VALID)

    order_tuples = [(s["page_number"], s["chunk_index"]) for s in res_g["sources"]]
    expected_order = [(1, 1), (1, 2), (2, 1), (2, 2)]
    check("G1: source list preserves exact page/chunk deterministic order",
          order_tuples == expected_order,
          f"order: {order_tuples}")


# ===========================================================================
# Test H -- Empty Retrieval
# ===========================================================================

print("\n--- Test H: Empty retrieval ---")

mock_generate = mock.MagicMock(return_value="Should not be called")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=[]), \
     mock.patch("app.services.rag_qa.generate_answer", mock_generate):

    res_h = answer_question("Unmatched question", comic_id=COMIC_ID_VALID)

    check("H1: LLM generate_answer is NOT called on empty retrieval",
          mock_generate.call_count == 0,
          f"call count: {mock_generate.call_count}")

    check("H2: fallback answer returned on empty retrieval",
          res_h["answer"] == FALLBACK_ANSWER,
          f"answer: {res_h['answer']!r}")

    check("H3: sources list is empty",
          res_h["sources"] == [],
          f"sources: {res_h['sources']}")

    check("H4: question and comic_id preserved",
          res_h["question"] == "Unmatched question" and res_h["comic_id"] == COMIC_ID_VALID)


# ===========================================================================
# Test I -- Query Normalization Regression (M-11)
# ===========================================================================

print("\n--- Test I: Query normalization regression ---")

check("I1: normalize_query trims and collapses repeated whitespace",
      normalize_query("   What   did   Victor   say   about   his mother?  ") == "What did Victor say about his mother?")

check("I2: normalize_query preserves already-clean query",
      normalize_query("What did Victor say?") == "What did Victor say?")

check("I3: normalize_query handles whitespace-only string",
      normalize_query("    \t\n  ") == "")

check("I4: normalize_query handles empty string",
      normalize_query("") == "")

with mock.patch("app.services.rag_qa.retrieve_chunks") as mock_retrieve, \
     mock.patch("app.services.rag_qa.generate_answer") as mock_gen:

    res_i = answer_question("   \t   ", comic_id=COMIC_ID_VALID)

    check("I5: whitespace question bypasses retrieval and LLM",
          mock_retrieve.call_count == 0 and mock_gen.call_count == 0)

    check("I6: validation message returned for empty question",
          res_i["answer"] == "Please provide a valid question." and res_i["sources"] == [])


# ===========================================================================
# Test J -- Existing Configuration Regression
# ===========================================================================

print("\n--- Test J: Configuration regression checks ---")

vs_path = pathlib.Path("app/services/vector_store.py").read_text(encoding="utf-8")
retriever_path = pathlib.Path("app/services/retriever.py").read_text(encoding="utf-8")
rag_qa_path = pathlib.Path("app/services/rag_qa.py").read_text(encoding="utf-8")
llm_path = pathlib.Path("app/services/llm.py").read_text(encoding="utf-8")

check("J1: cosine metric in vector_store.py",
      "cosine" in vs_path)

check("J2: collection name 'comic_pages' in vector_store.py",
      "comic_pages" in vs_path)

check("J3: DEFAULT_DISTANCE_THRESHOLD == 0.35",
      DEFAULT_DISTANCE_THRESHOLD == 0.35,
      f"got: {DEFAULT_DISTANCE_THRESHOLD}")

check("J4: top_k default = 5 in retriever.py",
      "top_k: int = 5" in retriever_path or "top_k=5" in retriever_path)

check("J5: deterministic sorting present in retriever.py",
      "final_chunks.sort" in retriever_path)

check("J6: comic_id filtering present in vector_store.py",
      '"comic_id"' in vs_path or "'comic_id'" in vs_path)

check("J7: [PAGE X | CHUNK Y] context markers in rag_qa.py",
      "PAGE" in rag_qa_path and "CHUNK" in rag_qa_path)

check("J8: separator '--------------------' in rag_qa.py",
      "--------------------" in rag_qa_path)

check("J9: exact M-13 fallback phrase defined in rag_qa.py",
      'FALLBACK_ANSWER = "I could not find relevant information in the comic."' in rag_qa_path)

check("J10: exact M-13 fallback phrase pinned in llm.py",
      "I could not find relevant information in the comic." in llm_path)


# ===========================================================================
# Test K -- No Extra LLM Validation Call
# ===========================================================================

print("\n--- Test K: No extra LLM validation call ---")

call_counter = mock.MagicMock(return_value="Answer from single LLM invocation.")

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", call_counter):

    res_k1 = answer_question("What did Victor say?", comic_id=COMIC_ID_VALID)

    check("K1: exactly ONE LLM call made for valid retrieval (no extra LLM validation call)",
          call_counter.call_count == 1,
          f"call count: {call_counter.call_count}")

call_counter.reset_mock()

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=[]), \
     mock.patch("app.services.rag_qa.generate_answer", call_counter):

    res_k2 = answer_question("Unmatched question", comic_id=COMIC_ID_VALID)

    check("K2: exactly ZERO LLM calls made for empty retrieval",
          call_counter.call_count == 0,
          f"call count: {call_counter.call_count}")

call_counter.reset_mock()

with mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
     mock.patch("app.services.rag_qa.generate_answer", call_counter):

    res_k3 = answer_question("   ", comic_id=COMIC_ID_VALID)

    check("K3: exactly ZERO LLM calls made for empty/whitespace query",
          call_counter.call_count == 0,
          f"call count: {call_counter.call_count}")


# ===========================================================================
# Direct Unit Tests for validate_answer_and_sources
# ===========================================================================

print("\n--- Direct Unit Tests: validate_answer_and_sources ---")

# Unit 1: Empty answer
ans1, src1 = validate_answer_and_sources("", [{"comic_id": COMIC_ID_VALID, "distance": 0.2}], COMIC_ID_VALID)
check("U1: validate_answer_and_sources with empty string -> (FALLBACK, [])",
      ans1 == FALLBACK_ANSWER and src1 == [])

# Unit 2: Whitespace answer
ans2, src2 = validate_answer_and_sources("   ", [{"comic_id": COMIC_ID_VALID, "distance": 0.2}], COMIC_ID_VALID)
check("U2: validate_answer_and_sources with whitespace -> (FALLBACK, [])",
      ans2 == FALLBACK_ANSWER and src2 == [])

# Unit 3: Exact fallback
ans3, src3 = validate_answer_and_sources(FALLBACK_ANSWER, [{"comic_id": COMIC_ID_VALID, "distance": 0.2}], COMIC_ID_VALID)
check("U3: validate_answer_and_sources with exact fallback -> (FALLBACK, [])",
      ans3 == FALLBACK_ANSWER and src3 == [])

# Unit 4: Normal answer with valid and invalid sources
mixed_sources = [
    {"comic_id": COMIC_ID_VALID, "page_number": 1, "chunk_id": "c1", "chunk_index": 1, "distance": 0.20},
    {"comic_id": COMIC_ID_OTHER, "page_number": 1, "chunk_id": "c2", "chunk_index": 2, "distance": 0.15},
    {"comic_id": COMIC_ID_VALID, "page_number": 1, "chunk_id": "c3", "chunk_index": 3, "distance": 0.40},
    {"comic_id": COMIC_ID_VALID, "page_number": 2, "chunk_id": "c4", "chunk_index": 1, "distance": None},
    {"comic_id": COMIC_ID_VALID, "page_number": 2, "chunk_id": "c5", "chunk_index": 2, "distance": 0.35},
]
ans4, src4 = validate_answer_and_sources("A valid answer.", mixed_sources, COMIC_ID_VALID, distance_threshold=0.35)
check("U4: validate_answer_and_sources filters wrong comic, distance > threshold, and None distance",
      ans4 == "A valid answer." and [s["chunk_id"] for s in src4] == ["c1", "c5"],
      f"surviving chunk_ids: {[s['chunk_id'] for s in src4]}")


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)

print(f"Results: {passed}/{total} checks passed ({failed} failed)")

for name, status, detail in results:
    if status == FAIL:
        print(f"  [FAIL] {name} -- {detail}")

print()
if failed == 0:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL ({failed} check(s) failed)")

sys.exit(0 if failed == 0 else 1)
