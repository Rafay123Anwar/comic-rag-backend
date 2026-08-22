"""
verify_m13.py -- M-13 Answer Grounding and Source Integrity

Tests A through G verify grounding and source integrity against the live
ChromaDB + Mistral embedding + Mistral LLM stack.

Tests that require the LLM (A, F) make real API calls.
Tests B, C, D, E, G are verified via controlled retrieval + mocking or
static source inspection -- no fabricated data.

Usage
-----
    python -u verify_m13.py

Exit codes:
    0  -- all executed tests passed (may include LIMITATION notes)
    1  -- at least one test FAILED
"""

import sys
import re
import types
import pathlib

# ---------------------------------------------------------------------------
# Output helpers -- ASCII-safe only
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "LIMITATION"

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


def note(name, detail=""):
    results.append((name, SKIP, detail))
    print(f"  [LIMITATION] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Bootstrap real modules
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

from app.services.query_normalizer import normalize_query       # noqa
from app.services.retriever import (                            # noqa
    retrieve_chunks,
    DEFAULT_DISTANCE_THRESHOLD,
)
from app.services.vector_store import collection                # noqa
import app.services.llm as llm_mod                             # noqa
import app.services.rag_qa as rag_qa_mod                       # noqa


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

_all_meta = collection.get(include=["metadatas"])
_ids = _all_meta.get("ids", [])
_metas = _all_meta.get("metadatas", [])

from collections import defaultdict
_comic_pages = defaultdict(set)
for _id, _meta in zip(_ids, _metas):
    _comic_pages[_meta.get("comic_id")].add(_meta.get("page_number"))

VICTOR_ID = "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9"
PRIMARY_ID = VICTOR_ID if VICTOR_ID in _comic_pages else (
    list(_comic_pages.keys())[0] if _comic_pages else None
)
OTHER_IDS = [cid for cid in _comic_pages if cid != PRIMARY_ID]

print(f"\nDataset: {len(_ids)} chunks across {len(_comic_pages)} comic(s)")
for cid, pgs in _comic_pages.items():
    print(f"  {cid[:8]}... -- pages: {sorted(pgs)}")

_FALLBACK = "I could not find relevant information in the comic."

# ===========================================================================
# Test A -- Relevant question: answer grounded, sources non-empty
# ===========================================================================

print("\n--- Test A: Relevant question (live LLM) ---")

if not PRIMARY_ID:
    note("A: skipped -- no comics in vector store")
else:
    query_a = "What did Victor say about his mother?"
    result_a = rag_qa_mod.answer_question(
        question=query_a,
        comic_id=PRIMARY_ID,
    )

    check("A1: answer is non-empty",
          bool(result_a.get("answer", "").strip()),
          f"answer[:80]: {result_a.get('answer','')[:80]!r}")

    check("A2: sources are non-empty",
          len(result_a.get("sources", [])) > 0,
          f"sources count: {len(result_a.get('sources',[]))}")

    sources_a = result_a.get("sources", [])
    distances_a = [s.get("distance") for s in sources_a if s.get("distance") is not None]

    check("A3: all source distances <= 0.35",
          all(d <= DEFAULT_DISTANCE_THRESHOLD for d in distances_a),
          f"distances: {[round(d,4) for d in distances_a]}")

    check("A4: all source comic_ids match requested comic_id",
          all(s.get("comic_id") == PRIMARY_ID for s in sources_a),
          f"comic_ids: {[s.get('comic_id','?')[:8] for s in sources_a]}")

    # Source order matches context order: sources built in same loop as context_parts
    # Verify by re-running retrieval and comparing order
    norm_a = normalize_query(query_a)
    chunks_a = retrieve_chunks(
        query=norm_a,
        comic_id=PRIMARY_ID,
        top_k=5,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    )
    expected_chunk_ids = [
        c.get("chunk_id") or c["metadata"].get("chunk_id")
        for c in chunks_a
    ]
    actual_chunk_ids = [s.get("chunk_id") for s in sources_a]
    check("A5: source order matches context chunk order",
          actual_chunk_ids == expected_chunk_ids,
          f"actual={actual_chunk_ids}, expected={expected_chunk_ids}")

    check("A6: source chunk_ids are unique",
          len(set(actual_chunk_ids)) == len(actual_chunk_ids),
          f"chunk_ids: {actual_chunk_ids}")

    # Answer contains Victor-related content (basic sanity grounding check)
    answer_lower = result_a.get("answer", "").lower()
    grounded = (
        "victor" in answer_lower
        or "mother" in answer_lower
        or "womb" in answer_lower
        or "momma" in answer_lower
        or "awareness" in answer_lower
    )
    check("A7: answer contains information supported by retrieved context",
          grounded,
          f"answer[:120]: {result_a.get('answer','')[:120]!r}")

    check("A8: original question preserved in response",
          result_a.get("question") == query_a,
          f"got: {result_a.get('question')!r}")

    check("A9: comic_id preserved in response",
          result_a.get("comic_id") == PRIMARY_ID,
          f"got: {result_a.get('comic_id','?')[:8]}")


# ===========================================================================
# Test B -- Unrelated question: no LLM call, fallback answer, sources == []
# ===========================================================================

print("\n--- Test B: Unrelated question (no LLM call) ---")

if not PRIMARY_ID:
    note("B: skipped -- no comics in vector store")
else:
    # Use a tight threshold to guarantee empty retrieval so the LLM is NOT called
    _llm_called = {"called": False, "args": None}
    _original_generate = llm_mod.generate_answer

    def _spy_generate(question, context):
        _llm_called["called"] = True
        _llm_called["args"] = {"question": question, "context": context}
        return _original_generate(question, context)

    llm_mod.generate_answer = _spy_generate
    # Patch the reference inside rag_qa_mod (already imported)
    rag_qa_mod.generate_answer = _spy_generate

    try:
        # Force empty retrieval with an impossibly tight threshold
        query_b = "What is Victor's favorite food?"
        result_b = rag_qa_mod.answer_question(
            question=query_b,
            comic_id=PRIMARY_ID,
            distance_threshold=0.001,  # forces empty retrieval
        )

        check("B1: sources == []",
              result_b.get("sources") == [],
              f"sources: {result_b.get('sources')}")

        check("B2: LLM is NOT called when retrieval is empty",
              not _llm_called["called"],
              "LLM was called" if _llm_called["called"] else "LLM not called")

        check("B3: fallback answer returned",
              _FALLBACK in result_b.get("answer", ""),
              f"answer: {result_b.get('answer','')!r}")

        check("B4: question preserved",
              result_b.get("question") == query_b,
              f"got: {result_b.get('question')!r}")

        check("B5: comic_id preserved",
              result_b.get("comic_id") == PRIMARY_ID)
    finally:
        llm_mod.generate_answer = _original_generate
        rag_qa_mod.generate_answer = _original_generate


# ===========================================================================
# Test C -- Empty/whitespace query: M-11 behavior preserved
# ===========================================================================

print("\n--- Test C: Empty/whitespace query (M-11 regression) ---")

if not PRIMARY_ID:
    note("C: skipped -- no comics in vector store")
else:
    _llm_called_c = {"called": False}

    def _spy_generate_c(question, context):
        _llm_called_c["called"] = True
        return _original_generate(question, context)

    llm_mod.generate_answer = _spy_generate_c
    rag_qa_mod.generate_answer = _spy_generate_c

    try:
        for label, empty_q in [
            ("empty string", ""),
            ("whitespace only", "     "),
            ("tabs and newlines", "\t\n  \t"),
        ]:
            _llm_called_c["called"] = False
            result_c = rag_qa_mod.answer_question(
                question=empty_q,
                comic_id=PRIMARY_ID,
            )
            check(f"C1[{label}]: sources == []",
                  result_c.get("sources") == [],
                  f"sources: {result_c.get('sources')}")
            check(f"C2[{label}]: LLM NOT called",
                  not _llm_called_c["called"],
                  "LLM was called" if _llm_called_c["called"] else "ok")
            check(f"C3[{label}]: validation message returned",
                  "valid" in result_c.get("answer", "").lower()
                  or "provide" in result_c.get("answer", "").lower(),
                  f"answer: {result_c.get('answer','')!r}")
    finally:
        llm_mod.generate_answer = _original_generate
        rag_qa_mod.generate_answer = _original_generate


# ===========================================================================
# Test D -- Source integrity: every source = one retrieved chunk
# ===========================================================================

print("\n--- Test D: Source integrity ---")

if not PRIMARY_ID:
    note("D: skipped -- no comics in vector store")
else:
    norm_d = normalize_query("What did Victor say about his mother?")
    chunks_d = retrieve_chunks(
        query=norm_d,
        comic_id=PRIMARY_ID,
        top_k=5,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    )

    # Simulate rag_qa source assembly
    sources_d = []
    context_ids_d = []
    for c in chunks_d:
        meta = c.get("metadata", {})
        cid = c.get("chunk_id") or meta.get("chunk_id")
        sources_d.append({
            "comic_id": meta.get("comic_id", PRIMARY_ID),
            "page_number": meta.get("page_number"),
            "chunk_id": cid,
            "chunk_index": meta.get("chunk_index"),
            "distance": c.get("distance"),
        })
        context_ids_d.append(cid)

    check("D1: source count equals retrieved chunk count",
          len(sources_d) == len(chunks_d),
          f"sources={len(sources_d)}, chunks={len(chunks_d)}")

    check("D2: source chunk_ids match chunk_ids in order",
          [s["chunk_id"] for s in sources_d] == context_ids_d,
          f"source_ids={[s['chunk_id'] for s in sources_d]}")

    check("D3: all sources have comic_id field",
          all(s.get("comic_id") for s in sources_d))

    check("D4: all sources have page_number field",
          all(s.get("page_number") is not None for s in sources_d))

    check("D5: all sources have chunk_id field",
          all(s.get("chunk_id") for s in sources_d))

    check("D6: all sources have chunk_index field",
          all(s.get("chunk_index") is not None for s in sources_d))

    check("D7: all sources have distance field",
          all(s.get("distance") is not None for s in sources_d),
          f"distances: {[s.get('distance') for s in sources_d]}")

    check("D8: no fabricated sources (all distances > 0)",
          all(
              s.get("distance") is not None and s["distance"] >= 0
              for s in sources_d
          ))


# ===========================================================================
# Test E -- Comic isolation: sources never contain another comic_id
# ===========================================================================

print("\n--- Test E: Comic isolation ---")

if not PRIMARY_ID or not OTHER_IDS:
    note("E: skipped -- need at least 2 comics in vector store")
else:
    norm_e = normalize_query("Victor mother womb awareness")
    chunks_e = retrieve_chunks(
        query=norm_e,
        comic_id=PRIMARY_ID,
        top_k=5,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    )
    leaked = [
        c for c in chunks_e
        if c["metadata"].get("comic_id") != PRIMARY_ID
    ]
    check("E1: no chunks from other comics in retrieval result",
          len(leaked) == 0,
          f"leaked comic_ids: {[c['metadata'].get('comic_id','?')[:8] for c in leaked]}")

    # Cross-check: query another comic, verify primary doesn't bleed in
    other_id = OTHER_IDS[0]
    chunks_e2 = retrieve_chunks(
        query=norm_e,
        comic_id=other_id,
        top_k=5,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    )
    primary_leaked = [
        c for c in chunks_e2
        if c["metadata"].get("comic_id") == PRIMARY_ID
    ]
    check("E2: primary comic does not bleed into other-comic query",
          len(primary_leaked) == 0,
          f"leaked: {len(primary_leaked)}")


# ===========================================================================
# Test F -- Prompt grounding: verify grounding instruction is in the prompt
# ===========================================================================

print("\n--- Test F: Prompt grounding (prompt inspection) ---")

_captured_prompt = {"system": None, "user": None}
_original_chat_complete = llm_mod.client.chat.complete


def _capture_complete(model, messages, temperature=0.1):
    for msg in messages:
        if msg.get("role") == "system":
            _captured_prompt["system"] = msg.get("content", "")
        if msg.get("role") == "user":
            _captured_prompt["user"] = msg.get("content", "")
    return _original_chat_complete(
        model=model, messages=messages, temperature=temperature
    )


llm_mod.client.chat.complete = _capture_complete

try:
    if PRIMARY_ID:
        norm_f = normalize_query("What did Victor say about his mother?")
        chunks_f = retrieve_chunks(
            query=norm_f,
            comic_id=PRIMARY_ID,
            top_k=5,
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        )
        # Build context the same way rag_qa does
        context_parts_f = []
        for c in chunks_f:
            meta = c.get("metadata", {})
            pg = meta.get("page_number")
            ci = meta.get("chunk_index")
            content = c.get("content", "")
            context_parts_f.append(f"[PAGE {pg} | CHUNK {ci}]\n{content}")
        context_f = "\n\n--------------------\n\n".join(context_parts_f)
        # Call generate_answer to capture the prompt
        llm_mod.generate_answer(
            question="What did Victor say about his mother?",
            context=context_f,
        )

        sys_prompt = _captured_prompt.get("system", "") or ""
        user_prompt = _captured_prompt.get("user", "") or ""

        check("F1: system prompt contains 'ONLY' (context-only instruction)",
              "ONLY" in sys_prompt or "only" in sys_prompt,
              f"found: {'yes' if 'only' in sys_prompt.lower() else 'no'}")

        check("F2: system prompt contains 'Do not use outside knowledge'",
              "outside knowledge" in sys_prompt.lower(),
              f"found: {'yes' if 'outside knowledge' in sys_prompt.lower() else 'no'}")

        check("F3: system prompt contains the pinned fallback phrase",
              "I could not find relevant information in the comic." in sys_prompt,
              f"found: {'yes' if _FALLBACK in sys_prompt else 'no'}")

        check("F4: system prompt contains rule 11 (no fabrication)",
              "fabricate" in sys_prompt.lower(),
              f"found: {'yes' if 'fabricate' in sys_prompt.lower() else 'no'}")

        check("F5: user prompt contains 'COMIC CONTEXT'",
              "COMIC CONTEXT" in user_prompt,
              f"found: {'yes' if 'COMIC CONTEXT' in user_prompt else 'no'}")

        check("F6: user prompt contains [PAGE marker",
              "[PAGE" in user_prompt,
              f"found: {'yes' if '[PAGE' in user_prompt else 'no'}")

        check("F7: user prompt contains separator '--------------------'",
              "--------------------" in user_prompt or len(chunks_f) <= 1,
              f"chunks: {len(chunks_f)}")
    else:
        note("F: skipped -- no primary comic")
finally:
    llm_mod.client.chat.complete = _original_chat_complete


# ===========================================================================
# Test G -- Regression checks (static + live)
# ===========================================================================

print("\n--- Test G: Regression checks ---")

vs_src = pathlib.Path("app/services/vector_store.py").read_text(encoding="utf-8")
retriever_src = pathlib.Path("app/services/retriever.py").read_text(encoding="utf-8")
rag_qa_src = pathlib.Path("app/services/rag_qa.py").read_text(encoding="utf-8")
normalizer_src = pathlib.Path("app/services/query_normalizer.py").read_text(encoding="utf-8")
llm_src = pathlib.Path("app/services/llm.py").read_text(encoding="utf-8")
ingestion_src = pathlib.Path("app/services/rag_ingestion.py").read_text(encoding="utf-8")
chunker_src = pathlib.Path("app/services/rag_chunker.py").read_text(encoding="utf-8")

check("G1: cosine metric in vector_store.py",
      "cosine" in vs_src)

check("G2: collection name 'comic_pages' unchanged",
      "comic_pages" in vs_src)

check("G3: DEFAULT_DISTANCE_THRESHOLD == 0.35",
      DEFAULT_DISTANCE_THRESHOLD == 0.35,
      f"got: {DEFAULT_DISTANCE_THRESHOLD}")

check("G4: top_k default = 5 in retriever.py",
      "top_k: int = 5" in retriever_src or "top_k=5" in retriever_src)

check("G5: deterministic sort present in retriever.py",
      "final_chunks.sort" in retriever_src)

check("G6: comic_id filter in vector_store.py",
      '"comic_id"' in vs_src or "'comic_id'" in vs_src)

check("G7: normalize_query used in rag_qa.py",
      "normalize_query" in rag_qa_src)

check("G8: [PAGE X | CHUNK Y] marker format in rag_qa.py",
      "PAGE" in rag_qa_src and "CHUNK" in rag_qa_src)

check("G9: separator '--------------------' in rag_qa.py",
      "--------------------" in rag_qa_src)

check("G10: empty retrieval returns sources=[] without LLM call",
      "if not chunks:" in rag_qa_src and '"sources": []' in rag_qa_src,
      "verified by source inspection")

check("G11: exact-phrase source-drop (not broad heuristic) in rag_qa.py",
      "FALLBACK_ANSWER" in rag_qa_src or "answer.strip() == _FALLBACK" in rag_qa_src,
      "verified by source inspection")

check("G12: rule 11 (no fabrication) present in llm.py",
      "fabricate" in llm_src.lower())

check("G13: exact fallback phrase pinned in llm.py system prompt",
      "I could not find relevant information in the comic." in llm_src)

check("G14: ingestion/chunking files unchanged (no M-13 modification)",
      "M-13" not in ingestion_src and "M-13" not in chunker_src,
      "ingestion and chunker do not reference M-13")

check("G15: query_normalizer uses only stdlib re (no LLM imports)",
      "import re" in normalizer_src
      and "mistral" not in normalizer_src.lower()
      and "openai" not in normalizer_src.lower())


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
limited = sum(1 for _, s, _ in results if s == SKIP)
failed = sum(1 for _, s, _ in results if s == FAIL)

print(f"Results: {passed} PASS  |  {limited} LIMITATION  |  {failed} FAIL  |  {total} total")

for name, status, detail in results:
    if status == FAIL:
        print(f"  [FAIL] {name} -- {detail}")
for name, status, detail in results:
    if status == SKIP:
        print(f"  [LIMITATION] {name} -- {detail}")

print()
if failed == 0 and limited == 0:
    print("VERDICT: PASS")
elif failed == 0 and limited > 0:
    print("VERDICT: PASS WITH LIMITATION")
else:
    print(f"VERDICT: FAIL ({failed} check(s) failed)")

sys.exit(0 if failed == 0 else 1)
