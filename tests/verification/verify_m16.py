"""
verify_m16.py -- M-16 Backend API Contract & Frontend-Ready Endpoint Validation

Deterministic verification suite testing:
- Test A: Upload contract (valid upload, response model, comic_id, rag fields)
- Test B: Invalid upload (empty file, unsupported format, missing filename)
- Test C: Get comic (valid UUID, existing comic, non-existing comic 404, invalid UUID 400)
- Test D: Comic-scoped ask (POST /comics/{comic_id}/ask, response contract, sources structure)
- Test E: Generic ask (POST /ask and POST /comics/ask, shared contract)
- Test F: Empty question (empty string, whitespace, retrieval/LLM bypass)
- Test G: Error contract (safe JSON {"detail": ...}, no tracebacks/paths/keys)
- Test H: OpenAPI verification (/openapi.json, /docs, schemas, paths)
- Test I: RAG regression (cosine, top_k=5, threshold=0.35, sorting, normalization, fallback)
- Test J: No duplicate RAG logic (all ask routes delegate to answer_question)

Console output is ASCII-only.

Usage
-----
    python -u verify_m16.py
"""

import sys
import asyncio
import pathlib
import unittest.mock as mock
import httpx

# ---------------------------------------------------------------------------
# Output Helpers (ASCII-only)
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

results = []
all_responses_captured = []


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
# Import Application
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

from app.main import app
from app.services.query_normalizer import normalize_query
from app.services.retriever import DEFAULT_DISTANCE_THRESHOLD
from app.services.rag_qa import FALLBACK_ANSWER


VALID_UUID = "22222222-3333-4444-5555-666666666666"
NON_EXISTENT_UUID = "88888888-8888-8888-8888-888888888888"
INVALID_UUID = "bad-uuid-format"

SAMPLE_CHUNKS = [
    {
        "chunk_id": f"{VALID_UUID}_page_1_chunk_1",
        "content": "Victor recalls the warm advice his mother gave him in childhood.",
        "metadata": {
            "comic_id": VALID_UUID,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{VALID_UUID}_page_1_chunk_1",
        },
        "distance": 0.22,
    },
    {
        "chunk_id": f"{VALID_UUID}_page_1_chunk_2",
        "content": "His mother told him she was proud of his intellect.",
        "metadata": {
            "comic_id": VALID_UUID,
            "page_number": 1,
            "chunk_index": 2,
            "chunk_id": f"{VALID_UUID}_page_1_chunk_2",
        },
        "distance": 0.29,
    },
]


async def run_all_tests():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:

        # ===================================================================
        # Test A -- Upload Contract
        # ===================================================================
        print("\n--- Test A: Upload contract ---")

        fake_pages = [{"page_number": 1, "status": "success", "analysis": {}}]
        fake_save_data = {"comic": {"id": "dummy-id", "name": "sample_comic"}}
        fake_ingest_res = {"documents": 1, "chunks": 2}

        with mock.patch("app.routes.comic.extract_image", return_value=fake_pages), \
             mock.patch("app.routes.comic.analyze_pages", return_value=fake_pages), \
             mock.patch("app.routes.comic.save_comic_json", return_value=fake_save_data), \
             mock.patch("app.routes.comic.ingest_comic_to_rag", return_value=fake_ingest_res):

            files_a = {"file": ("sample_comic.jpg", b"\xff\xd8\xff\xe0valid_image_bytes", "image/jpeg")}
            res_a = await client.post("/comics/upload", files=files_a)
            all_responses_captured.append(res_a)

            check("A1: upload returns 200 OK", res_a.status_code == 200, f"status: {res_a.status_code}")
            data_a = res_a.json()
            check("A2: response contains comic_id", bool(data_a.get("comic_id")))
            check("A3: response contains rag_ingested == True", data_a.get("rag_ingested") is True)
            check("A4: response contains rag_chunks_stored == 2", data_a.get("rag_chunks_stored") == 2)
            check("A5: response contains message and total_pages", "message" in data_a and "total_pages" in data_a)

        # ===================================================================
        # Test B -- Invalid Upload
        # ===================================================================
        print("\n--- Test B: Invalid upload validation ---")

        # 0-byte file
        files_b1 = {"file": ("empty.png", b"", "image/png")}
        res_b1 = await client.post("/comics/upload", files=files_b1)
        all_responses_captured.append(res_b1)
        check("B1: 0-byte upload returns 400 Bad Request", res_b1.status_code == 400)
        check("B2: detail notes empty file", "empty" in res_b1.json().get("detail", "").lower())

        # Unsupported format
        files_b2 = {"file": ("script.py", b"print('hello')", "text/x-python")}
        res_b2 = await client.post("/comics/upload", files=files_b2)
        all_responses_captured.append(res_b2)
        check("B3: unsupported extension returns 400 Bad Request", res_b2.status_code == 400)
        check("B4: detail notes unsupported format", "unsupported" in res_b2.json().get("detail", "").lower())

        # Missing upload file
        res_b3 = await client.post("/comics/upload", data={})
        all_responses_captured.append(res_b3)
        check("B5: missing upload file returns 4xx", res_b3.status_code in (400, 422))

        # ===================================================================
        # Test C -- Get Comic
        # ===================================================================
        print("\n--- Test C: Get comic endpoint ---")

        # Existing comic
        dummy_comic_data = {"comic": {"id": VALID_UUID, "name": "Doom Story"}, "pages": []}
        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("builtins.open", mock.mock_open(read_data='{"comic": {"id": "' + VALID_UUID + '", "name": "Doom Story"}, "pages": []}')):

            res_c1 = await client.get(f"/comics/{VALID_UUID}")
            all_responses_captured.append(res_c1)
            check("C1: existing comic returns 200 OK", res_c1.status_code == 200)
            check("C2: returned data is JSON serializable comic info", res_c1.json().get("comic", {}).get("name") == "Doom Story")

        # Non-existing comic (404)
        res_c2 = await client.get(f"/comics/{NON_EXISTENT_UUID}")
        all_responses_captured.append(res_c2)
        check("C3: non-existent comic returns 404 Not Found", res_c2.status_code == 404)
        check("C4: error detail is 'Comic not found'", res_c2.json().get("detail") == "Comic not found")

        # Invalid UUID format (400)
        res_c3 = await client.get(f"/comics/{INVALID_UUID}")
        all_responses_captured.append(res_c3)
        check("C5: invalid UUID format returns 400 Bad Request", res_c3.status_code == 400)

        # ===================================================================
        # Test D -- Comic-Scoped Ask (/comics/{comic_id}/ask)
        # ===================================================================
        print("\n--- Test D: Comic-scoped ask (/comics/{comic_id}/ask) ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
             mock.patch("app.services.rag_qa.generate_answer", return_value="Victor said his mother believed in him."):

            res_d = await client.post(
                f"/comics/{VALID_UUID}/ask",
                json={"question": "What did Victor say about his mother?"}
            )
            all_responses_captured.append(res_d)

            check("D1: comic-scoped ask returns 200 OK", res_d.status_code == 200)
            data_d = res_d.json()
            check("D2: comic_id preserved in response", data_d.get("comic_id") == VALID_UUID)
            check("D3: question preserved in response", data_d.get("question") == "What did Victor say about his mother?")
            check("D4: answer generated and preserved", data_d.get("answer") == "Victor said his mother believed in him.")
            check("D5: sources list contains 2 chunks", len(data_d.get("sources", [])) == 2)

            src0 = data_d["sources"][0]
            check("D6: source contains typed fields (comic_id, page_number, chunk_id, chunk_index, distance)",
                  src0.get("comic_id") == VALID_UUID
                  and src0.get("page_number") == 1
                  and src0.get("chunk_id") == f"{VALID_UUID}_page_1_chunk_1"
                  and src0.get("chunk_index") == 1
                  and isinstance(src0.get("distance"), float))

        # ===================================================================
        # Test E -- Generic Ask (POST /ask and POST /comics/ask)
        # ===================================================================
        print("\n--- Test E: Generic ask endpoints (/ask and /comics/ask) ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
             mock.patch("app.services.rag_qa.generate_answer", return_value="Victor's mother was loving."):

            # Root POST /ask
            res_e1 = await client.post(
                "/ask",
                json={"comic_id": VALID_UUID, "question": "What did Victor say about his mother?"}
            )
            all_responses_captured.append(res_e1)
            check("E1: root /ask returns 200 OK", res_e1.status_code == 200)
            data_e1 = res_e1.json()
            check("E2: root /ask returns matching response structure",
                  data_e1.get("comic_id") == VALID_UUID and data_e1.get("answer") == "Victor's mother was loving." and len(data_e1.get("sources", [])) == 2)

            # /comics/ask
            res_e2 = await client.post(
                "/comics/ask",
                json={"comic_id": VALID_UUID, "question": "What did Victor say about his mother?"}
            )
            all_responses_captured.append(res_e2)
            check("E3: /comics/ask returns 200 OK with identical contract",
                  res_e2.status_code == 200 and res_e2.json().get("comic_id") == VALID_UUID)

        # ===================================================================
        # Test F -- Empty & Whitespace Question
        # ===================================================================
        print("\n--- Test F: Empty and whitespace question validation ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks") as mock_ret_f, \
             mock.patch("app.services.rag_qa.generate_answer") as mock_gen_f:

            # Empty string
            res_f1 = await client.post(f"/comics/{VALID_UUID}/ask", json={"question": ""})
            all_responses_captured.append(res_f1)
            check("F1: empty question returns 200 with validation message",
                  res_f1.status_code == 200 and res_f1.json().get("answer") == "Please provide a valid question." and res_f1.json().get("sources") == [])

            # Whitespace string
            res_f2 = await client.post("/ask", json={"comic_id": VALID_UUID, "question": "   \n\t  "})
            all_responses_captured.append(res_f2)
            check("F2: whitespace question returns 200 with validation message",
                  res_f2.status_code == 200 and res_f2.json().get("answer") == "Please provide a valid question." and res_f2.json().get("sources") == [])

            check("F3: retrieval is bypassed for empty/whitespace queries", mock_ret_f.call_count == 0)
            check("F4: LLM is bypassed for empty/whitespace queries", mock_gen_f.call_count == 0)

        # ===================================================================
        # Test G -- Error Contract & Sanitization
        # ===================================================================
        print("\n--- Test G: Error contract and sanitization ---")

        # 400 Bad Request
        res_g1 = await client.post("/comics/ask", json={"comic_id": "invalid", "question": "test"})
        all_responses_captured.append(res_g1)
        check("G1: 400 error returns {'detail': ...}", "detail" in res_g1.json())

        # 404 Not Found
        res_g2 = await client.get(f"/comics/{NON_EXISTENT_UUID}")
        all_responses_captured.append(res_g2)
        check("G2: 404 error returns {'detail': ...}", "detail" in res_g2.json())

        # 500 Unexpected Server Error
        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.routes.comic.answer_question", side_effect=RuntimeError("Hidden internal fault in C:\\Server\\Private")):

            res_g3 = await client.post(f"/comics/{VALID_UUID}/ask", json={"question": "Test"})
            all_responses_captured.append(res_g3)
            check("G3: 500 error returns safe {'detail': ...}", res_g3.status_code == 500 and "detail" in res_g3.json())
            detail_g3 = res_g3.json().get("detail", "")
            check("G4: 500 error contains no path or traceback", "C:\\Server" not in detail_g3 and "Traceback" not in detail_g3)

        # ===================================================================
        # Test H -- OpenAPI & Swagger Verification
        # ===================================================================
        print("\n--- Test H: OpenAPI & Swagger schema verification ---")

        # GET /openapi.json
        res_h1 = await client.get("/openapi.json")
        all_responses_captured.append(res_h1)
        check("H1: /openapi.json loads successfully (status 200)", res_h1.status_code == 200)

        schema = res_h1.json()
        paths = schema.get("paths", {})

        check("H2: /comics/upload path documented in OpenAPI", "/comics/upload" in paths)
        check("H3: /comics/{comic_id} path documented in OpenAPI", "/comics/{comic_id}" in paths)
        check("H4: /comics/{comic_id}/ask path documented in OpenAPI", "/comics/{comic_id}/ask" in paths)
        check("H5: /comics/ask path documented in OpenAPI", "/comics/ask" in paths)
        check("H6: /ask path documented in OpenAPI", "/ask" in paths)

        components = schema.get("components", {}).get("schemas", {})
        check("H7: QuestionRequest schema present in OpenAPI components", "QuestionRequest" in components)
        check("H8: QuestionResponse schema present in OpenAPI components", "QuestionResponse" in components)
        check("H9: SourceItem schema present in OpenAPI components", "SourceItem" in components)
        check("H10: ComicUploadResponse schema present in OpenAPI components", "ComicUploadResponse" in components)

        # GET /docs
        res_h2 = await client.get("/docs")
        all_responses_captured.append(res_h2)
        check("H11: Swagger UI /docs endpoint loads successfully (status 200)", res_h2.status_code == 200)

        # ===================================================================
        # Test I -- RAG Configuration Regression
        # ===================================================================
        print("\n--- Test I: RAG configuration regression ---")

        vs_src = pathlib.Path("app/services/vector_store.py").read_text(encoding="utf-8")
        ret_src = pathlib.Path("app/services/retriever.py").read_text(encoding="utf-8")
        qa_src = pathlib.Path("app/services/rag_qa.py").read_text(encoding="utf-8")
        llm_src = pathlib.Path("app/services/llm.py").read_text(encoding="utf-8")

        check("I1: cosine metric in vector_store.py", "cosine" in vs_src)
        check("I2: collection 'comic_pages' in vector_store.py", "comic_pages" in vs_src)
        check("I3: DEFAULT_DISTANCE_THRESHOLD == 0.35", DEFAULT_DISTANCE_THRESHOLD == 0.35)
        check("I4: top_k default = 5 in retriever.py", "top_k: int = 5" in ret_src or "top_k=5" in ret_src)
        check("I5: deterministic sort present in retriever.py", "final_chunks.sort" in ret_src)
        check("I6: comic_id filter present in vector_store.py", '"comic_id"' in vs_src or "'comic_id'" in vs_src)
        check("I7: query normalization used in rag_qa.py", "normalize_query" in qa_src)
        check("I8: grounding fallback phrase in rag_qa.py", FALLBACK_ANSWER in qa_src)
        check("I9: answer/source validation function present in rag_qa.py", "validate_answer_and_sources" in qa_src)
        check("I10: rule 11 (no fabrication) present in llm.py", "fabricate" in llm_src.lower())

        # ===================================================================
        # Test J -- No Duplicate RAG Logic
        # ===================================================================
        print("\n--- Test J: No duplicate RAG logic (all routes use answer_question) ---")

        mock_qa = mock.MagicMock(return_value={
            "comic_id": VALID_UUID,
            "question": "What happened?",
            "answer": "Victor reflected on his past.",
            "sources": []
        })

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.routes.comic.answer_question", mock_qa):

            # Call comic-scoped ask
            await client.post(f"/comics/{VALID_UUID}/ask", json={"question": "What happened?"})
            # Call generic /comics/ask
            await client.post("/comics/ask", json={"comic_id": VALID_UUID, "question": "What happened?"})
            # Call root /ask
            await client.post("/ask", json={"comic_id": VALID_UUID, "question": "What happened?"})

            check("J1: all 3 ask endpoints delegate to the single answer_question() service",
                  mock_qa.call_count == 3,
                  f"call count: {mock_qa.call_count}")

            # Verify identical arguments passed
            for call_args in mock_qa.call_args_list:
                _, kwargs = call_args
                check("J2: answer_question called with correct question and comic_id",
                      kwargs.get("question") == "What happened?" and kwargs.get("comic_id") == VALID_UUID)

        # ===================================================================
        # Security Sanity Check across all captured responses
        # ===================================================================
        print("\n--- Security Sanity Check across all captured responses ---")

        leaks_detected = []
        for i, res in enumerate(all_responses_captured):
            raw_text = res.text
            if "Traceback" in raw_text:
                leaks_detected.append(f"Response {i} leaked 'Traceback'")
            if "sk-" in raw_text:
                leaks_detected.append(f"Response {i} leaked 'sk-'")
            if "C:\\Users\\" in raw_text or "C:/Users/" in raw_text:
                leaks_detected.append(f"Response {i} leaked Windows User path")

        check("S1: zero responses leak Python tracebacks", len(leaks_detected) == 0, f"leaks: {leaks_detected}")
        check("S2: zero responses leak API keys", not any("sk-" in l for l in leaks_detected))
        check("S3: zero responses leak internal user file paths", not any("User path" in l for l in leaks_detected))


def main():
    asyncio.run(run_all_tests())

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


if __name__ == "__main__":
    main()
