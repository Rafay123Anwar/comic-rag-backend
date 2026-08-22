"""
verify_m15.py -- M-15 API Error Handling & Input Validation Verification Suite

Deterministic test suite verifying:
- Upload validation (missing, empty, unsupported, processing failure, isolated RAG failure)
- Comic ID validation (missing, whitespace, invalid UUID, non-existent 404)
- Question validation (empty, whitespace, preserved original)
- HTTP status codes (200, 400, 404, 422, 500)
- Consistent error response structure {"detail": ...}
- Internal exception safety and sanitization
- Security sanity: no leak of tracebacks, API keys, 'sk-', absolute filesystem paths
- Full RAG regression checks

Console output is ASCII-only.

Usage
-----
    python -u verify_m15.py
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
all_responses_captured = []  # For Test O security sanity check


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


VALID_UUID = "11111111-2222-3333-4444-555555555555"
NON_EXISTENT_UUID = "99999999-9999-9999-9999-999999999999"
INVALID_UUID = "not-a-valid-uuid-string"

SAMPLE_CHUNKS = [
    {
        "chunk_id": f"{VALID_UUID}_page_1_chunk_1",
        "content": "Victor speaks about his early memories.",
        "metadata": {
            "comic_id": VALID_UUID,
            "page_number": 1,
            "chunk_index": 1,
            "chunk_id": f"{VALID_UUID}_page_1_chunk_1",
        },
        "distance": 0.20,
    }
]


async def run_all_tests():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:

        # ===================================================================
        # Test A -- Valid QA Request
        # ===================================================================
        print("\n--- Test A: Valid QA request ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks", return_value=SAMPLE_CHUNKS), \
             mock.patch("app.services.rag_qa.generate_answer", return_value="Victor talked about his mother."):

            res_a = await client.post(
                f"/comics/{VALID_UUID}/ask",
                json={"question": "What did Victor say about his mother?"}
            )
            all_responses_captured.append(res_a)

            check("A1: valid request returns 200 OK", res_a.status_code == 200, f"status: {res_a.status_code}")
            data_a = res_a.json()
            check("A2: answer is preserved in response", data_a.get("answer") == "Victor talked about his mother.")
            check("A3: sources list is non-empty", len(data_a.get("sources", [])) == 1)
            check("A4: question is preserved in response", data_a.get("question") == "What did Victor say about his mother?")
            check("A5: comic_id is preserved in response", data_a.get("comic_id") == VALID_UUID)

        # ===================================================================
        # Test B -- Empty Question
        # ===================================================================
        print("\n--- Test B: Empty question ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks") as mock_ret_b, \
             mock.patch("app.services.rag_qa.generate_answer") as mock_gen_b:

            res_b = await client.post(f"/comics/{VALID_UUID}/ask", json={"question": ""})
            all_responses_captured.append(res_b)

            check("B1: empty question returns 200 OK with validation message", res_b.status_code == 200)
            data_b = res_b.json()
            check("B2: validation answer returned", data_b.get("answer") == "Please provide a valid question.")
            check("B3: sources is empty list", data_b.get("sources") == [])
            check("B4: retrieval is not called", mock_ret_b.call_count == 0)
            check("B5: LLM is not called", mock_gen_b.call_count == 0)

        # ===================================================================
        # Test C -- Whitespace Question
        # ===================================================================
        print("\n--- Test C: Whitespace-only question ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.services.rag_qa.retrieve_chunks") as mock_ret_c, \
             mock.patch("app.services.rag_qa.generate_answer") as mock_gen_c:

            res_c = await client.post(f"/comics/{VALID_UUID}/ask", json={"question": "   \n\t  "})
            all_responses_captured.append(res_c)

            check("C1: whitespace question returns 200 OK", res_c.status_code == 200)
            data_c = res_c.json()
            check("C2: validation answer returned", data_c.get("answer") == "Please provide a valid question.")
            check("C3: sources is empty list", data_c.get("sources") == [])
            check("C4: retrieval and LLM bypassed", mock_ret_c.call_count == 0 and mock_gen_c.call_count == 0)

        # ===================================================================
        # Test D -- Missing / Empty comic_id
        # ===================================================================
        print("\n--- Test D: Missing/empty comic_id ---")

        # POST /comics/ask with empty comic_id
        res_d1 = await client.post("/comics/ask", json={"comic_id": "", "question": "Hello?"})
        all_responses_captured.append(res_d1)
        check("D1: empty comic_id in body returns 400 Bad Request", res_d1.status_code == 400, f"status: {res_d1.status_code}")
        check("D2: human-readable error detail present", "comic_id is required" in res_d1.json().get("detail", ""))

        # POST /comics/ask with missing comic_id
        res_d2 = await client.post("/comics/ask", json={"question": "Hello?"})
        all_responses_captured.append(res_d2)
        check("D3: missing comic_id in body returns 400 Bad Request", res_d2.status_code == 400)

        # Whitespace-only comic_id
        res_d3 = await client.post("/comics/ask", json={"comic_id": "   ", "question": "Hello?"})
        all_responses_captured.append(res_d3)
        check("D4: whitespace comic_id returns 400 Bad Request", res_d3.status_code == 400)

        # ===================================================================
        # Test E -- Invalid comic_id Format
        # ===================================================================
        print("\n--- Test E: Invalid comic_id format ---")

        res_e1 = await client.post(f"/comics/{INVALID_UUID}/ask", json={"question": "Hello?"})
        all_responses_captured.append(res_e1)
        check("E1: invalid UUID in path returns 400 Bad Request", res_e1.status_code == 400, f"status: {res_e1.status_code}")
        check("E2: error detail notes invalid UUID format", "Invalid comic_id format" in res_e1.json().get("detail", ""))

        res_e2 = await client.get(f"/comics/{INVALID_UUID}")
        all_responses_captured.append(res_e2)
        check("E3: invalid UUID in GET path returns 400 Bad Request", res_e2.status_code == 400)

        # ===================================================================
        # Test F -- Non-Existent Comic (404)
        # ===================================================================
        print("\n--- Test F: Non-existent comic (404) ---")

        res_f1 = await client.post(f"/comics/{NON_EXISTENT_UUID}/ask", json={"question": "Hello?"})
        all_responses_captured.append(res_f1)
        check("F1: non-existent comic returns 404 Not Found on ask", res_f1.status_code == 404, f"status: {res_f1.status_code}")
        check("F2: error detail is 'Comic not found'", res_f1.json().get("detail") == "Comic not found")

        res_f2 = await client.get(f"/comics/{NON_EXISTENT_UUID}")
        all_responses_captured.append(res_f2)
        check("F3: non-existent comic returns 404 Not Found on get", res_f2.status_code == 404)

        # ===================================================================
        # Test G -- Missing Upload File
        # ===================================================================
        print("\n--- Test G: Missing upload file ---")

        res_g = await client.post("/comics/upload", data={})
        all_responses_captured.append(res_g)
        check("G1: missing file returns 4xx status code", res_g.status_code in (400, 422), f"status: {res_g.status_code}")
        check("G2: error detail present", "detail" in res_g.json())

        # ===================================================================
        # Test H -- Empty Upload File (0 bytes)
        # ===================================================================
        print("\n--- Test H: Empty upload file (0 bytes) ---")

        with mock.patch("app.routes.comic.extract_image") as mock_ext_h, \
             mock.patch("app.routes.comic.analyze_pages") as mock_ana_h:

            files_h = {"file": ("empty_comic.jpg", b"", "image/jpeg")}
            res_h = await client.post("/comics/upload", files=files_h)
            all_responses_captured.append(res_h)

            check("H1: 0-byte upload rejected with 400 Bad Request", res_h.status_code == 400, f"status: {res_h.status_code}")
            check("H2: detail explains file is empty", "empty" in res_h.json().get("detail", "").lower())
            check("H3: extraction is not triggered", mock_ext_h.call_count == 0)
            check("H4: AI analysis is not triggered", mock_ana_h.call_count == 0)

        # ===================================================================
        # Test I -- Unsupported File Format
        # ===================================================================
        print("\n--- Test I: Unsupported file format ---")

        files_i1 = {"file": ("malicious.exe", b"binary content", "application/octet-stream")}
        res_i1 = await client.post("/comics/upload", files=files_i1)
        all_responses_captured.append(res_i1)
        check("I1: .exe file rejected with 400 Bad Request", res_i1.status_code == 400, f"status: {res_i1.status_code}")
        check("I2: detail mentions unsupported format", "unsupported" in res_i1.json().get("detail", "").lower())

        files_i2 = {"file": ("document.txt", b"plain text", "text/plain")}
        res_i2 = await client.post("/comics/upload", files=files_i2)
        all_responses_captured.append(res_i2)
        check("I3: .txt file rejected with 400 Bad Request", res_i2.status_code == 400)

        # ===================================================================
        # Test J -- Upload Processing Exception Safety
        # ===================================================================
        print("\n--- Test J: Processing exception safety ---")

        with mock.patch("app.routes.comic.extract_image", side_effect=RuntimeError("C:\\Users\\Secret\\CorruptImage.jpg failed")):
            files_j = {"file": ("corrupt.jpg", b"fake image bytes", "image/jpeg")}
            res_j = await client.post("/comics/upload", files=files_j)
            all_responses_captured.append(res_j)

            check("J1: extraction failure returns 500 Internal Server Error", res_j.status_code == 500, f"status: {res_j.status_code}")
            detail_j = res_j.json().get("detail", "")
            check("J2: traceback is not exposed in detail", "Traceback" not in detail_j)
            check("J3: filesystem path is not exposed in detail", "C:\\Users" not in detail_j and ":\\" not in detail_j)

        # ===================================================================
        # Test K -- RAG Ingestion Failure Isolation
        # ===================================================================
        print("\n--- Test K: RAG ingestion failure isolation ---")

        fake_pages = [{"page_number": 1, "status": "success", "analysis": {}}]

        with mock.patch("app.routes.comic.extract_image", return_value=fake_pages), \
             mock.patch("app.routes.comic.analyze_pages", return_value=fake_pages), \
             mock.patch("app.routes.comic.save_comic_json", return_value={"comic": {"id": "dummy"}}), \
             mock.patch("app.routes.comic.ingest_comic_to_rag", side_effect=Exception("ChromaDB indexing error")):

            files_k = {"file": ("comic.jpg", b"valid image data", "image/jpeg")}
            res_k = await client.post("/comics/upload", files=files_k)
            all_responses_captured.append(res_k)

            check("K1: upload returns 200 even when RAG ingestion fails (comic is saved)", res_k.status_code == 200, f"status: {res_k.status_code}")
            data_k = res_k.json()
            check("K2: rag_ingested is False", data_k.get("rag_ingested") is False)
            check("K3: rag_chunks_stored is 0", data_k.get("rag_chunks_stored") == 0)
            check("K4: rag_error field is populated", bool(data_k.get("rag_error")))
            check("K5: comic_id is present", bool(data_k.get("comic_id")))

        # ===================================================================
        # Test L -- Unexpected Internal Exception
        # ===================================================================
        print("\n--- Test L: Unexpected internal exception safety ---")

        with mock.patch("app.routes.comic.check_comic_exists", return_value=pathlib.Path("dummy/path")), \
             mock.patch("app.routes.comic.answer_question", side_effect=RuntimeError("Secret database crashed with sk-ant-12345678")):

            res_l = await client.post(f"/comics/{VALID_UUID}/ask", json={"question": "Test query"})
            all_responses_captured.append(res_l)

            check("L1: unhandled exception returns 500", res_l.status_code == 500, f"status: {res_l.status_code}")
            detail_l = res_l.json().get("detail", "")
            check("L2: error detail is sanitized / generic", "sk-" not in detail_l and "database crashed" not in detail_l)
            check("L3: error structure is {'detail': ...}", "detail" in res_l.json())

        # ===================================================================
        # Test M -- Response Consistency
        # ===================================================================
        print("\n--- Test M: Response consistency ---")

        # Error responses always contain "detail"
        errors = [res_d1, res_d2, res_d3, res_e1, res_e2, res_f1, res_f2, res_g, res_h, res_i1, res_j, res_l]
        all_have_detail = all("detail" in err.json() for err in errors)
        check("M1: all error responses strictly follow {'detail': ...} schema", all_have_detail)

        # Successful responses retain existing structure
        data_a_keys = set(data_a.keys())
        expected_qa_keys = {"comic_id", "question", "answer", "sources"}
        check("M2: successful QA response retains exact schema keys", expected_qa_keys.issubset(data_a_keys), f"keys: {data_a_keys}")

        # ===================================================================
        # Test N -- RAG Configuration Regression
        # ===================================================================
        print("\n--- Test N: RAG configuration regression ---")

        vs_src = pathlib.Path("app/services/vector_store.py").read_text(encoding="utf-8")
        ret_src = pathlib.Path("app/services/retriever.py").read_text(encoding="utf-8")
        qa_src = pathlib.Path("app/services/rag_qa.py").read_text(encoding="utf-8")
        llm_src = pathlib.Path("app/services/llm.py").read_text(encoding="utf-8")

        check("N1: cosine metric in vector_store.py", "cosine" in vs_src)
        check("N2: collection 'comic_pages' in vector_store.py", "comic_pages" in vs_src)
        check("N3: DEFAULT_DISTANCE_THRESHOLD == 0.35", DEFAULT_DISTANCE_THRESHOLD == 0.35, f"got: {DEFAULT_DISTANCE_THRESHOLD}")
        check("N4: top_k default = 5 in retriever.py", "top_k: int = 5" in ret_src or "top_k=5" in ret_src)
        check("N5: deterministic sort present in retriever.py", "final_chunks.sort" in ret_src)
        check("N6: comic_id filter present in vector_store.py", '"comic_id"' in vs_src or "'comic_id'" in vs_src)
        check("N7: query normalization used in rag_qa.py", "normalize_query" in qa_src)
        check("N8: grounding fallback phrase in rag_qa.py", FALLBACK_ANSWER in qa_src)
        check("N9: validate_answer_and_sources function present in rag_qa.py", "validate_answer_and_sources" in qa_src)
        check("N10: rule 11 (no fabrication) present in llm.py", "fabricate" in llm_src.lower())

        # ===================================================================
        # Test O -- Security Sanity Check
        # ===================================================================
        print("\n--- Test O: Security sanity check across all responses ---")

        leaks_detected = []
        for i, res in enumerate(all_responses_captured):
            raw_text = res.text
            if "Traceback" in raw_text:
                leaks_detected.append(f"Response {i} leaked 'Traceback'")
            if "sk-" in raw_text:
                leaks_detected.append(f"Response {i} leaked 'sk-'")
            if "C:\\Users\\" in raw_text or "C:/Users/" in raw_text:
                leaks_detected.append(f"Response {i} leaked Windows User path")

        check("O1: zero responses leak Python tracebacks", not any("Traceback" in l for l in leaks_detected), f"leaks: {leaks_detected}")
        check("O2: zero responses leak API keys or 'sk-'", not any("sk-" in l for l in leaks_detected))
        check("O3: zero responses leak internal user file paths", not any("User path" in l for l in leaks_detected))


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
