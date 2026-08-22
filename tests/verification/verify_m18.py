"""
verify_m18.py -- M-18 Conversation Memory, Context-Aware Chat & Backend Chat API Verification Suite

Tests:
  A. Conversation creation
  B. Conversation UUID validity
  C. Conversation persistence
  D. Message persistence
  E. Message ordering
  F. GET conversation
  G. DELETE conversation
  H. Invalid conversation UUID
  I. Non-existent conversation
  J. Comic/conversation mismatch & non-existent comic
  K. Valid conversation question
  L. Follow-up question with history
  M. Conversation history passed to LLM
  N. Comic context passed to LLM
  O. Sources still come only from retrieval
  P. Wrong comic isolation
  Q. Empty question
  R. Whitespace question
  S. Empty retrieval
  T. Exact fallback behavior
  U. Source distance validation
  V. Source comic_id validation
  W. Existing endpoint regression
  X. OpenAPI documentation
  Y. Error sanitization
  Z. No duplicate RAG logic
  AA. No extra LLM call
  AB. Conversation message limit
  AC. Startup lazy-import regression
"""

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

# Results tracking
_total = 0
_passed = 0
_failed = 0
_failures = []


def check(name: str, condition: bool, info: str = "") -> bool:
    global _total, _passed, _failed
    _total += 1
    if condition:
        _passed += 1
        print(f"  [PASS] {name}" + (f" -- {info}" if info else ""))
        return True
    else:
        _failed += 1
        msg = f"  [FAIL] {name}" + (f" -- {info}" if info else "")
        print(msg)
        _failures.append(msg)
        return False


# Setup mock storage comic for testing
TEST_COMIC_ID = "11111111-2222-3333-4444-555555555555"
TEST_COMIC_DIR = Path("storage/comics") / TEST_COMIC_ID
TEST_COMIC_DIR.mkdir(parents=True, exist_ok=True)
TEST_COMIC_JSON = TEST_COMIC_DIR / "comic.json"

sample_comic_data = {
    "comic": {
        "id": TEST_COMIC_ID,
        "name": "Test Superhero",
        "source_format": "cbz",
        "total_pages": 2,
        "successful_pages": 2,
        "failed_pages": 0
    },
    "comic_content": {
        "full_text": "PAGE 1\nVictor speaks about his mother Momma.\n\nPAGE 2\nVictor reveals he built a suit."
    },
    "pages": [
        {"page_number": 1, "filename": "page_001.jpg", "status": "success"},
        {"page_number": 2, "filename": "page_002.jpg", "status": "success"}
    ]
}
with open(TEST_COMIC_JSON, "w", encoding="utf-8") as f:
    json.dump(sample_comic_data, f)

sys.path.insert(0, ".")

# Import app components
from app.main import app
from app.core.config import MAX_CONVERSATION_MESSAGES, CONVERSATIONS_DIR
from app.services.conversation import (
    create_conversation,
    get_conversation,
    append_message,
    get_messages,
    delete_conversation,
    validate_conversation_id
)
import app.services.rag_qa as rag_qa_mod
import app.services.llm as llm_mod
import app.services.retriever as retriever_mod


async def main():
    print("\n============================================================")
    print("M-18 CONVERSATION MEMORY & CHAT API VERIFICATION SUITE")
    print("============================================================\n")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

        # ---------------------------------------------------------------------------
        # Test A & B & C: Conversation Creation, UUID, Persistence
        # ---------------------------------------------------------------------------
        print("--- Test A-C: Conversation Creation & Persistence ---")
        conv_res = await client.post("/conversations", json={"comic_id": TEST_COMIC_ID})
        check("A1: POST /conversations returns 200 OK", conv_res.status_code == 200, f"status: {conv_res.status_code}")
        conv_data = conv_res.json()
        conv_id = conv_data.get("conversation_id", "")

        # UUID check
        try:
            parsed_uuid = uuid.UUID(conv_id)
            valid_uuid = str(parsed_uuid) == conv_id
        except Exception:
            valid_uuid = False
        check("B1: conversation_id is valid UUID string", valid_uuid, f"got: {conv_id}")
        check("B2: comic_id matches request", conv_data.get("comic_id") == TEST_COMIC_ID)
        check("B3: created_at and updated_at timestamps present", bool(conv_data.get("created_at") and conv_data.get("updated_at")))

        # Persistence check
        file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
        check("C1: conversation persisted to storage/conversations JSON file", file_path.exists(), f"path: {file_path}")
        record_on_disk = get_conversation(conv_id)
        check("C2: get_conversation loads record accurately", record_on_disk is not None and record_on_disk.get("conversation_id") == conv_id)
        check("C3: messages initialized as empty list", record_on_disk.get("messages") == [])

        # ---------------------------------------------------------------------------
        # Test D & E: Message Persistence & Ordering
        # ---------------------------------------------------------------------------
        print("\n--- Test D-E: Message Persistence & Ordering ---")
        append_message(conv_id, "user", "What did Victor say about his mother?")
        append_message(conv_id, "assistant", "Victor said his mother always supported him.")
        append_message(conv_id, "user", "What did he say next?")

        msgs = get_messages(conv_id)
        check("D1: 3 messages appended and retrieved", len(msgs) == 3, f"count: {len(msgs)}")
        check("D2: user and assistant roles preserved", [m["role"] for m in msgs] == ["user", "assistant", "user"])
        check("D3: message contents preserved accurately", msgs[0]["content"] == "What did Victor say about his mother?")
        check("E1: chronological message ordering preserved", msgs[1]["content"] == "Victor said his mother always supported him." and msgs[2]["content"] == "What did he say next?")

        # Reject invalid role
        role_rejected = False
        try:
            append_message(conv_id, "admin", "Invalid role message")
        except ValueError:
            role_rejected = True
        check("D4: invalid message role rejected", role_rejected)

        # ---------------------------------------------------------------------------
        # Test F: GET Conversation Details
        # ---------------------------------------------------------------------------
        print("\n--- Test F: GET Conversation Details ---")
        get_res = await client.get(f"/conversations/{conv_id}")
        check("F1: GET /conversations/{id} returns 200 OK", get_res.status_code == 200, f"status: {get_res.status_code}")
        detail = get_res.json()
        check("F2: returned detail contains conversation_id", detail.get("conversation_id") == conv_id)
        check("F3: returned detail contains 3 messages", len(detail.get("messages", [])) == 3)
        check("F4: returned message contains role, content, timestamp", all("role" in m and "content" in m and "timestamp" in m for m in detail["messages"]))

        # ---------------------------------------------------------------------------
        # Test G: DELETE Conversation
        # ---------------------------------------------------------------------------
        print("\n--- Test G: DELETE Conversation ---")
        temp_conv = create_conversation(TEST_COMIC_ID)
        temp_id = temp_conv["conversation_id"]
        del_res = await client.delete(f"/conversations/{temp_id}")
        check("G1: DELETE /conversations/{id} returns 200 OK", del_res.status_code == 200, f"status: {del_res.status_code}")
        check("G2: conversation JSON deleted from disk", not (CONVERSATIONS_DIR / f"{temp_id}.json").exists())
        check("G3: subsequent GET returns 404", (await client.get(f"/conversations/{temp_id}")).status_code == 404)
        check("G4: subsequent DELETE returns 404", (await client.delete(f"/conversations/{temp_id}")).status_code == 404)

        # ---------------------------------------------------------------------------
        # Test H & I & J: Error Handling (Invalid UUID, 404, Comic Mismatch)
        # ---------------------------------------------------------------------------
        print("\n--- Test H-J: Validation & Error Handling ---")
        inv_res = await client.get("/conversations/not-a-uuid")
        check("H1: invalid conversation UUID returns 400 Bad Request", inv_res.status_code == 400, f"status: {inv_res.status_code}")
        check("H2: error detail notes invalid UUID format", "UUID" in inv_res.json().get("detail", ""))

        non_res = await client.get(f"/conversations/{uuid.uuid4()}")
        check("I1: non-existent conversation returns 404 Not Found", non_res.status_code == 404, f"status: {non_res.status_code}")
        check("I2: detail states 'Conversation not found'", non_res.json().get("detail") == "Conversation not found")

        bad_create = await client.post("/conversations", json={"comic_id": "non-existent-comic-id"})
        check("J1: create with invalid comic UUID returns 400", bad_create.status_code == 400)
        bad_create_404 = await client.post("/conversations", json={"comic_id": str(uuid.uuid4())})
        check("J2: create with non-existent comic returns 404", bad_create_404.status_code == 404)

        # ---------------------------------------------------------------------------
        # Test K & L & M & N: Context-Aware Chat & LLM Parameter Forwarding
        # ---------------------------------------------------------------------------
        print("\n--- Test K-N: Context-Aware Chat Flow & LLM Forwarding ---")
        chat_conv = create_conversation(TEST_COMIC_ID)
        chat_id = chat_conv["conversation_id"]

        fake_chunks = [
            {
                "chunk_id": f"{TEST_COMIC_ID}_page_1_chunk_1",
                "content": "Victor loved his mother Momma deeply.",
                "metadata": {"comic_id": TEST_COMIC_ID, "page_number": 1, "chunk_index": 1, "chunk_id": f"{TEST_COMIC_ID}_page_1_chunk_1"},
                "distance": 0.22
            }
        ]

        captured_llm_calls = []

        def mock_gen_answer(question, context, conversation_history=None):
            captured_llm_calls.append({
                "question": question,
                "context": context,
                "conversation_history": conversation_history
            })
            if "mother" in question.lower() or "who" in question.lower():
                return "Victor's mother was named Momma."
            return "He said that she was his guiding light."

        with patch.object(rag_qa_mod, "retrieve_chunks", return_value=fake_chunks), \
             patch.object(rag_qa_mod, "generate_answer", side_effect=mock_gen_answer):

            # Turn 1
            q1_res = await client.post(f"/conversations/{chat_id}/ask", json={"question": "Who was Victor's mother?"})
            check("K1: Turn 1 returns 200 OK", q1_res.status_code == 200)
            q1_data = q1_res.json()
            check("K2: Turn 1 answer generated", q1_data.get("answer") == "Victor's mother was named Momma.")
            check("K3: Turn 1 conversation_id preserved", q1_data.get("conversation_id") == chat_id)
            check("K4: Turn 1 sources populated", len(q1_data.get("sources", [])) == 1)

            # Verify message history stored 2 messages
            chat_state = get_conversation(chat_id)
            check("K5: user and assistant messages stored after Turn 1", len(chat_state["messages"]) == 2)

            # Turn 2 (Follow-up)
            q2_res = await client.post(f"/conversations/{chat_id}/ask", json={"question": "What did he say about her?"})
            check("L1: Turn 2 returns 200 OK", q2_res.status_code == 200)
            q2_data = q2_res.json()
            check("L2: Turn 2 answer generated", q2_data.get("answer") == "He said that she was his guiding light.")

            # Check that LLM received conversation history on Turn 2
            check("M1: generate_answer called 2 times", len(captured_llm_calls) == 2)
            turn2_call = captured_llm_calls[1]
            hist_passed = turn2_call.get("conversation_history")
            check("M2: conversation history passed to LLM on Turn 2", hist_passed is not None and len(hist_passed) == 2)
            check("M3: Turn 1 Q and A present in history passed to LLM", 
                  hist_passed[0]["content"] == "Who was Victor's mother?" and hist_passed[1]["content"] == "Victor's mother was named Momma.")
            check("N1: [PAGE 1 | CHUNK 1] marker present in context", "[PAGE 1 | CHUNK 1]" in turn2_call["context"])

        # ---------------------------------------------------------------------------
        # Test O & P: Source Integrity & Comic Isolation
        # ---------------------------------------------------------------------------
        print("\n--- Test O-P: Source Integrity & Comic Isolation ---")
        check("O1: sources come strictly from retrieved chunks", q2_data["sources"][0]["chunk_id"] == f"{TEST_COMIC_ID}_page_1_chunk_1")
        check("O2: source has valid distance <= 0.35", q2_data["sources"][0]["distance"] == 0.22)
        check("P1: all source comic_ids match conversation comic_id", all(s["comic_id"] == TEST_COMIC_ID for s in q2_data["sources"]))

        # ---------------------------------------------------------------------------
        # Test Q & R: Empty and Whitespace Question Handling
        # ---------------------------------------------------------------------------
        print("\n--- Test Q-R: Empty / Whitespace Questions ---")
        with patch.object(rag_qa_mod, "retrieve_chunks") as mock_ret, \
             patch.object(rag_qa_mod, "generate_answer") as mock_llm:

            empty_res = await client.post(f"/conversations/{chat_id}/ask", json={"question": ""})
            check("Q1: empty question returns 200 with validation message", empty_res.status_code == 200)
            check("Q2: answer is 'Please provide a valid question.'", empty_res.json().get("answer") == "Please provide a valid question.")
            check("Q3: sources is empty list", empty_res.json().get("sources") == [])
            check("Q4: retrieval not called for empty query", mock_ret.call_count == 0)
            check("Q5: LLM not called for empty query", mock_llm.call_count == 0)

            ws_res = await client.post(f"/conversations/{chat_id}/ask", json={"question": "   \n\t  "})
            check("R1: whitespace question returns validation message", ws_res.json().get("answer") == "Please provide a valid question.")
            check("R2: sources is empty list", ws_res.json().get("sources") == [])

        # ---------------------------------------------------------------------------
        # Test S & T: Empty Retrieval & Exact Fallback
        # ---------------------------------------------------------------------------
        print("\n--- Test S-T: Empty Retrieval & Grounding Fallback ---")
        fb_conv = create_conversation(TEST_COMIC_ID)
        fb_id = fb_conv["conversation_id"]

        with patch.object(rag_qa_mod, "retrieve_chunks", return_value=[]), \
             patch.object(rag_qa_mod, "generate_answer") as mock_llm_fb:

            fb_res = await client.post(f"/conversations/{fb_id}/ask", json={"question": "What is Victor's favorite dessert?"})
            check("S1: empty retrieval returns 200 OK", fb_res.status_code == 200)
            check("S2: fallback phrase returned on empty retrieval", fb_res.json().get("answer") == "I could not find relevant information in the comic.")
            check("S3: sources is empty list on empty retrieval", fb_res.json().get("sources") == [])
            check("S4: LLM is NOT called on empty retrieval", mock_llm_fb.call_count == 0)
            check("T1: exact fallback interaction stored in history", len(get_messages(fb_id)) == 2)

        # ---------------------------------------------------------------------------
        # Test U & V: Distance & Comic ID Source Filtering
        # ---------------------------------------------------------------------------
        print("\n--- Test U-V: Distance & Comic ID Validation ---")
        bad_distance_chunks = [
            {
                "chunk_id": f"{TEST_COMIC_ID}_p1_c1",
                "content": "Valid chunk",
                "metadata": {"comic_id": TEST_COMIC_ID, "page_number": 1, "chunk_index": 1, "chunk_id": f"{TEST_COMIC_ID}_p1_c1"},
                "distance": 0.20
            },
            {
                "chunk_id": f"{TEST_COMIC_ID}_p1_c2",
                "content": "Too distant chunk",
                "metadata": {"comic_id": TEST_COMIC_ID, "page_number": 1, "chunk_index": 2, "chunk_id": f"{TEST_COMIC_ID}_p1_c2"},
                "distance": 0.85
            },
            {
                "chunk_id": "other-comic_p1_c1",
                "content": "Wrong comic chunk",
                "metadata": {"comic_id": "other-comic", "page_number": 1, "chunk_index": 1, "chunk_id": "other-comic_p1_c1"},
                "distance": 0.25
            }
        ]

        with patch.object(rag_qa_mod, "retrieve_chunks", return_value=bad_distance_chunks), \
             patch.object(rag_qa_mod, "generate_answer", return_value="Here is the grounded answer."):

            filter_res = await client.post(f"/conversations/{chat_id}/ask", json={"question": "Tell me about Victor."})
            filt_sources = filter_res.json().get("sources", [])
            check("U1: distance > threshold filtered out of sources", len(filt_sources) == 1, f"sources count: {len(filt_sources)}")
            check("U2: surviving source has valid distance", filt_sources[0]["distance"] == 0.20)
            check("V1: wrong comic_id filtered out of sources", filt_sources[0]["comic_id"] == TEST_COMIC_ID)

        # ---------------------------------------------------------------------------
        # Test W: Existing Endpoints Backwards Compatibility
        # ---------------------------------------------------------------------------
        print("\n--- Test W: Existing Endpoints Backwards Compatibility ---")
        with patch.object(rag_qa_mod, "retrieve_chunks", return_value=fake_chunks), \
             patch.object(rag_qa_mod, "generate_answer", return_value="Answer from existing route."):

            # Root /ask
            r_ask = await client.post("/ask", json={"comic_id": TEST_COMIC_ID, "question": "What happened?"})
            check("W1: root /ask returns 200 OK", r_ask.status_code == 200)

            # /comics/ask
            c_ask = await client.post("/comics/ask", json={"comic_id": TEST_COMIC_ID, "question": "What happened?"})
            check("W2: /comics/ask returns 200 OK", c_ask.status_code == 200)

            # /comics/{comic_id}/ask
            p_ask = await client.post(f"/comics/{TEST_COMIC_ID}/ask", json={"question": "What happened?"})
            check("W3: /comics/{id}/ask returns 200 OK", p_ask.status_code == 200)

            # GET /comics/{comic_id}
            g_comic = await client.get(f"/comics/{TEST_COMIC_ID}")
            check("W4: GET /comics/{id} returns 200 OK", g_comic.status_code == 200)

        # ---------------------------------------------------------------------------
        # Test X: OpenAPI Contract & Schema Verification
        # ---------------------------------------------------------------------------
        print("\n--- Test X: OpenAPI Contract & Schemas ---")
        openapi_res = await client.get("/openapi.json")
        check("X1: /openapi.json loads successfully (200 OK)", openapi_res.status_code == 200)
        openapi_doc = openapi_res.json()
        paths = openapi_doc.get("paths", {})

        check("X2: /conversations path documented in OpenAPI", "/conversations" in paths)
        check("X3: /conversations/{conversation_id} path documented in OpenAPI", "/conversations/{conversation_id}" in paths)
        check("X4: /conversations/{conversation_id}/ask path documented in OpenAPI", "/conversations/{conversation_id}/ask" in paths)

        schemas = openapi_doc.get("components", {}).get("schemas", {})
        check("X5: ConversationResponse schema documented", "ConversationResponse" in schemas)
        check("X6: ConversationDetailResponse schema documented", "ConversationDetailResponse" in schemas)
        check("X7: ConversationQuestionResponse schema documented", "ConversationQuestionResponse" in schemas)
        check("X8: SourceItem schema present in components", "SourceItem" in schemas)

        # ---------------------------------------------------------------------------
        # Test Y: Error Sanitization & Security
        # ---------------------------------------------------------------------------
        print("\n--- Test Y: Error Sanitization & Security ---")
        err_samples = [
            (await client.get("/conversations/bad-id")).text,
            (await client.post("/conversations", json={"comic_id": "bad-id"})).text,
            (await client.get(f"/conversations/{uuid.uuid4()}")).text
        ]
        check("Y1: zero error responses leak Python tracebacks", not any("Traceback" in e for e in err_samples))
        check("Y2: zero error responses leak API keys or 'sk-'", not any("sk-" in e for e in err_samples))
        check("Y3: zero error responses leak absolute server filesystem paths", not any(":\\Users\\" in e or "/home/" in e for e in err_samples))

        # ---------------------------------------------------------------------------
        # Test Z & AA: No Duplicate Logic & Single LLM Call
        # ---------------------------------------------------------------------------
        print("\n--- Test Z-AA: Architecture Cleanliness & Efficiency ---")
        with patch("app.routes.conversation.answer_question", wraps=rag_qa_mod.answer_question) as spy_qa, \
             patch.object(rag_qa_mod, "retrieve_chunks", return_value=fake_chunks), \
             patch.object(rag_qa_mod, "generate_answer", return_value="Single call answer.") as spy_llm:

            await client.post(f"/conversations/{chat_id}/ask", json={"question": "Test question."})
            check("Z1: conversation ask delegates to answer_question service", spy_qa.call_count == 1)
            check("AA1: exactly ONE LLM call made per question (no extra question rewrite LLM call)", spy_llm.call_count == 1)

        # ---------------------------------------------------------------------------
        # Test AB: Message Limit Enforcement
        # ---------------------------------------------------------------------------
        print("\n--- Test AB: Conversation Message Limit ---")
        limit_conv = create_conversation(TEST_COMIC_ID)
        limit_id = limit_conv["conversation_id"]

        for i in range(15):
            append_message(limit_id, "user", f"Question {i}")
            append_message(limit_id, "assistant", f"Answer {i}")

        recent_msgs = get_messages(limit_id, limit=MAX_CONVERSATION_MESSAGES)
        check("AB1: get_messages with limit returns exactly MAX_CONVERSATION_MESSAGES", len(recent_msgs) == MAX_CONVERSATION_MESSAGES, f"got {len(recent_msgs)}, max {MAX_CONVERSATION_MESSAGES}")
        check("AB2: recent messages contain the newest entries", recent_msgs[-1]["content"] == "Answer 14")

        # ---------------------------------------------------------------------------
        # Test AC: Startup Performance & Lazy Imports
        # ---------------------------------------------------------------------------
        print("\n--- Test AC: Startup Lazy Import Verification ---")
        check("AC1: langchain_text_splitters not loaded on app import", "langchain_text_splitters" not in sys.modules)
        check("AC2: fitz not loaded on app import", "fitz" not in sys.modules)
        check("AC3: chromadb not loaded on app import", "chromadb" not in sys.modules)

    # Cleanup test conversation files
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        try:
            conv_file.unlink()
        except Exception:
            pass

    # Cleanup test comic dir
    try:
        TEST_COMIC_JSON.unlink(missing_ok=True)
        TEST_COMIC_DIR.rmdir()
    except Exception:
        pass

    print("\n============================================================")
    print(f"M-18 Verification Results: {_passed}/{_total} checks passed ({_failed} failed)")
    if _failed == 0:
        print("VERDICT: PASS")
        sys.exit(0)
    else:
        print("VERDICT: FAIL")
        for f in _failures:
            print(" ", f)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
