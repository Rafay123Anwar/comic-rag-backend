"""
verify_integration.py -- Frontend ↔ Backend Synchronization & Integration Verification Suite

Covers:
  1. GET /comics lists all ingested comics accurately from storage
  2. GET /comics/{comic_id}/conversation creates / loads comic-scoped chat
  3. Context-aware QA persists user question & assistant answer to conversation JSON
  4. Comic scoping: Comic A and Comic B conversations remain completely isolated
  5. Persistence: Re-querying /comics/{comic_id}/conversation returns historical messages
  6. DELETE /comics/{comic_id}:
     - removes comic directory (comic.json, pages, page images)
     - removes upload files
     - removes ChromaDB chunks for that comic only
     - removes all conversation files for that comic
     - leaves other comics, chunks, and conversations intact
  7. Post-deletion guarantees:
     - GET /comics no longer includes deleted comic
     - GET /comics/{deleted_id} returns 404
     - GET /comics/{deleted_id}/conversation returns 404
     - ChromaDB search for deleted comic returns 0 results
  8. Error handling:
     - DELETE invalid UUID returns 400
     - DELETE non-existent UUID returns 404
"""
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from app.core.config import COMICS_DIR, CONVERSATIONS_DIR, UPLOADS_DIR
from app.main import app
from app.services.storage import save_comic_json
from app.services.vector_store import add_chunks, collection, search_chunks

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
        _failures.append(name)
        print(f"  [FAIL] {name}" + (f" -- {info}" if info else ""))
        return False


async def run_all_tests():
    global _total, _passed, _failed, _failures
    print("=" * 60)
    print("FRONTEND <-> BACKEND INTEGRATION VERIFICATION SUITE")
    print("=" * 60)

    comic_a_id = str(uuid.uuid4())
    comic_b_id = str(uuid.uuid4())

    try:
        # Create test mock data for Comic A
        save_comic_json(
            comic_id=comic_a_id,
            comic_name="Test Comic A",
            source_format="cbz",
            pages=[
                {
                    "page_number": 1,
                    "filename": "page_001.jpg",
                    "status": "success",
                    "analysis": {
                        "page_summary": "Hero A saving the world.",
                        "text": {"full_text": "Hero A fights villain X in city A."}
                    }
                }
            ]
        )

        # Create test mock data for Comic B
        save_comic_json(
            comic_id=comic_b_id,
            comic_name="Test Comic B",
            source_format="cbr",
            pages=[
                {
                    "page_number": 1,
                    "filename": "page_001.jpg",
                    "status": "success",
                    "analysis": {
                        "page_summary": "Hero B in space.",
                        "text": {"full_text": "Hero B travels across nebula B."}
                    }
                }
            ]
        )

        # Create dummy upload files
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        upload_a = UPLOADS_DIR / f"{comic_a_id}.cbz"
        upload_b = UPLOADS_DIR / f"{comic_b_id}.cbr"
        upload_a.write_text("dummy comic a content")
        upload_b.write_text("dummy comic b content")

        # Insert dummy vector chunks for Comic A and Comic B
        embedding_dim = 1024
        dummy_vec_a = [0.1] * embedding_dim
        dummy_vec_b = [0.2] * embedding_dim

        add_chunks(
            chunks=[{
                "chunk_id": f"{comic_a_id}_p1_c1",
                "content": "Hero A fights villain X in city A.",
                "metadata": {"comic_id": comic_a_id, "page_number": 1, "chunk_index": 0}
            }],
            embeddings=[dummy_vec_a]
        )

        add_chunks(
            chunks=[{
                "chunk_id": f"{comic_b_id}_p1_c1",
                "content": "Hero B travels across nebula B.",
                "metadata": {"comic_id": comic_b_id, "page_number": 1, "chunk_index": 0}
            }],
            embeddings=[dummy_vec_b]
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # ----------------------------------------------------
            # TEST 1 & 2: GET /comics lists both A and B
            # ----------------------------------------------------
            print("\n--- TEST 1 & 2: GET /comics list synchronization ---")
            res = await client.get("/comics")
            check("GET /comics returns 200 OK", res.status_code == 200)
            comics = res.json()
            comic_ids = [c["comic_id"] for c in comics]
            check("Comic A present in GET /comics", comic_a_id in comic_ids)
            check("Comic B present in GET /comics", comic_b_id in comic_ids)

            comic_a_entry = next(c for c in comics if c["comic_id"] == comic_a_id)
            check("Comic A title preserved", comic_a_entry["title"] == "Test Comic A")
            check("Comic A format preserved", comic_a_entry["source_format"] == "cbz")

            # ----------------------------------------------------
            # TEST 3: Select A -> Load conversation for Comic A
            # ----------------------------------------------------
            print("\n--- TEST 3: Select Comic A conversation initialization ---")
            conv_res_a = await client.get(f"/comics/{comic_a_id}/conversation")
            check("GET /comics/{comic_a_id}/conversation returns 200 OK", conv_res_a.status_code == 200)
            conv_data_a = conv_res_a.json()
            conv_a_id = conv_data_a["conversation_id"]
            check("Conversation A has valid UUID", len(conv_a_id) == 36)
            check("Conversation A comic_id matches Comic A", conv_data_a["comic_id"] == comic_a_id)
            check("Conversation A initially has 0 messages", len(conv_data_a["messages"]) == 0)

            # ----------------------------------------------------
            # TEST 4: Ask Question on Comic A
            # ----------------------------------------------------
            print("\n--- TEST 4: Ask question on Comic A & persistence ---")
            with patch("app.services.rag_qa.retrieve_chunks") as mock_retrieve, \
                 patch("app.services.rag_qa.generate_answer") as mock_generate:
                mock_retrieve.return_value = [{
                    "chunk_id": f"{comic_a_id}_p1_c1",
                    "content": "Hero A fights villain X in city A.",
                    "metadata": {
                        "comic_id": comic_a_id,
                        "page_number": 1,
                        "chunk_id": f"{comic_a_id}_p1_c1",
                        "chunk_index": 0
                    },
                    "distance": 0.15
                }]
                mock_generate.return_value = "Hero A is fighting villain X."

                ask_res = await client.post(
                    f"/conversations/{conv_a_id}/ask",
                    json={"question": "Who is Hero A fighting?"}
                )
                check("POST /conversations/{id}/ask returns 200 OK", ask_res.status_code == 200)
                ans_data = ask_res.json()
                check("Answer returned correctly", ans_data["answer"] == "Hero A is fighting villain X.")

            # ----------------------------------------------------
            # TEST 5: Refresh / re-fetch Comic A conversation
            # ----------------------------------------------------
            print("\n--- TEST 5: Browser refresh / persistent chat recovery ---")
            reload_res_a = await client.get(f"/comics/{comic_a_id}/conversation")
            check("GET /comics/{comic_a_id}/conversation reloaded successfully", reload_res_a.status_code == 200)
            reloaded_a = reload_res_a.json()
            check("Same conversation_id retained", reloaded_a["conversation_id"] == conv_a_id)
            check("Conversation has 2 persisted messages (user + assistant)", len(reloaded_a["messages"]) == 2)
            check("User message content matches", reloaded_a["messages"][0]["content"] == "Who is Hero A fighting?")
            check("Assistant message content matches", reloaded_a["messages"][1]["content"] == "Hero A is fighting villain X.")

            # ----------------------------------------------------
            # TEST 6: Switch A -> B
            # ----------------------------------------------------
            print("\n--- TEST 6: Switch to Comic B (Scoping check) ---")
            conv_res_b = await client.get(f"/comics/{comic_b_id}/conversation")
            check("GET /comics/{comic_b_id}/conversation returns 200 OK", conv_res_b.status_code == 200)
            conv_data_b = conv_res_b.json()
            conv_b_id = conv_data_b["conversation_id"]
            check("Comic B has distinct conversation_id", conv_b_id != conv_a_id)
            check("Comic B messages are empty initially", len(conv_data_b["messages"]) == 0)

            # Add question to Comic B
            with patch("app.services.rag_qa.retrieve_chunks") as mock_retrieve, \
                 patch("app.services.rag_qa.generate_answer") as mock_generate:
                mock_retrieve.return_value = [{
                    "chunk_id": f"{comic_b_id}_p1_c1",
                    "content": "Hero B travels across nebula B.",
                    "metadata": {
                        "comic_id": comic_b_id,
                        "page_number": 1,
                        "chunk_id": f"{comic_b_id}_p1_c1",
                        "chunk_index": 0
                    },
                    "distance": 0.12
                }]
                mock_generate.return_value = "Hero B is in space."

                await client.post(
                    f"/conversations/{conv_b_id}/ask",
                    json={"question": "Where is Hero B?"}
                )

            # ----------------------------------------------------
            # TEST 7: Switch back B -> A
            # ----------------------------------------------------
            print("\n--- TEST 7: Switch back B -> A (History reappears intact) ---")
            switch_back_a = await client.get(f"/comics/{comic_a_id}/conversation")
            check("Switch back to Comic A returns 200 OK", switch_back_a.status_code == 200)
            msgs_a = switch_back_a.json()["messages"]
            check("Comic A still has exactly 2 messages", len(msgs_a) == 2)
            check("Comic A messages contain Comic A question", msgs_a[0]["content"] == "Who is Hero A fighting?")

            # ----------------------------------------------------
            # TEST 8 & 9: DELETE Comic A
            # ----------------------------------------------------
            print("\n--- TEST 8 & 9: DELETE Comic A full synchronization ---")
            del_res = await client.delete(f"/comics/{comic_a_id}")
            check("DELETE /comics/{comic_a_id} returns 200 OK", del_res.status_code == 200)
            check("DELETE response confirms deletion", del_res.json()["comic_id"] == comic_a_id)

            # Verify physical storage deleted
            comic_a_dir = Path(COMICS_DIR) / comic_a_id
            check("Comic A directory removed from storage/comics", not comic_a_dir.exists())
            check("Comic A upload file removed from storage/uploads", not upload_a.exists())

            # Verify conversation deleted
            conv_a_file = Path(CONVERSATIONS_DIR) / f"{conv_a_id}.json"
            check("Comic A conversation JSON removed from storage/conversations", not conv_a_file.exists())

            # Verify ChromaDB chunks for Comic A deleted
            docs_a = collection.get(where={"comic_id": comic_a_id})
            check("ChromaDB chunks for Comic A removed (0 chunks)", len(docs_a.get("ids", [])) == 0)

            # Verify Comic B resources and ChromaDB chunks remain intact
            comic_b_dir = Path(COMICS_DIR) / comic_b_id
            check("Comic B directory still exists", comic_b_dir.exists())
            check("Comic B upload file still exists", upload_b.exists())
            conv_b_file = Path(CONVERSATIONS_DIR) / f"{conv_b_id}.json"
            check("Comic B conversation file still exists", conv_b_file.exists())
            docs_b = collection.get(where={"comic_id": comic_b_id})
            check("ChromaDB chunks for Comic B still present", len(docs_b.get("ids", [])) == 1)

            # Verify GET /comics no longer includes A
            list_after_del = await client.get("/comics")
            ids_after = [c["comic_id"] for c in list_after_del.json()]
            check("Deleted Comic A not present in GET /comics", comic_a_id not in ids_after)
            check("Comic B still present in GET /comics", comic_b_id in ids_after)

            # Verify GET /comics/{comic_a_id} returns 404
            get_deleted_comic = await client.get(f"/comics/{comic_a_id}")
            check("GET /comics/{deleted_id} returns 404 Not Found", get_deleted_comic.status_code == 404)

            # Verify GET /comics/{comic_a_id}/conversation returns 404
            get_deleted_conv = await client.get(f"/comics/{comic_a_id}/conversation")
            check("GET /comics/{deleted_id}/conversation returns 404 Not Found", get_deleted_conv.status_code == 404)

            # ----------------------------------------------------
            # TEST 10: Delete a comic while another is selected
            # ----------------------------------------------------
            print("\n--- TEST 10: Comic B remains fully functional ---")
            conv_b_check = await client.get(f"/comics/{comic_b_id}/conversation")
            check("Comic B conversation loads with its 2 messages", len(conv_b_check.json()["messages"]) == 2)

            # ----------------------------------------------------
            # TEST 11: Error handling on invalid/non-existent delete
            # ----------------------------------------------------
            print("\n--- TEST 11: DELETE error handling ---")
            del_again = await client.delete(f"/comics/{comic_a_id}")
            check("DELETE on non-existent comic returns 404", del_again.status_code == 404)
            del_invalid = await client.delete("/comics/not-a-uuid")
            check("DELETE on invalid UUID format returns 400", del_invalid.status_code == 400)

            # Clean up Comic B
            await client.delete(f"/comics/{comic_b_id}")

    finally:
        # Clean up any leftover test files
        for test_id in [comic_a_id, comic_b_id]:
            shutil.rmtree(Path(COMICS_DIR) / test_id, ignore_errors=True)
            for f in UPLOADS_DIR.glob(f"{test_id}.*"):
                f.unlink(missing_ok=True)
            for f in CONVERSATIONS_DIR.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        d = json.load(file)
                        if d.get("comic_id") == test_id:
                            f.unlink(missing_ok=True)
                except Exception:
                    pass

    print("\n" + "=" * 60)
    print(f"Integration Verification Results: {_passed}/{_total} passed ({_failed} failed)")
    print("=" * 60)
    if _failed > 0:
        print("FAILURES:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("VERDICT: ALL TESTS PASSED")


if __name__ == "__main__":
    import sys
    asyncio.run(run_all_tests())
