"""
verify_m11.py -- M-11 Query Normalization Verification Script

Tests A through E are self-contained unit tests that do NOT call
the embedding model, vector store, or LLM.  They verify only the
normalization logic and the contract between rag_qa.answer_question
and its dependencies via lightweight mocking.

Usage
-----
    python verify_m11.py
"""

import sys
import re
import types
import importlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "[PASS]" if condition else "[FAIL]"
    print(f"  {icon} {name}" + (f" -- {detail}" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# Import the module under test (pure Python, no I/O)
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

# Step 1: Pre-stub third-party libs that will be imported transitively
# by app.services modules.  Do this BEFORE importing anything from app.*
# so that those imports succeed without real API keys or ChromaDB.

def _make_stub(module_path, attrs):
    """Register a fake module in sys.modules."""
    parts = module_path.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pm = types.ModuleType(parent)
            pm.__path__ = []
            pm.__package__ = parent
            sys.modules[parent] = pm
    mod = types.ModuleType(module_path)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[module_path] = mod
    parent = ".".join(parts[:-1])
    if parent in sys.modules:
        setattr(sys.modules[parent], parts[-1], mod)
    return mod


# Third-party stubs (chromadb, mistralai, dotenv)
_make_stub("chromadb", {"PersistentClient": lambda **kw: None})
_make_stub("mistralai", {})
_make_stub("mistralai.client", {"Mistral": lambda api_key=None: None})
_make_stub("dotenv", {"load_dotenv": lambda: None})

# app.config stub (consumed at top of embedding.py, llm.py, etc.)
_make_stub("app.config", {
    "MISTRAL_API_KEY": "stub",
    "MAX_AI_WORKERS": 3,
    "MAX_AI_RETRIES": 5,
})

# Step 2: Import the REAL modules that we actually want to test.
# These only depend on stdlib 're', so they import cleanly now.
import importlib.util as _ilu
import os as _os

def _load_real(dotted, filepath):
    """Load a real .py file as a module under the given dotted name."""
    spec = _ilu.spec_from_file_location(dotted, filepath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[dotted] = mod
    # attach to parent package
    parts = dotted.split(".")
    parent = ".".join(parts[:-1])
    if parent in sys.modules:
        setattr(sys.modules[parent], parts[-1], mod)
    spec.loader.exec_module(mod)
    return mod

_BASE = _os.path.abspath("app/services")

# Ensure app and app.services are real package entries pointing at disk
for _pkg, _dir in [("app", _os.path.abspath("app")),
                   ("app.services", _BASE)]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [_dir]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

normalize_query_mod = _load_real(
    "app.services.query_normalizer",
    _os.path.join(_BASE, "query_normalizer.py"),
)
from app.services.query_normalizer import normalize_query  # noqa: E402

# Step 3: Stub the remaining app.services dependencies that rag_qa imports,
# AFTER the real modules are loaded so stubs don't block disk access.

_SENTINEL_CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "content": "Victor spoke warmly about his mother.",
        "metadata": {
            "comic_id": "comic-abc",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_id": "chunk-1",
        },
        "distance": 0.20,
    }
]

_last_retrieve_call = {}  # records arguments passed to retrieve_chunks


def _stub_retrieve_chunks(query, comic_id, top_k=5, distance_threshold=0.35):
    _last_retrieve_call.update(
        query=query,
        comic_id=comic_id,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )
    return _SENTINEL_CHUNKS


_make_stub("app.services.retriever", {
    "DEFAULT_DISTANCE_THRESHOLD": 0.35,
    "retrieve_chunks": _stub_retrieve_chunks,
})


def _stub_generate_answer(question, context):
    return f"Answer based on context for: {question}"


_make_stub("app.services.llm", {"generate_answer": _stub_generate_answer})
_make_stub("app.services.embedding", {
    "generate_embedding": lambda text: [0.1] * 10,
})
_make_stub("app.services.vector_store", {
    "search_chunks": lambda **kw: {},
    "reset_collection": lambda: None,
    "collection": None,
})

# Step 4: Load rag_qa from disk (its imports are now all satisfied by stubs)
rag_qa_mod = _load_real(
    "app.services.rag_qa",
    _os.path.join(_BASE, "rag_qa.py"),
)




# ===========================================================================
# Test A -- Normal query: behavior is unchanged
# ===========================================================================

print("\n--- Test A: Normal query ---")
_last_retrieve_call.clear()

clean_q = "What did Victor say about his mother?"
result_a = rag_qa_mod.answer_question(
    question=clean_q,
    comic_id="comic-abc",
)

check("A1: question preserved in response", result_a["question"] == clean_q,
      f"got: {result_a['question']!r}")
check("A2: retrieve_chunks called with exact clean query",
      _last_retrieve_call.get("query") == clean_q,
      f"embedded query: {_last_retrieve_call.get('query')!r}")
check("A3: top_k=5 forwarded", _last_retrieve_call.get("top_k") == 5,
      f"top_k: {_last_retrieve_call.get('top_k')}")
check("A4: distance_threshold=0.35 forwarded",
      _last_retrieve_call.get("distance_threshold") == 0.35,
      f"threshold: {_last_retrieve_call.get('distance_threshold')}")
check("A5: comic_id filter forwarded",
      _last_retrieve_call.get("comic_id") == "comic-abc",
      f"comic_id: {_last_retrieve_call.get('comic_id')!r}")


# ===========================================================================
# Test B -- Whitespace-heavy query: same embedding as Test A
# ===========================================================================

print("\n--- Test B: Whitespace-heavy query ---")
_last_retrieve_call.clear()

messy_q = "   What   did   Victor   say   about   his mother?   "
result_b = rag_qa_mod.answer_question(
    question=messy_q,
    comic_id="comic-abc",
)

norm_b = _last_retrieve_call.get("query", "")
norm_a = normalize_query(clean_q)

check("B1: original question preserved in response",
      result_b["question"] == messy_q,
      f"got: {result_b['question']!r}")
check("B2: embedded query equals normalized clean query",
      norm_b == normalize_query(messy_q),
      f"embedded: {norm_b!r}  expected: {normalize_query(messy_q)!r}")
check("B3: Test A and Test B produce same embedded query",
      normalize_query(messy_q) == normalize_query(clean_q),
      f"A-normalized: {normalize_query(clean_q)!r}  "
      f"B-normalized: {normalize_query(messy_q)!r}")


# ===========================================================================
# Test C -- Empty query: no embedding, no search, sources=[]
# ===========================================================================

print("\n--- Test C: Empty query ---")
_last_retrieve_call.clear()

for label, empty_q in [
    ("empty string", ""),
    ("whitespace only", "     "),
    ("tabs only", "\t\t"),
]:
    result_c = rag_qa_mod.answer_question(
        question=empty_q,
        comic_id="comic-abc",
    )
    check(f"C1[{label}]: sources=[]",
          result_c["sources"] == [],
          f"sources: {result_c['sources']}")
    check(f"C2[{label}]: no retrieve_chunks call",
          _last_retrieve_call == {},
          f"retrieve call recorded: {_last_retrieve_call}")
    check(f"C3[{label}]: validation answer returned",
          "valid" in result_c["answer"].lower() or
          "provide" in result_c["answer"].lower(),
          f"answer: {result_c['answer']!r}")
    _last_retrieve_call.clear()


# ===========================================================================
# Test D -- Named-entity preservation: "Victor" stays "Victor"
# ===========================================================================

print("\n--- Test D: Named-entity preservation ---")

entity_queries = [
    "What did Victor say?",
    "  Victor  and  his  mother  ",
    "Tell me about Victor's childhood.",
]
for q in entity_queries:
    norm = normalize_query(q)
    check(f"D: 'Victor' preserved in {q!r}",
          "Victor" in norm,
          f"normalized: {norm!r}")

check("D: normalize_query does NOT add extra words",
      normalize_query("Victor") == "Victor",
      f"got: {normalize_query('Victor')!r}")


# ===========================================================================
# Test E -- Regression: config constants remain unchanged
# ===========================================================================

print("\n--- Test E: Regression checks ---")

from app.services.retriever import DEFAULT_DISTANCE_THRESHOLD  # noqa: E402
from app.services.vector_store import collection  # noqa: E402 (may be None stub)

check("E1: DEFAULT_DISTANCE_THRESHOLD == 0.35",
      DEFAULT_DISTANCE_THRESHOLD == 0.35,
      f"got: {DEFAULT_DISTANCE_THRESHOLD}")

# Verify vector_store collection name and metric via module source text
import inspect, pathlib  # noqa: E402

vs_src = pathlib.Path("app/services/vector_store.py").read_text(encoding="utf-8")
check("E2: collection name is 'comic_pages'",
      "comic_pages" in vs_src,
      "searched vector_store.py source")
check("E3: cosine metric configured",
      "cosine" in vs_src,
      "searched vector_store.py source")
check("E4: comic_id filter in search_chunks",
      "comic_id" in vs_src,
      "searched vector_store.py source")

# Verify retriever top_k default
import inspect as _inspect  # noqa: E402
retriever_src = pathlib.Path("app/services/retriever.py").read_text(encoding="utf-8")
check("E5: top_k default = 5 in retriever.py",
      "top_k: int = 5" in retriever_src or "top_k=5" in retriever_src,
      "searched retriever.py source")
check("E6: deterministic ordering present",
      "final_chunks.sort" in retriever_src,
      "searched retriever.py source")

# Verify normalize_query is pure (no I/O, no LLM)
normalizer_src = pathlib.Path("app/services/query_normalizer.py").read_text(encoding="utf-8")
check("E7: query_normalizer imports only 're' (no LLM/HTTP imports)",
      "import re" in normalizer_src and
      "mistral" not in normalizer_src.lower() and
      "openai" not in normalizer_src.lower() and
      "requests" not in normalizer_src.lower(),
      "searched query_normalizer.py source")


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = total - passed

print(f"Results: {passed}/{total} checks passed")
for name, status, detail in results:
    if status == FAIL:
        print(f"  [FAIL] {name} -- {detail}")

print()
if failed == 0:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL ({failed} check(s) failed)")

sys.exit(0 if failed == 0 else 1)
