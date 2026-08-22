import sys
from pathlib import Path

# Configure utf-8 encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.supabase import (
    get_supabase_client,
    ensure_bucket_exists,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_URL,
)
from app.services.vector_store import (
    get_active_backend_name,
    is_supabase_vector_enabled,
)

print("=" * 60)
print("SUPABASE INTEGRATION & VECTOR STORE DIAGNOSTIC")
print("=" * 60)

client = get_supabase_client()

if client is None:
    print("[ERROR] Supabase client NOT connected.")
    print("   Please ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY are set in backend/.env.")
else:
    print(f"[OK] Supabase client initialized (URL: {SUPABASE_URL})")

    # 1. Storage Bucket Test
    bucket_ok = ensure_bucket_exists()
    if bucket_ok:
        print(f"[OK] Storage Bucket '{SUPABASE_STORAGE_BUCKET}' is accessible.")
    else:
        print(f"[ERROR] Storage Bucket '{SUPABASE_STORAGE_BUCKET}' is NOT accessible.")

    # 2. Vector Table Test (comic_page_chunks)
    try:
        table_res = client.table("comic_page_chunks").select("id").limit(1).execute()
        print("[OK] Vector Table 'comic_page_chunks' is accessible in Supabase.")
    except Exception as e:
        print(f"[WARNING] Vector Table 'comic_page_chunks' NOT found or error: {e}")
        print("   -> Run 'supabase_pgvector_setup.sql' in Supabase SQL Editor to create it.")

    # 3. Vector Search RPC Test (match_comic_chunks)
    try:
        # Dummy 1024-dim zero vector test
        dummy_embedding = [0.0] * 1024
        rpc_res = client.rpc(
            "match_comic_chunks",
            {
                "query_embedding": dummy_embedding,
                "match_threshold": 1.0,
                "match_count": 1,
                "p_comic_id": "test",
            }
        ).execute()
        print("[OK] Vector RPC 'match_comic_chunks' is working properly.")
    except Exception as e:
        print(f"[WARNING] Vector RPC 'match_comic_chunks' error: {e}")
        print("   -> Run 'supabase_pgvector_setup.sql' in Supabase SQL Editor to define the function.")

print("-" * 60)
active_backend = get_active_backend_name()
print(f"Current Active Vector Backend: [{active_backend.upper()}]")
if active_backend == "supabase":
    print("[ACTIVE] All comic page chunks and search queries will use Supabase pgvector!")
else:
    print("[FALLBACK] Operating in local ChromaDB mode. Once Supabase SQL setup is run, it will automatically switch to Supabase.")
print("=" * 60)