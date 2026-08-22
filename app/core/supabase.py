import logging
import mimetypes
import threading
import time
from pathlib import Path
from typing import Optional

from app.core.config import (
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_URL,
)

logger = logging.getLogger("comic_rag")

_thread_local = threading.local()


def get_supabase_client():
    """
    Returns a thread-local instance of the Supabase Client.
    Thread-local isolation prevents HTTP/2 connection collisions during concurrent file uploads.
    """
    if hasattr(_thread_local, "client") and _thread_local.client is not None:
        return _thread_local.client

    if not SUPABASE_URL:
        return None

    api_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if not api_key:
        return None

    try:
        from supabase import create_client
        _thread_local.client = create_client(SUPABASE_URL, api_key)
        return _thread_local.client
    except Exception as e:
        logger.warning("[SUPABASE] Failed to initialize Supabase client: %s", str(e))
        return None


def reset_supabase_client():
    """Resets the thread-local Supabase client connection."""
    _thread_local.client = None


def is_supabase_storage_enabled() -> bool:
    """Returns True if Supabase storage credentials are configured."""
    return get_supabase_client() is not None


def ensure_bucket_exists(bucket_name: str = SUPABASE_STORAGE_BUCKET) -> bool:
    """
    Ensures the private storage bucket exists in Supabase.
    """
    client = get_supabase_client()
    if not client:
        return False

    try:
        buckets = client.storage.list_buckets()
        existing_names = [b.name for b in buckets] if buckets else []
        if bucket_name not in existing_names:
            client.storage.create_bucket(
                bucket_name,
                options={"public": False}
            )
            logger.info("[SUPABASE] Created private storage bucket '%s'", bucket_name)
        return True
    except Exception as e:
        logger.warning("[SUPABASE] Could not ensure bucket '%s': %s", bucket_name, str(e))
        return False


def upload_file_to_storage(
    storage_path: str,
    local_file_path: Path,
    bucket_name: str = SUPABASE_STORAGE_BUCKET,
    content_type: Optional[str] = None,
    max_retries: int = 3
) -> Optional[str]:
    """
    Uploads a local file to Supabase Storage with retry logic.
    Returns the storage_path if successful, None otherwise.
    """
    client = get_supabase_client()
    if not client:
        return None

    if not local_file_path.exists() or not local_file_path.is_file():
        logger.warning("[SUPABASE] Local file not found for upload: %s", local_file_path)
        return None

    mime = content_type or mimetypes.guess_type(str(local_file_path))[0] or "application/octet-stream"
    file_bytes = local_file_path.read_bytes()
    cleaned_path = storage_path.lstrip("/")

    for attempt in range(1, max_retries + 1):
        try:
            client = get_supabase_client()
            client.storage.from_(bucket_name).upload(
                path=cleaned_path,
                file=file_bytes,
                file_options={"content-type": mime, "upsert": "true"}
            )
            return cleaned_path
        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
                # Reset thread-local client to recover from broken socket
                _thread_local.client = None
            else:
                logger.warning("[SUPABASE] Upload failed for %s -> %s (attempt %d): %s", local_file_path, storage_path, attempt, str(e))
    return None


def upload_bytes_to_storage(
    storage_path: str,
    data: bytes,
    bucket_name: str = SUPABASE_STORAGE_BUCKET,
    content_type: str = "application/octet-stream"
) -> Optional[str]:
    """
    Uploads raw bytes to Supabase Storage.
    Returns storage_path if successful, None otherwise.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        cleaned_path = storage_path.lstrip("/")
        client.storage.from_(bucket_name).upload(
            path=cleaned_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        return cleaned_path
    except Exception as e:
        logger.warning("[SUPABASE] Upload bytes failed for %s: %s", storage_path, str(e))
        return None


def download_bytes_from_storage(
    storage_path: str,
    bucket_name: str = SUPABASE_STORAGE_BUCKET
) -> Optional[bytes]:
    """
    Downloads object bytes from Supabase Storage.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        cleaned_path = storage_path.lstrip("/")
        data = client.storage.from_(bucket_name).download(cleaned_path)
        return data
    except Exception as e:
        logger.warning("[SUPABASE] Download failed for %s: %s", storage_path, str(e))
        return None


def delete_storage_files(
    storage_paths: list[str],
    bucket_name: str = SUPABASE_STORAGE_BUCKET
) -> bool:
    """
    Deletes a list of file paths from Supabase Storage.
    """
    client = get_supabase_client()
    if not client or not storage_paths:
        return False

    try:
        cleaned_paths = [p.lstrip("/") for p in storage_paths if p]
        if cleaned_paths:
            client.storage.from_(bucket_name).remove(cleaned_paths)
        return True
    except Exception as e:
        logger.warning("[SUPABASE] Deletion failed for %d files: %s", len(storage_paths), str(e))
        return False


def get_signed_storage_url(
    storage_path: str,
    expires_in: int = 3600,
    bucket_name: str = SUPABASE_STORAGE_BUCKET
) -> Optional[str]:
    """
    Generates a secure, time-limited signed URL for private asset access.
    """
    client = get_supabase_client()
    if not client or not storage_path:
        return None

    try:
        cleaned_path = storage_path.lstrip("/")
        res = client.storage.from_(bucket_name).create_signed_url(cleaned_path, expires_in)
        if isinstance(res, dict) and "signedURL" in res:
            return res["signedURL"]
        if hasattr(res, "signed_url"):
            return res.signed_url
        return None
    except Exception as e:
        logger.warning("[SUPABASE] Signed URL creation failed for %s: %s", storage_path, str(e))
        return None


def get_signed_storage_urls(
    storage_paths: list[str],
    expires_in: int = 3600,
    bucket_name: str = SUPABASE_STORAGE_BUCKET
) -> dict[str, Optional[str]]:
    """
    Generates time-limited signed URLs for a batch of private storage paths in a single request.
    Returns a dictionary mapping storage_path -> signed_url (or None if unavailable/error).
    """
    client = get_supabase_client()
    if not client or not storage_paths:
        return {p: None for p in storage_paths}

    results: dict[str, Optional[str]] = {}
    clean_to_orig = {p.lstrip("/"): p for p in storage_paths if p}
    cleaned_paths = list(clean_to_orig.keys())

    if not cleaned_paths:
        return results

    # Batch in chunks of 100
    batch_size = 100
    for i in range(0, len(cleaned_paths), batch_size):
        chunk = cleaned_paths[i:i + batch_size]
        try:
            res_list = client.storage.from_(bucket_name).create_signed_urls(chunk, expires_in)
            if isinstance(res_list, list):
                for item in res_list:
                    if isinstance(item, dict):
                        p = item.get("path", "")
                        orig_p = clean_to_orig.get(p, p)
                        signed_u = item.get("signedURL") or item.get("signedUrl")
                        results[orig_p] = signed_u
                    elif hasattr(item, "path"):
                        p = getattr(item, "path", "")
                        orig_p = clean_to_orig.get(p, p)
                        signed_u = getattr(item, "signed_url", None) or getattr(item, "signedURL", None)
                        results[orig_p] = signed_u
        except Exception as e:
            logger.warning("[SUPABASE] Batch signed URLs creation failed (%d paths): %s", len(chunk), str(e))
            for p in chunk:
                orig_p = clean_to_orig.get(p, p)
                results[orig_p] = None

    # Ensure all requested paths are present in output dictionary
    for p in storage_paths:
        if p not in results:
            results[p] = None

    return results
