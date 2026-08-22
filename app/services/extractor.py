import os
import shutil
import subprocess as sp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    COMICS_DIR,
    SEVEN_ZIP_PATH,
    TEMP_DIR,
)
from app.core.logging import logger

IMAGE_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS
SEVEN_ZIP = SEVEN_ZIP_PATH

# Maximum dimension and compression settings for instant low-resolution progressive preview
THUMBNAIL_MAX_SIZE = (480, 720)
THUMBNAIL_JPEG_QUALITY = 45


# ============================================================
# Dual-Resolution Thumbnail & Preview Helpers
# ============================================================

def create_page_thumbnail(img_path: Path, thumb_file: Path) -> bool:
    """
    Generates a compressed, low-resolution JPEG thumbnail (~20-30KB) from a high-res page image.
    Optimized for instant initial render, progressive loading, and sidebar navigation.
    """
    try:
        if not img_path.exists() or img_path.stat().st_size == 0:
            return False

        thumb_file.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as im:
            if im.mode in ("RGBA", "P", "LA", "CMYK"):
                im = im.convert("RGB")
            im.thumbnail(THUMBNAIL_MAX_SIZE, Image.Resampling.BILINEAR)
            im.save(thumb_file, "JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True, progressive=True)
        return True
    except Exception as err:
        logger.warning("[THUMBNAIL] Error creating thumbnail %s from %s: %s", thumb_file.name, img_path.name, err)
        return False


def ensure_page_thumbnail(comic_id: str, page_number: int, local_img_path: Path) -> Path | None:
    """
    Ensures a lightweight JPEG thumbnail exists for a page image.
    Returns Path to thumbnail if generated or existing, None if generation failed.
    """
    try:
        thumb_dir = (Path(COMICS_DIR) / comic_id / "thumbnails").resolve()
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_file = thumb_dir / f"thumb_p{page_number:03d}.jpg"
        if thumb_file.exists() and thumb_file.stat().st_size > 0:
            return thumb_file

        if not local_img_path.exists():
            return None

        success = create_page_thumbnail(local_img_path, thumb_file)
        return thumb_file if success else None
    except Exception as e:
        logger.warning("[THUMBNAIL] Failed to ensure thumbnail for page %d: %s", page_number, str(e))
        return None


def generate_all_thumbnails(comic_id: str, pages: list[dict]) -> None:
    """
    Generates lightweight JPEG thumbnails in parallel for all extracted comic pages.
    Populates 'thumbnail_path' for every page dictionary.
    """
    try:
        thumb_dir = (Path(COMICS_DIR) / comic_id / "thumbnails").resolve()
        thumb_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for p in pages:
            page_num = p.get("page_number")
            img_path_str = p.get("image_path")
            if not page_num or not img_path_str:
                continue
            img_path = Path(img_path_str)
            thumb_file = thumb_dir / f"thumb_p{page_num:03d}.jpg"
            p["thumbnail_path"] = str(thumb_file)
            tasks.append((img_path, thumb_file))

        def _worker(item):
            img_p, th_p = item
            if not th_p.exists() or th_p.stat().st_size == 0:
                create_page_thumbnail(img_p, th_p)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_worker, tasks))

    except Exception as e:
        logger.warning("Failed during bulk thumbnail generation for comic %s: %s", comic_id, e)


# ============================================================
# CBR / CBZ Archives
# ============================================================

def extract_cbr(cbr_path: str, comic_id: str) -> list[dict]:
    """
    Extracts pages from a CBR archive.
    """
    return extract_archive(
        archive_path=cbr_path,
        comic_id=comic_id,
        archive_type="CBR"
    )


def extract_cbz(cbz_path: str, comic_id: str) -> list[dict]:
    """
    Extracts pages from a CBZ archive.
    """
    return extract_archive(
        archive_path=cbz_path,
        comic_id=comic_id,
        archive_type="CBZ"
    )


def extract_archive(
    archive_path: str,
    comic_id: str,
    archive_type: str
) -> list[dict]:
    """
    Extracts comic pages from an archive (CBR/CBZ) using 7-Zip.
    Generates dual assets: high-res original pages + low-res progressive thumbnails.
    """
    if not os.path.exists(SEVEN_ZIP):
        raise FileNotFoundError(
            f"7-Zip executable not found at: {SEVEN_ZIP}"
        )

    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"Comic archive file not found: {archive_path}"
        )

    pages_dir = Path(COMICS_DIR) / comic_id / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(TEMP_DIR) / comic_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    try:
        # Extract archive contents to temporary directory
        sp.run(
            [
                SEVEN_ZIP,
                "x",
                archive_path,
                f"-o{temp_dir}",
                "-y"
            ],
            check=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True
        )

        # Collect all valid image files
        image_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                extension = Path(file).suffix.lower()
                if extension in IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / file)

        if not image_files:
            raise ValueError("No valid comic page images found in archive.")

        # Sort images alphabetically / deterministically
        image_files.sort(key=lambda path: path.as_posix().lower())

        pages = []
        for index, image_path in enumerate(image_files, start=1):
            extension = image_path.suffix.lower()
            page_filename = f"page_{index:03d}{extension}"
            destination = pages_dir / page_filename

            shutil.copy2(image_path, destination)
            pages.append(
                {
                    "page_number": index,
                    "filename": page_filename,
                    "image_path": str(destination),
                }
            )

        generate_all_thumbnails(comic_id, pages)

        duration = time.perf_counter() - start_time
        logger.info(
            "[PERF] Extraction completed: %s pages in %.2fs (with dual-resolution thumbnails)",
            len(pages),
            duration
        )

        return pages

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# PDF Documents
# ============================================================

def extract_pdf(pdf_path: str, comic_id: str) -> list[dict]:
    """
    Converts pages from a PDF document into high-res JPEG images and dual-stage thumbnails.
    Preserves PyMuPDF (fitz) lazy loading inside function body for fast startup.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pages_dir = Path(COMICS_DIR) / comic_id / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    try:
        import fitz
        document = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF document: {str(e)}")

    pages = []
    try:
        total_pages = len(document)
        if total_pages == 0:
            raise ValueError("PDF document contains no pages.")

        for page_index in range(total_pages):
            page_number = page_index + 1
            page = document[page_index]

            # 2x resolution for high-res reading and enhanced OCR recognition
            matrix = fitz.Matrix(2, 2)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            page_filename = f"page_{page_number:03d}.jpg"
            destination = pages_dir / page_filename

            pixmap.save(str(destination))
            pages.append(
                {
                    "page_number": page_number,
                    "filename": page_filename,
                    "image_path": str(destination),
                }
            )

        generate_all_thumbnails(comic_id, pages)

        duration = time.perf_counter() - start_time
        logger.info(
            "[PERF] Extraction completed: %s pages in %.2fs (with dual-resolution thumbnails)",
            len(pages),
            duration
        )

    finally:
        document.close()

    return pages


# ============================================================
# Standalone Images
# ============================================================

def extract_image(image_path: str, comic_id: str) -> list[dict]:
    """
    Processes a single standalone image as Page 1 of a comic with dual resolutions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    extension = Path(image_path).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    start_time = time.perf_counter()

    pages_dir = Path(COMICS_DIR) / comic_id / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_filename = f"page_001{extension}"
    destination = pages_dir / page_filename

    shutil.copy2(image_path, destination)

    pages = [
        {
            "page_number": 1,
            "filename": page_filename,
            "image_path": str(destination),
        }
    ]

    generate_all_thumbnails(comic_id, pages)

    duration = time.perf_counter() - start_time
    logger.info(
        "[PERF] Extraction completed: %s pages in %.2fs",
        len(pages),
        duration
    )

    return pages