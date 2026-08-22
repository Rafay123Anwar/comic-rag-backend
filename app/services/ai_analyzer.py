import base64
import io
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

from mistralai.client import Mistral
from mistralai.client.errors import MistralError

from app.core.config import (
    ENABLE_HYBRID_OCR,
    MAX_AI_RETRIES,
    MAX_AI_WORKERS,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)
from app.core.logging import logger

client = Mistral(
    api_key=MISTRAL_API_KEY
)



MODEL_NAME = MISTRAL_MODEL

# -----------------------------
# Module-Level Analysis Prompt
# -----------------------------

# COMIC_PAGE_ANALYSIS_PROMPT = """
# You are a strict Comic Page Extraction Engine for a multimodal RAG pipeline.

# Analyze ONLY the comic page image provided to you.

# Your job has TWO goals:

# 1. Transcribe every readable piece of text as accurately as possible.
# 2. Describe only the visual information that is directly visible.

# ==================================================
# CRITICAL GROUNDING RULES
# ==================================================

# 1. ONLY use information visible in this image.

# 2. NEVER use external knowledge.

# 3. NEVER identify characters by their known names.
#    For example, do NOT write:
#    "Victor Von Doom"
#    "Batman"
#    "Spider-Man"

#    Instead describe them visually:
#    "muscular male figure with long white hair"

# 4. NEVER infer:
#    - character identity
#    - relationships
#    - intentions
#    - backstory
#    - emotions that are not visually obvious
#    - events before or after the page
#    - information from other comic pages

# 5. Do not guess missing information.

# ==================================================
# OCR / TEXT TRANSCRIPTION
# ==================================================

# TEXT ACCURACY IS THE HIGHEST PRIORITY.

# Transcribe EVERY readable word that appears in the image.

# This includes:

# - Speech bubbles
# - Thought bubbles
# - Narration boxes
# - Captions
# - Sound effects
# - Titles
# - Chapter titles
# - Signs
# - Labels
# - Logos containing readable words
# - Credits
# - URLs
# - Issue numbers
# - Other visible written text

# IMPORTANT:

# 1. Transcribe the text EXACTLY as visible.

# 2. Do NOT summarize.

# 3. Do NOT paraphrase.

# 4. Do NOT improve grammar.

# 5. Do NOT correct spelling.

# 6. Do NOT replace an uncertain word with a logical word.

# 7. Preserve capitalization when clearly visible.

# 8. Preserve punctuation when clearly visible.

# 9. Preserve repeated punctuation.

# 10. Preserve contractions.

# 11. Preserve words exactly even if they appear grammatically incorrect.

# 12. If a word cannot be confidently read, use:

# [unclear]

# 13. NEVER guess an unreadable word using story context.

# 14. If only part of a word is readable:

# [unclear]

# 15. If there is no readable text:

# ""

# ==================================================
# TEXT READING ORDER
# ==================================================

# Read text in normal comic reading order.

# Generally:

# - Top to bottom
# - Left to right
# - Follow panel order
# - Within a panel, follow the natural bubble/caption order

# Do NOT reorder text based on what you think the story means.

# Each distinct text block should be preserved as a separate item.


# ==================================================
# SPEAKER / NARRATOR ATTRIBUTION
# ==================================================

# Preserve speaker or narrator attribution ONLY when it is explicitly
# supported by the visible comic page.

# For each dialogue or narration item, identify its source when
# possible.

# Examples:

# If the page explicitly shows:

# DOOM:
# "I remember my mother."

# Represent it as:

# "DOOM: I remember my mother."

# If a narration box is visibly associated with a named character:

# "DOCTOR DOOM (NARRATION): I remember everything."

# If the narrator is NOT explicitly identified:

# "NARRATOR: I remember everything."

# Do NOT guess who the narrator is.

# Do NOT assign narration to a character merely because that character
# appears visually on the page.

# Do NOT use outside comic knowledge to identify the narrator.

# If the text itself contains a character name, preserve that name
# exactly as written.

# If speaker attribution is uncertain, preserve the text without
# inventing an attribution.


# ==================================================
# VISUAL DESCRIPTION
# ==================================================

# Describe ONLY directly visible information.

# CHARACTERS / FIGURES:

# Describe:
# - number of visible figures
# - clothing
# - colors
# - hair
# - masks
# - body features
# - visible accessories
# - position

# Do NOT name them.

# ACTIONS / POSES:

# Describe only physical actions that are directly visible.

# For example:

# "Both arms are raised."

# "Figure is lying on the ground."

# "Person is holding a sword."

# Do NOT write:

# "He is angry."

# "She is attacking him because she hates him."

# unless the visual evidence makes that completely explicit.

# ENVIRONMENT:

# Describe the visible setting.

# OBJECTS:

# List important visible objects.

# BACKGROUND:

# Describe important background elements.

# ==================================================
# PAGE STRUCTURE
# ==================================================

# Identify the visible panel structure.

# If it is one full-page illustration:

# "Single full-page illustration"

# If there are multiple panels:

# "3 panels arranged vertically"

# Do not invent panels that are not visible.

# ==================================================
# OUTPUT
# ==================================================

# Return ONLY valid JSON.

# Do NOT use Markdown.

# Do NOT write ```json.

# Do NOT write any explanation before or after the JSON.

# Use EXACTLY this structure:

# {
#     "page_summary": "",
#     "panels_detected": 0,
#     "text": {
#         "full_text": "",
#         "dialogue_and_narration": [],
#         "sound_effects": [],
#         "signs_and_labels": []
#     },
#     "visual_description": {
#         "characters": [],
#         "actions": [],
#         "environment": "",
#         "objects": [],
#         "background": "",
#         "other_details": ""
#     }
# }

# ==================================================
# FINAL OCR CHECK
# ==================================================

# Before returning the JSON, perform one final visual check.

# Look again at the entire image and verify:

# - Did you capture every speech bubble?
# - Did you capture every narration box?
# - Did you capture every visible title?
# - Did you capture readable signs?
# - Did you capture sound effects?
# - Did you capture credits?
# - Did you capture small readable text?
# - Did you accidentally correct any spelling?
# - Did you accidentally invent any missing words?

# If uncertain, use [unclear].

# Never guess.
# """

COMIC_PAGE_ANALYSIS_PROMPT = """
You are a strict Comic Page Extraction Engine for a multimodal RAG pipeline.

This engine must work correctly on ANY type of sequential art page, including
but not limited to: Western-style superhero comics, manga, manhwa, manhua,
webtoons, graphic novels, indie comics, romance comics, horror comics,
slice-of-life comics, and any other comic/illustrated storytelling format.
Do not assume any particular genre, publisher, or art style.

Analyze ONLY the comic page image provided to you.

Your job has TWO goals:

1. Transcribe every readable piece of text as accurately as possible.
2. Describe only the visual information that is directly visible.

==================================================
CRITICAL GROUNDING RULES
==================================================

1. ONLY use information visible in this image.

2. NEVER use external knowledge, including knowledge of well-known
   franchises, characters, authors, or publishers this page might
   resemble.

3. NEVER identify characters by any known/famous name, regardless of
   genre or franchise. For example, do NOT write:
   "Victor Von Doom"
   "Batman"
   "Spider-Man"
   "Naruto"
   "Goku"

   Instead describe them visually:
   "muscular male figure with long white hair"
   "young woman with short black hair and school uniform"

4. NEVER infer:
   - character identity
   - relationships
   - intentions
   - backstory
   - emotions that are not visually obvious
   - events before or after the page
   - information from other comic pages

5. Do not guess missing information.

==================================================
OCR / TEXT TRANSCRIPTION
==================================================

TEXT ACCURACY IS THE HIGHEST PRIORITY.

Transcribe EVERY readable word that appears in the image, in any
language or script present on the page (English, Japanese, Korean,
Chinese, Urdu, or any other language/script).

This includes:

- Speech bubbles
- Thought bubbles
- Narration boxes
- Captions
- Sound effects
- Titles
- Chapter titles
- Signs
- Labels
- Logos containing readable words
- Credits
- URLs
- Issue numbers
- Other visible written text

IMPORTANT:

1. Transcribe the text EXACTLY as visible.

2. Do NOT summarize.

3. Do NOT paraphrase.

4. Do NOT improve grammar.

5. Do NOT correct spelling.

6. Do NOT replace an uncertain word with a logical word.

7. Preserve capitalization when clearly visible.

8. Preserve punctuation when clearly visible.

9. Preserve repeated punctuation.

10. Preserve contractions.

11. Preserve words exactly even if they appear grammatically incorrect.

12. If a word cannot be confidently read, use:

[unclear]

13. NEVER guess an unreadable word using story context.

14. If only part of a word is readable:

[unclear]

15. If there is no readable text:

""

==================================================
TEXT READING ORDER
==================================================

Read text in the natural reading order for the page's own layout.

- For left-to-right pages (most Western comics, webtoons), follow:
  top to bottom, left to right, following panel order.
- For right-to-left pages (traditional manga layout), follow the
  page's own visible panel flow (typically top-right to bottom-left)
  rather than forcing a left-to-right order.
- Within a panel, follow the natural bubble/caption order as it
  appears visually.

Do NOT reorder text based on what you think the story means.

Each distinct text block should be preserved as a separate item.


==================================================
SPEAKER / NARRATOR ATTRIBUTION
==================================================

Preserve speaker or narrator attribution ONLY when it is explicitly
supported by the visible comic page.

For each dialogue or narration item, identify its source when
possible.

Examples:

If the page explicitly shows a name label before dialogue:

NAME:
"I remember my mother."

Represent it as:

"NAME: I remember my mother."

If a narration box is visibly associated with a named character:

"NAME (NARRATION): I remember everything."

If the narrator is NOT explicitly identified:

"NARRATOR: I remember everything."

Do NOT guess who the narrator is.

Do NOT assign narration to a character merely because that character
appears visually on the page.

Do NOT use outside comic knowledge to identify the narrator.

If the text itself contains a character name, preserve that name
exactly as written.

If speaker attribution is uncertain, preserve the text without
inventing an attribution.


==================================================
VISUAL DESCRIPTION
==================================================

Describe ONLY directly visible information.

CHARACTERS / FIGURES:

Describe:
- number of visible figures
- clothing
- colors
- hair
- masks
- body features
- visible accessories
- position

Do NOT name them.

ACTIONS / POSES:

Describe only physical actions that are directly visible.

For example:

"Both arms are raised."

"Figure is lying on the ground."

"Person is holding a sword."

Do NOT write:

"He is angry."

"She is attacking him because she hates him."

unless the visual evidence makes that completely explicit.

ENVIRONMENT:

Describe the visible setting.

OBJECTS:

List important visible objects.

BACKGROUND:

Describe important background elements.

==================================================
PAGE STRUCTURE
==================================================

Identify the visible panel structure.

If it is one full-page illustration:

"Single full-page illustration"

If there are multiple panels:

"3 panels arranged vertically"

If the panel layout is irregular or overlapping (common in manga/webtoons):

"Irregular panel layout with overlapping panels"

Do not invent panels that are not visible.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Do NOT use Markdown.

Do NOT write ```json.

Do NOT write any explanation before or after the JSON.

Use EXACTLY this structure:

{
    "page_summary": "",
    "panels_detected": 0,
    "text": {
        "full_text": "",
        "dialogue_and_narration": [],
        "sound_effects": [],
        "signs_and_labels": []
    },
    "visual_description": {
        "characters": [],
        "actions": [],
        "environment": "",
        "objects": [],
        "background": "",
        "other_details": ""
    }
}

==================================================
FINAL OCR CHECK
==================================================

Before returning the JSON, perform one final visual check.

Look again at the entire image and verify:

- Did you capture every speech bubble?
- Did you capture every narration box?
- Did you capture every visible title?
- Did you capture readable signs?
- Did you capture sound effects?
- Did you capture credits?
- Did you capture small readable text?
- Did you capture text in any non-English script present on the page?
- Did you accidentally correct any spelling?
- Did you accidentally invent any missing words?

If uncertain, use [unclear].

Never guess.
Include both white dialogue bubbles and black narration caption boxes. Do not ignore sound effects (SFX) or stylized text.
"""


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Determines if an exception represents an HTTP 429 Rate Limit error.
    Checks the Mistral SDK MistralError status_code, underlying response status_code,
    or common rate limit error messages.
    """
    if isinstance(exc, MistralError) and getattr(exc, "status_code", None) == 429:
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
    if raw_resp is not None and getattr(raw_resp, "status_code", None) == 429:
        return True
    err_str = str(exc).lower()
    if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
        return True
    return False


def get_retry_after(exc: Exception) -> float | None:
    """
    Extracts Retry-After header or provider delay duration from exception details.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
        if raw_resp is not None:
            headers = getattr(raw_resp, "headers", None)
    if headers:
        retry_val = headers.get("retry-after") or headers.get("Retry-After")
        if retry_val:
            try:
                val = float(retry_val)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass

    # Extract delay from error message if available (e.g. 'try again in 12.4s')
    err_str = str(exc)
    match = re.search(r"(?:try again in|retry after|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", err_str, re.I)
    if match:
        try:
            val = float(match.group(1))
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    return None


# -----------------------------
# Rate Limit Coordinator (M-25)
# -----------------------------

class RateLimitCoordinator:
    """
    Centralized rate-limit and recovery coordination across worker threads.
    - Implements slot reservation to prevent synchronized retry stampedes after 429s.
    - Staggers recovery dispatches so workers resume gradually rather than in lockstep bursts.
    - Maintains normal full concurrency of workers when API is healthy.
    - Validates clearance immediately before dispatch to guarantee no collisions during extended cooldowns.
    """

    def __init__(
        self,
        target_workers: int = MAX_AI_WORKERS,
        healthy_interval: float = 0.20,
        recovery_interval: float = 1.80
    ):
        self.target_workers = target_workers
        self.healthy_interval = healthy_interval
        self.recovery_interval = recovery_interval

        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._recovery_until = 0.0
        self._next_available_slot = 0.0
        self._last_dispatch_time = 0.0
        self._active_requests = 0
        self._rate_limit_events = 0
        self._total_cooldown_wait = 0.0

    def before_request(self, page_number: int, is_retry: bool = False, attempt: int = 0) -> float:
        """
        Coordinates and reserves an authorized dispatch slot for a worker.
        Handles both initial slot reservation and final pre-dispatch clearance,
        ensuring workers never collide or fire during active global cooldowns.
        Returns the total wait time in seconds.
        """
        wait_start = time.perf_counter()

        while True:
            # Phase 1: Slot Reservation
            with self._lock:
                now = time.monotonic()
                earliest = max(now, self._cooldown_until)

                is_recovering = (now < self._recovery_until) or is_retry
                if is_recovering:
                    # During rate-limit recovery: stagger dispatches smoothly
                    interval = self.recovery_interval + random.uniform(0.1, 0.4)
                else:
                    # Healthy concurrent throughput: gentle pacing
                    interval = self.healthy_interval

                # Allocate slot at or after next available slot and cooldown
                slot = max(earliest, self._next_available_slot)
                self._next_available_slot = slot + interval

            # Sleep outside the lock until our reserved slot arrives
            delay = slot - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            # Phase 2: Pre-Dispatch Verification
            with self._lock:
                now = time.monotonic()
                # Check if a new 429 occurred while we were sleeping
                if now < self._cooldown_until:
                    # Cooldown was extended by another event while we slept; loop and re-reserve
                    continue

                # Clearance granted: record dispatch and increment active count
                self._last_dispatch_time = now
                self._active_requests += 1
                active = self._active_requests
                break

        total_waited = time.perf_counter() - wait_start
        if total_waited > 0.05:
            with self._lock:
                self._total_cooldown_wait += total_waited

        logger.info(
            "[PERF] AI concurrency | active=%d | target=%d",
            active,
            self.target_workers
        )

        return total_waited

    def on_success(self):
        """Called when an API request completes successfully."""
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1

    def on_other_error(self):
        """Called when an API request fails with a non-rate-limit error."""
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1

    def on_rate_limit(
        self,
        page_number: int,
        attempt: int,
        exc: Exception,
        max_retries: int = 5
    ) -> float:
        """
        Called when a worker receives an HTTP 429 rate-limit error.
        Activates global cooldown and recovery mode across all workers.
        Returns the computed cooldown duration in seconds.
        """
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1

            self._rate_limit_events += 1
            now = time.monotonic()

            retry_after = get_retry_after(exc)
            if retry_after is not None and retry_after > 0:
                base_wait = retry_after
            else:
                # Exponential backoff based on attempt with generous base
                base_wait = min(3 * (2 ** attempt), 40)

            jitter = random.uniform(1.0, 3.0)
            cooldown = base_wait + jitter

            # Extend global cooldown and recovery window
            self._cooldown_until = max(self._cooldown_until, now + base_wait)
            self._recovery_until = max(
                self._recovery_until,
                self._cooldown_until + (self.recovery_interval * self.target_workers)
            )

            # Push next available slot beyond the new cooldown
            self._next_available_slot = max(self._next_available_slot, self._cooldown_until)

        logger.warning(
            "[PERF] Rate limit detected | page=%s | attempt=%d | cooldown=%.2fs",
            page_number,
            attempt + 1,
            cooldown
        )

        return cooldown

    @property
    def rate_limit_events(self) -> int:
        with self._lock:
            return self._rate_limit_events

    @property
    def total_cooldown_wait(self) -> float:
        with self._lock:
            return self._total_cooldown_wait


def _parse_ai_response_json(raw_result: str) -> dict:
    """
    Robust JSON parser for AI page analysis responses.
    Handles strict decoding, markdown code fences, embedded JSON objects, and unescaped newlines.
    """
    # 1. Direct parse with strict=False (allows unescaped control chars/newlines in strings)
    try:
        return json.loads(raw_result, strict=False)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip(), strict=False)
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Extract outermost JSON object { ... }
    brace_match = re.search(r"(\{.*\})", raw_result, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip(), strict=False)
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Fallback cleanup: remove leading/trailing ```
    cleaned = raw_result.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip(), strict=False)

def _encode_and_scale_image(image_path: str, max_dimension: int = 1600, quality: int = 85) -> str:
    """
    Loads an image, resizes it so its longest dimension does not exceed max_dimension
    (using LANCZOS resampling to preserve sharp text), and encodes to base64 JPEG (quality=85).
    """
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        longest = max(orig_w, orig_h)

        if longest > max_dimension:
            scale = max_dimension / float(longest)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


_easyocr_reader = None
_easyocr_lock = threading.Lock()


def get_easyocr_reader():
    """Lazy-loads EasyOCR Reader instance thread-safely."""
    global _easyocr_reader
    if _easyocr_reader is None:
        with _easyocr_lock:
            if _easyocr_reader is None:
                try:
                    import easyocr
                    _easyocr_reader = easyocr.Reader(["en"], gpu=False)
                except Exception as e:
                    logger.warning("EasyOCR initialization error: %s", e)
                    _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


def _extract_draft_ocr(image_path: str, max_dim: int = 1000) -> str:
    """
    Extracts fast preliminary draft OCR text using EasyOCR to assist Vision LLM.
    """
    reader = get_easyocr_reader()
    if not reader:
        return ""
    try:
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            buffer.seek(0)
            results = reader.readtext(buffer.getvalue(), detail=0)
            return "\n".join(results).strip()
    except Exception as e:
        logger.debug("Draft OCR error on %s: %s", image_path, e)
        return ""


# -----------------------------
# Single Page Analysis
# -----------------------------

def analyze_page(image_path: str, draft_ocr: str = "") -> dict:

    image_base64 = _encode_and_scale_image(
        image_path,
        max_dimension=1600,
        quality=85
    )

    if ENABLE_HYBRID_OCR and not draft_ocr:
        draft_ocr = _extract_draft_ocr(image_path)

    if draft_ocr:
        prompt_text = (
            f"{COMIC_PAGE_ANALYSIS_PROMPT}\n\n"
            f"==================================================\n"
            f"DRAFT OCR TRANSCRIPTION (From local preprocessing):\n"
            f"Verify, correct, complete, and structure this draft against the actual image:\n"
            f'"""\n{draft_ocr}\n"""\n'
        )
    else:
        prompt_text = COMIC_PAGE_ANALYSIS_PROMPT

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,
    )

    raw_result = response.choices[0].message.content

    # Convert AI response to JSON
    return _parse_ai_response_json(raw_result)

# -----------------------------
# Build Comic-Level Full Text
# -----------------------------

def build_comic_full_text(results: list) -> str:
    """
    Page-level OCR text ko page order mein combine karta hai.

    Important:
    - AI ko dobara entire comic dene ki zaroorat nahi.
    - Existing page OCR ko hi combine kiya jata hai.
    - Page number preserve hota hai.
    """

    full_text_parts = []

    ordered_results = sorted(
        results,
        key=lambda x: x["page_number"]
    )

    for page in ordered_results:

        if page.get("status") != "success":
            continue

        analysis = page.get("analysis") or {}

        text_data = analysis.get(
            "text",
            {}
        )

        page_text = text_data.get(
            "full_text",
            ""
        )

        if not isinstance(page_text, str):
            continue

        page_text = page_text.strip()

        if not page_text:
            continue

        full_text_parts.append(
            f"PAGE {page['page_number']}\n"
            f"{page_text}"
        )

    return "\n\n".join(
        full_text_parts
    )


def analyze_pages(
    pages: list,
    max_workers: int = MAX_AI_WORKERS,
    max_retries: int = MAX_AI_RETRIES,
    on_page_complete: Optional[Callable[[dict, int, int], None]] = None
) -> list:

    coordinator = RateLimitCoordinator(target_workers=max_workers)

    def process_page(page):

        page_number = page["page_number"]
        image_path = page["image_path"]
        page_start = time.perf_counter()
        attempts = 0
        retry_wait_total = 0.0

        if not Path(image_path).exists():
            page_duration = time.perf_counter() - page_start
            logger.warning("[AI] Page image %s not found on local disk.", image_path)
            return {
                "page_number": page_number,
                "filename": page.get("filename", Path(image_path).name),
                "image_path": image_path,
                "analysis": None,
                "metadata": {
                    "page_number": page_number
                },
                "status": "error",
                "error": f"Image file not found: {image_path}",
                "perf": {
                    "duration": page_duration,
                    "attempts": 1,
                    "retry_wait": 0.0
                }
            }

        for attempt in range(
            max_retries + 1
        ):
            attempts += 1

            # Request slot authorization from the coordinator
            wait_time = coordinator.before_request(
                page_number=page_number,
                is_retry=(attempt > 0),
                attempt=attempt
            )
            retry_wait_total += wait_time

            try:

                result = analyze_page(
                    image_path
                )

                coordinator.on_success()

                page_duration = time.perf_counter() - page_start

                logger.info(
                    "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs",
                    page_number,
                    page_duration,
                    attempts,
                    retry_wait_total
                )

                return {
                    "page_number": page_number,
                    "filename": page.get("filename", Path(image_path).name),
                    "image_path": image_path,
                    "analysis": result,
                    "metadata": {
                        "page_number": page_number,
                        "has_text": bool(
                            (
                                result.get("text", {})
                                .get("full_text", "")
                                if isinstance(
                                    result.get("text", {}),
                                    dict
                                )
                                else ""
                            ).strip()
                        )
                    },
                    "status": "success",
                    "perf": {
                        "duration": page_duration,
                        "attempts": attempts,
                        "retry_wait": retry_wait_total
                    }
                }

            except Exception as e:

                error_message = str(e)

                if is_rate_limit_error(e):

                    if attempt >= max_retries:
                        coordinator.on_other_error()
                        page_duration = time.perf_counter() - page_start
                        logger.error(
                            "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED RATE LIMIT)",
                            page_number,
                            page_duration,
                            attempts,
                            retry_wait_total
                        )

                        return {
                            "page_number": page_number,
                            "filename": page.get("filename", Path(image_path).name),
                            "image_path": image_path,
                            "analysis": None,
                            "metadata": {
                                "page_number": page_number
                            },
                            "status": "error",
                            "error": error_message,
                            "perf": {
                                "duration": page_duration,
                                "attempts": attempts,
                                "retry_wait": retry_wait_total
                            }
                        }

                    # Register rate limit with coordinator and sleep with exponential backoff + jitter
                    cooldown = coordinator.on_rate_limit(
                        page_number=page_number,
                        attempt=attempt,
                        exc=e,
                        max_retries=max_retries
                    )
                    time.sleep(cooldown)
                    retry_wait_total += cooldown

                else:
                    # Non-rate-limit error (e.g. transient network or JSON format error)
                    coordinator.on_other_error()

                    if attempt < max_retries:
                        logger.warning(
                            "Page %s encountered error (%s). Retrying in 1.5s (attempt %s/%s)...",
                            page_number,
                            error_message,
                            attempt + 1,
                            max_retries
                        )
                        retry_start = time.perf_counter()
                        time.sleep(1.5 + random.uniform(0.1, 0.5))
                        retry_wait_total += time.perf_counter() - retry_start
                    else:
                        page_duration = time.perf_counter() - page_start
                        logger.error(
                            "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED ERROR)",
                            page_number,
                            page_duration,
                            attempts,
                            retry_wait_total
                        )

                        return {
                            "page_number": page_number,
                            "filename": page.get("filename", Path(image_path).name),
                            "image_path": image_path,
                            "analysis": None,
                            "metadata": {
                                "page_number": page_number
                            },
                            "status": "error",
                            "error": error_message,
                            "perf": {
                                "duration": page_duration,
                                "attempts": attempts,
                                "retry_wait": retry_wait_total
                            }
                        }

    results = []
    total_pages = len(pages)
    ai_start = time.perf_counter()

    # -----------------------------
    # Parallel Processing
    # -----------------------------

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                process_page,
                page
            ): page["page_number"]

            for page in pages
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            result = future.result()

            results.append(
                result
            )

            completed += 1

            if result["status"] == "success":
                logger.info(
                    "[%s/%s] Page %s analyzed successfully",
                    completed,
                    total_pages,
                    result["page_number"]
                )
            else:
                logger.error(
                    "[%s/%s] Page %s analysis failed: %s",
                    completed,
                    total_pages,
                    result["page_number"],
                    result.get("error", "Unknown error")
                )

            if on_page_complete is not None:
                try:
                    on_page_complete(result, completed, total_pages)
                except Exception as cb_err:
                    logger.warning("on_page_complete callback error: %s", str(cb_err))

    total_ai_duration = time.perf_counter() - ai_start

    # -----------------------------
    # Performance Aggregate Stats
    # -----------------------------
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    perf_list = [r.get("perf", {}) for r in results]
    pages_with_retries = sum(1 for p in perf_list if p.get("attempts", 1) > 1)
    total_retries = sum(max(0, p.get("attempts", 1) - 1) for p in perf_list)
    total_retry_wait = sum(p.get("retry_wait", 0.0) for p in perf_list)
    durations = [p.get("duration", 0.0) for p in perf_list]
    avg_page_duration = (sum(durations) / len(durations)) if durations else 0.0
    fastest_page = min(durations) if durations else 0.0
    slowest_page = max(durations) if durations else 0.0

    no_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) == 1]
    avg_no_retry = (sum(no_retry_durations) / len(no_retry_durations)) if no_retry_durations else 0.0
    with_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) > 1]
    avg_with_retry = (sum(with_retry_durations) / len(with_retry_durations)) if with_retry_durations else 0.0

    logger.info(
        "\n"
        "[PERF] ==================================================\n"
        "[PERF] AI ANALYSIS PERFORMANCE\n"
        "[PERF] Pages: %s\n"
        "[PERF] Workers: %s\n"
        "[PERF] Total duration: %.2fs\n"
        "[PERF] Successful: %s\n"
        "[PERF] Failed: %s\n"
        "[PERF] Pages with retries: %s\n"
        "[PERF] Total retries: %s\n"
        "[PERF] Total retry wait: %.2fs\n"
        "[PERF] Rate-limit events: %s\n"
        "[PERF] Total provider cooldown time: %.2fs\n"
        "[PERF] Average page duration: %.2fs\n"
        "[PERF]   - No-retry pages avg: %.2fs (%s pages)\n"
        "[PERF]   - With-retry pages avg: %.2fs (%s pages)\n"
        "[PERF] Fastest page: %.2fs\n"
        "[PERF] Slowest page: %.2fs\n"
        "[PERF] ==================================================",
        total_pages,
        max_workers,
        total_ai_duration,
        successful,
        failed,
        pages_with_retries,
        total_retries,
        total_retry_wait,
        coordinator.rate_limit_events,
        coordinator.total_cooldown_wait,
        avg_page_duration,
        avg_no_retry,
        len(no_retry_durations),
        avg_with_retry,
        len(with_retry_durations),
        fastest_page,
        slowest_page
    )

    # -----------------------------
    # Restore Page Order
    # -----------------------------

    results.sort(
        key=lambda x: x["page_number"]
    )

    return results