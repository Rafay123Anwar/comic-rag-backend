# from fastapi import responses
# import base64
# import io
# import json
# import random
# import re
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Callable, Optional
# from PIL import Image

# from mistralai.client import Mistral
# from mistralai.client.errors import MistralError

# from app.core.config import (
#     ENABLE_HYBRID_OCR,
#     MAX_AI_RETRIES,
#     MAX_AI_WORKERS,
#     MISTRAL_API_KEY,
#     MISTRAL_MODEL,
# )
# from app.core.logging import logger

# client = Mistral(
#     api_key=MISTRAL_API_KEY
# )



# MODEL_NAME = MISTRAL_MODEL

# # -----------------------------
# # Module-Level Analysis Prompt
# # -----------------------------

# # COMIC_PAGE_ANALYSIS_PROMPT = """
# # You are a strict Comic Page Extraction Engine for a multimodal RAG pipeline.

# # Analyze ONLY the comic page image provided to you.

# # Your job has TWO goals:

# # 1. Transcribe every readable piece of text as accurately as possible.
# # 2. Describe only the visual information that is directly visible.

# # ==================================================
# # CRITICAL GROUNDING RULES
# # ==================================================

# # 1. ONLY use information visible in this image.

# # 2. NEVER use external knowledge.

# # 3. NEVER identify characters by their known names.
# #    For example, do NOT write:
# #    "Victor Von Doom"
# #    "Batman"
# #    "Spider-Man"

# #    Instead describe them visually:
# #    "muscular male figure with long white hair"

# # 4. NEVER infer:
# #    - character identity
# #    - relationships
# #    - intentions
# #    - backstory
# #    - emotions that are not visually obvious
# #    - events before or after the page
# #    - information from other comic pages

# # 5. Do not guess missing information.

# # ==================================================
# # OCR / TEXT TRANSCRIPTION
# # ==================================================

# # TEXT ACCURACY IS THE HIGHEST PRIORITY.

# # Transcribe EVERY readable word that appears in the image.

# # This includes:

# # - Speech bubbles
# # - Thought bubbles
# # - Narration boxes
# # - Captions
# # - Sound effects
# # - Titles
# # - Chapter titles
# # - Signs
# # - Labels
# # - Logos containing readable words
# # - Credits
# # - URLs
# # - Issue numbers
# # - Other visible written text

# # IMPORTANT:

# # 1. Transcribe the text EXACTLY as visible.

# # 2. Do NOT summarize.

# # 3. Do NOT paraphrase.

# # 4. Do NOT improve grammar.

# # 5. Do NOT correct spelling.

# # 6. Do NOT replace an uncertain word with a logical word.

# # 7. Preserve capitalization when clearly visible.

# # 8. Preserve punctuation when clearly visible.

# # 9. Preserve repeated punctuation.

# # 10. Preserve contractions.

# # 11. Preserve words exactly even if they appear grammatically incorrect.

# # 12. If a word cannot be confidently read, use:

# # [unclear]

# # 13. NEVER guess an unreadable word using story context.

# # 14. If only part of a word is readable:

# # [unclear]

# # 15. If there is no readable text:

# # ""

# # ==================================================
# # TEXT READING ORDER
# # ==================================================

# # Read text in normal comic reading order.

# # Generally:

# # - Top to bottom
# # - Left to right
# # - Follow panel order
# # - Within a panel, follow the natural bubble/caption order

# # Do NOT reorder text based on what you think the story means.

# # Each distinct text block should be preserved as a separate item.


# # ==================================================
# # SPEAKER / NARRATOR ATTRIBUTION
# # ==================================================

# # Preserve speaker or narrator attribution ONLY when it is explicitly
# # supported by the visible comic page.

# # For each dialogue or narration item, identify its source when
# # possible.

# # Examples:

# # If the page explicitly shows:

# # DOOM:
# # "I remember my mother."

# # Represent it as:

# # "DOOM: I remember my mother."

# # If a narration box is visibly associated with a named character:

# # "DOCTOR DOOM (NARRATION): I remember everything."

# # If the narrator is NOT explicitly identified:

# # "NARRATOR: I remember everything."

# # Do NOT guess who the narrator is.

# # Do NOT assign narration to a character merely because that character
# # appears visually on the page.

# # Do NOT use outside comic knowledge to identify the narrator.

# # If the text itself contains a character name, preserve that name
# # exactly as written.

# # If speaker attribution is uncertain, preserve the text without
# # inventing an attribution.


# # ==================================================
# # VISUAL DESCRIPTION
# # ==================================================

# # Describe ONLY directly visible information.

# # CHARACTERS / FIGURES:

# # Describe:
# # - number of visible figures
# # - clothing
# # - colors
# # - hair
# # - masks
# # - body features
# # - visible accessories
# # - position

# # Do NOT name them.

# # ACTIONS / POSES:

# # Describe only physical actions that are directly visible.

# # For example:

# # "Both arms are raised."

# # "Figure is lying on the ground."

# # "Person is holding a sword."

# # Do NOT write:

# # "He is angry."

# # "She is attacking him because she hates him."

# # unless the visual evidence makes that completely explicit.

# # ENVIRONMENT:

# # Describe the visible setting.

# # OBJECTS:

# # List important visible objects.

# # BACKGROUND:

# # Describe important background elements.

# # ==================================================
# # PAGE STRUCTURE
# # ==================================================

# # Identify the visible panel structure.

# # If it is one full-page illustration:

# # "Single full-page illustration"

# # If there are multiple panels:

# # "3 panels arranged vertically"

# # Do not invent panels that are not visible.

# # ==================================================
# # OUTPUT
# # ==================================================

# # Return ONLY valid JSON.

# # Do NOT use Markdown.

# # Do NOT write ```json.

# # Do NOT write any explanation before or after the JSON.

# # Use EXACTLY this structure:

# # {
# #     "page_summary": "",
# #     "panels_detected": 0,
# #     "text": {
# #         "full_text": "",
# #         "dialogue_and_narration": [],
# #         "sound_effects": [],
# #         "signs_and_labels": []
# #     },
# #     "visual_description": {
# #         "characters": [],
# #         "actions": [],
# #         "environment": "",
# #         "objects": [],
# #         "background": "",
# #         "other_details": ""
# #     }
# # }

# # ==================================================
# # FINAL OCR CHECK
# # ==================================================

# # Before returning the JSON, perform one final visual check.

# # Look again at the entire image and verify:

# # - Did you capture every speech bubble?
# # - Did you capture every narration box?
# # - Did you capture every visible title?
# # - Did you capture readable signs?
# # - Did you capture sound effects?
# # - Did you capture credits?
# # - Did you capture small readable text?
# # - Did you accidentally correct any spelling?
# # - Did you accidentally invent any missing words?

# # If uncertain, use [unclear].

# # Never guess.
# # """

# # COMIC_PAGE_ANALYSIS_PROMPT = """
# # You are a universal Comic Page Analyzer and Extraction Engine for a multimodal RAG pipeline.
# # Your task is to accurately extract all text and describe the visual contents of the provided page image.

# # This engine must process ANY genre, style, and sequential art format:
# # - Western Comics & Graphic Novels (Superhero, Sci-Fi, Dark Fantasy, Crime/Noir, Indie)
# # - Manga, Manhwa, & Webtoons (Shonen, Seinen, Shojo, Webcomics, vertical or horizontal layouts)
# # - Multi-panel grids, full-page splash art, covers, chapter title pages, or silent action scenes.

# # ==================================================
# # 1. COMPLETE TEXT TRANSCRIPTION & OCR (HIGHEST PRIORITY)
# # ==================================================
# # - Scan EVERY panel across the entire page from top to bottom (following natural reading order: left-to-right or right-to-left as drawn).
# # - Transcribe ALL text: speech bubbles, thought clouds, narration/monologue boxes, captions, signs, titles, and sound effects (SFX).
# # - Do NOT skip any dialogue or text boxes. Capture all text present on the page faithfully.
# # - Preserve exact wording, casing, and punctuation as visible.
# # - If a page has no text, return empty string "" for full_text and empty list [] for dialogue_and_narration.

# # ==================================================
# # 2. PRECISE & GROUNDED VISUAL EXTRACTION
# # ==================================================
# # - Describe ONLY what is directly visible in the image. Never invent characters, items, or events not present.
# # - In "characters": Describe the actual visible figures (appearance, clothing, expressions, gestures, and identities if established by visible dialogue/names).
# # - In "actions": List the physical actions and interactions taking place in the panels (e.g. movement, fighting, talking, emoting, working).
# # - In "environment": Describe the setting/location (e.g. city street, space station, forest, room, fantasy realm, battlefield).
# # - In "objects": List notable visible items, weapons, tools, vehicles, or artifacts.
# # - In "background": Describe background elements, lighting, weather, and visual atmosphere.

# # ==================================================
# # 3. OUTPUT FORMAT
# # ==================================================
# # Return ONLY a valid JSON object matching this exact schema:
# # {
# #     "page_summary": "Comprehensive narrative summary of the page's events, characters, visual action, and story progression.",
# #     "panels_detected": 1,
# #     "text": {
# #         "full_text": "Exact concatenated transcription of all readable text on this page in natural reading order.",
# #         "dialogue_and_narration": [
# #             "Exact transcribed dialogue or narration block 1",
# #             "Exact transcribed dialogue or narration block 2"
# #         ],
# #         "sound_effects": [],
# #         "signs_and_labels": []
# #     },
# #     "visual_description": {
# #         "characters": [
# #             "Description of visible character/figure 1",
# #             "Description of visible character/figure 2"
# #         ],
# #         "actions": [
# #             "Description of visible action or interaction 1",
# #             "Description of visible action or interaction 2"
# #         ],
# #         "environment": "Description of the physical setting or location.",
# #         "objects": [
# #             "Notable visible object or item 1",
# #             "Notable visible object or item 2"
# #         ],
# #         "background": "Description of background details, lighting, and mood.",
# #         "other_details": ""
# #     }
# # }
# # """

# COMIC_PAGE_ANALYSIS_PROMPT = """
# NOTE: The image(s) provided may be sequential top-to-bottom slices of a SINGLE tall comic page
# (with slight overlap between slices). If multiple images are given, treat them together as ONE
# continuous page — do not treat slices as separate pages or invent extra panels just because the
# page was sliced.

# You are a universal Comic Page Analyzer and Extraction Engine for a multimodal RAG pipeline.
# Your task is to accurately extract all text and describe the visual contents of the provided page image.

# # ==================================================
# # 1. COMPLETE TEXT TRANSCRIPTION & OCR (HIGHEST PRIORITY)
# # ==================================================
# # - Scan EVERY panel across the entire page from top to bottom (following natural reading order: left-to-right or right-to-left as drawn).
# # - Transcribe ALL text: speech bubbles, thought clouds, narration/monologue boxes, captions, signs, titles, and sound effects (SFX).
# # - Do NOT skip any dialogue or text boxes. Capture all text present on the page faithfully.
# # - Preserve exact wording, casing, and punctuation as visible.
# # - If a page has no text, return empty string "" for full_text and empty list [] for dialogue_and_narration.


# ==================================================
# 2. PRECISE & GROUNDED VISUAL EXTRACTION (PANEL-SCOPED)
# ==================================================
# - Identify each distinct panel on the page separately (a full-page splash counts as ONE panel).
# - For EACH panel, describe ONLY the figures/objects/actions physically visible WITHIN that panel's borders.
# - NEVER borrow, merge, or transfer a detail (emotion, pose, clothing, identity) from one panel into your
#   description of a different panel, even if the figures look similar.
# - If the SAME figure (by consistent visual traits: hair, clothing, size) appears in multiple panels,
#   describe it separately in each panel's context — do NOT assume it is a different, new character just
#   because the panel changed, AND do NOT assume it is the same character unless traits genuinely match.
# - Do NOT invent a figure, pose, or emotional expression (e.g. "distressed", "looking up") unless it is
#   unambiguously visible in that specific panel. If uncertain, omit the detail rather than guessing.

# ==================================================
# 3. OUTPUT FORMAT
# ==================================================
# Return ONLY a valid JSON object matching this exact schema:
# {
#     "page_summary": "...",
#     "panels_detected": 0,
#     "panels": [
#         {
#             "panel_index": 1,
#             "dialogue_and_narration": ["..."],
#             "characters": ["Description of visible figure 1 in THIS panel only", "..."],
#             "actions": ["..."],
#             "environment": "...",
#             "objects": ["..."]
#         }
#     ],
#     "text": {
#         "full_text": "...",
#         "dialogue_and_narration": [...],
#         "sound_effects": [],
#         "signs_and_labels": []
#     },
#     "visual_description": {
#         "characters": [...],
#         "actions": [...],
#         "environment": "...",
#         "objects": [...],
#         "background": "...",
#         "other_details": ""
#     }
# }
# """


# def is_rate_limit_error(exc: Exception) -> bool:
#     """
#     Determines if an exception represents an HTTP 429 Rate Limit error.
#     Checks the Mistral SDK MistralError status_code, underlying response status_code,
#     or common rate limit error messages.
#     """
#     if isinstance(exc, MistralError) and getattr(exc, "status_code", None) == 429:
#         return True
#     if getattr(exc, "status_code", None) == 429:
#         return True
#     raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#     if raw_resp is not None and getattr(raw_resp, "status_code", None) == 429:
#         return True
#     err_str = str(exc).lower()
#     if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
#         return True
#     return False


# def get_retry_after(exc: Exception) -> float | None:
#     """
#     Extracts Retry-After header or provider delay duration from exception details.
#     """
#     headers = getattr(exc, "headers", None)
#     if headers is None:
#         raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#         if raw_resp is not None:
#             headers = getattr(raw_resp, "headers", None)
#     if headers:
#         retry_val = headers.get("retry-after") or headers.get("Retry-After")
#         if retry_val:
#             try:
#                 val = float(retry_val)
#                 if val > 0:
#                     return val
#             except (ValueError, TypeError):
#                 pass

#     # Extract delay from error message if available (e.g. 'try again in 12.4s')
#     err_str = str(exc)
#     match = re.search(r"(?:try again in|retry after|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", err_str, re.I)
#     if match:
#         try:
#             val = float(match.group(1))
#             if val > 0:
#                 return val
#         except (ValueError, TypeError):
#             pass

#     return None


# # -----------------------------
# # Rate Limit Coordinator (M-25)
# # -----------------------------

# class RateLimitCoordinator:
#     """
#     Centralized rate-limit and recovery coordination across worker threads.
#     - Implements slot reservation to prevent synchronized retry stampedes after 429s.
#     - Staggers recovery dispatches so workers resume gradually rather than in lockstep bursts.
#     - Maintains normal full concurrency of workers when API is healthy.
#     - Validates clearance immediately before dispatch to guarantee no collisions during extended cooldowns.
#     """

#     def __init__(
#         self,
#         target_workers: int = MAX_AI_WORKERS,
#         healthy_interval: float = 0.20,
#         recovery_interval: float = 1.80
#     ):
#         self.target_workers = target_workers
#         self.healthy_interval = healthy_interval
#         self.recovery_interval = recovery_interval

#         self._lock = threading.Lock()
#         self._cooldown_until = 0.0
#         self._recovery_until = 0.0
#         self._next_available_slot = 0.0
#         self._last_dispatch_time = 0.0
#         self._active_requests = 0
#         self._rate_limit_events = 0
#         self._total_cooldown_wait = 0.0

#     def before_request(self, page_number: int, is_retry: bool = False, attempt: int = 0) -> float:
#         """
#         Coordinates and reserves an authorized dispatch slot for a worker.
#         Handles both initial slot reservation and final pre-dispatch clearance,
#         ensuring workers never collide or fire during active global cooldowns.
#         Returns the total wait time in seconds.
#         """
#         wait_start = time.perf_counter()

#         while True:
#             # Phase 1: Slot Reservation
#             with self._lock:
#                 now = time.monotonic()
#                 earliest = max(now, self._cooldown_until)

#                 is_recovering = (now < self._recovery_until) or is_retry
#                 if is_recovering:
#                     # During rate-limit recovery: stagger dispatches smoothly
#                     interval = self.recovery_interval + random.uniform(0.1, 0.4)
#                 else:
#                     # Healthy concurrent throughput: gentle pacing
#                     interval = self.healthy_interval

#                 # Allocate slot at or after next available slot and cooldown
#                 slot = max(earliest, self._next_available_slot)
#                 self._next_available_slot = slot + interval

#             # Sleep outside the lock until our reserved slot arrives
#             delay = slot - time.monotonic()
#             if delay > 0:
#                 time.sleep(delay)

#             # Phase 2: Pre-Dispatch Verification
#             with self._lock:
#                 now = time.monotonic()
#                 # Check if a new 429 occurred while we were sleeping
#                 if now < self._cooldown_until:
#                     # Cooldown was extended by another event while we slept; loop and re-reserve
#                     continue

#                 # Clearance granted: record dispatch and increment active count
#                 self._last_dispatch_time = now
#                 self._active_requests += 1
#                 active = self._active_requests
#                 break

#         total_waited = time.perf_counter() - wait_start
#         if total_waited > 0.05:
#             with self._lock:
#                 self._total_cooldown_wait += total_waited

#         logger.info(
#             "[PERF] AI concurrency | active=%d | target=%d",
#             active,
#             self.target_workers
#         )

#         return total_waited

#     def on_success(self):
#         """Called when an API request completes successfully."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_other_error(self):
#         """Called when an API request fails with a non-rate-limit error."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_rate_limit(
#         self,
#         page_number: int,
#         attempt: int,
#         exc: Exception,
#         max_retries: int = 5
#     ) -> float:
#         """
#         Called when a worker receives an HTTP 429 rate-limit error.
#         Activates global cooldown and recovery mode across all workers.
#         Returns the computed cooldown duration in seconds.
#         """
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#             self._rate_limit_events += 1
#             now = time.monotonic()

#             retry_after = get_retry_after(exc)
#             if retry_after is not None and retry_after > 0:
#                 base_wait = retry_after
#             else:
#                 # Exponential backoff based on attempt with generous base
#                 base_wait = min(3 * (2 ** attempt), 40)

#             jitter = random.uniform(1.0, 3.0)
#             cooldown = base_wait + jitter

#             # Extend global cooldown and recovery window
#             self._cooldown_until = max(self._cooldown_until, now + base_wait)
#             self._recovery_until = max(
#                 self._recovery_until,
#                 self._cooldown_until + (self.recovery_interval * self.target_workers)
#             )

#             # Push next available slot beyond the new cooldown
#             self._next_available_slot = max(self._next_available_slot, self._cooldown_until)

#         logger.warning(
#             "[PERF] Rate limit detected | page=%s | attempt=%d | cooldown=%.2fs",
#             page_number,
#             attempt + 1,
#             cooldown
#         )

#         return cooldown

#     @property
#     def rate_limit_events(self) -> int:
#         with self._lock:
#             return self._rate_limit_events

#     @property
#     def total_cooldown_wait(self) -> float:
#         with self._lock:
#             return self._total_cooldown_wait


# def _parse_ai_response_json(raw_result: str) -> dict:
#     """
#     Robust JSON parser for AI page analysis responses.
#     Handles strict decoding, markdown code fences, embedded JSON objects, and unescaped newlines.
#     """
#     # 1. Direct parse with strict=False (allows unescaped control chars/newlines in strings)
#     try:
#         return json.loads(raw_result, strict=False)
#     except (json.JSONDecodeError, TypeError):
#         pass

#     # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
#     code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
#     if code_match:
#         try:
#             return json.loads(code_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 3. Extract outermost JSON object { ... }
#     brace_match = re.search(r"(\{.*\})", raw_result, re.DOTALL)
#     if brace_match:
#         try:
#             return json.loads(brace_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 4. Fallback cleanup: remove leading/trailing ```
#     cleaned = raw_result.strip()
#     if cleaned.startswith("```json"):
#         cleaned = cleaned[7:]
#     elif cleaned.startswith("```"):
#         cleaned = cleaned[3:]
#     if cleaned.endswith("```"):
#         cleaned = cleaned[:-3]
#     return json.loads(cleaned.strip(), strict=False)

# def _encode_and_scale_image(image_path: str, max_dimension: int = 1600, quality: int = 85) -> str:
#     """
#     Loads an image, resizes it so its longest dimension does not exceed max_dimension
#     (using LANCZOS resampling to preserve sharp text), and encodes to base64 JPEG (quality=85).
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         orig_w, orig_h = img.size
#         longest = max(orig_w, orig_h)

#         if longest > max_dimension:
#             scale = max_dimension / float(longest)
#             new_w = max(1, int(orig_w * scale))
#             new_h = max(1, int(orig_h * scale))
#             img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         buffer = io.BytesIO()
#         img.save(buffer, format="JPEG", quality=quality, optimize=True)
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")


# _easyocr_reader = None
# _easyocr_lock = threading.Lock()

# # 👇 YAHAN NAYA FUNCTION ADD KARO (existing function ke turant baad)
# def _encode_and_scale_image_tiles(image_path: str, max_dimension: int = 1568, quality: int = 85) -> list[str]:
#     """
#     Splits a tall/wide comic page into near-square, overlapping tiles
#     (per Mistral's 1:1 aspect ratio recommendation) and returns a list
#     of base64-encoded JPEG tiles, ordered top-to-bottom.
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         w, h = img.size
#         aspect = h / w

#         if aspect <= 1.3:
#             return [_encode_and_scale_image(image_path, max_dimension, quality)]

#         tile_height = w
#         overlap = int(tile_height * 0.15)
#         tiles = []
#         y = 0
#         while y < h:
#             box = (0, y, w, min(y + tile_height, h))
#             tile = img.crop(box)
#             buffer = io.BytesIO()

#             tw, th = tile.size
#             longest = max(tw, th)
#             if longest > max_dimension:
#                 scale = max_dimension / float(longest)
#                 tile = tile.resize(
#                     (max(1, int(tw * scale)), max(1, int(th * scale))),
#                     Image.Resampling.LANCZOS
#                 )

#             tile.save(buffer, format="JPEG", quality=quality, optimize=True)
#             tiles.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

#             if y + tile_height >= h:
#                 break
#             y += tile_height - overlap

#         return tiles

# def get_easyocr_reader():
#     """Lazy-loads EasyOCR Reader instance thread-safely."""
#     global _easyocr_reader
#     if _easyocr_reader is None:
#         with _easyocr_lock:
#             if _easyocr_reader is None:
#                 try:
#                     import easyocr
#                     _easyocr_reader = easyocr.Reader(["en"], gpu=False)
#                 except Exception as e:
#                     logger.warning("EasyOCR initialization error: %s", e)
#                     _easyocr_reader = False
#     return _easyocr_reader if _easyocr_reader is not False else None


# def _extract_draft_ocr(image_path: str, max_dim: int = 1000) -> str:
#     """
#     Extracts fast preliminary draft OCR text using EasyOCR to assist Vision LLM.
#     """
#     reader = get_easyocr_reader()
#     if not reader:
#         return ""
#     try:
#         with Image.open(image_path) as img:
#             if img.mode not in ("RGB", "L"):
#                 img = img.convert("RGB")
#             w, h = img.size
#             if max(w, h) > max_dim:
#                 scale = max_dim / float(max(w, h))
#                 img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
#             buffer = io.BytesIO()
#             img.save(buffer, format="JPEG", quality=75)
#             buffer.seek(0)
#             results = reader.readtext(buffer.getvalue(), detail=0)
#             return "\n".join(results).strip()
#     except Exception as e:
#         logger.debug("Draft OCR error on %s: %s", image_path, e)
#         return ""


# # -----------------------------
# # Single Page Analysis
# # -----------------------------

# def analyze_page(image_path: str, draft_ocr: str = "") -> dict:

#     image_base64 = _encode_and_scale_image(
#         image_path,
#         max_dimension=1568,
#         quality=85
#     )

#     if ENABLE_HYBRID_OCR and not draft_ocr:
#         draft_ocr = _extract_draft_ocr(image_path)

#     if draft_ocr:
#         prompt_text = (
#             f"{COMIC_PAGE_ANALYSIS_PROMPT}\n\n"
#             f"==================================================\n"
#             f"DRAFT OCR TRANSCRIPTION (From local preprocessing):\n"
#             f"Verify, correct, complete, and structure this draft against the actual image:\n"
#             f'"""\n{draft_ocr}\n"""\n'
#         )
#     else:
#         prompt_text = COMIC_PAGE_ANALYSIS_PROMPT

#     # response = client.chat.complete(
#     #     model=MODEL_NAME,
#     #     messages=[
#     #         {
#     #             "role": "user",
#     #             "content": [
#     #                 {
#     #                     "type": "text",
#     #                     "text": prompt_text
#     #                 },
#     #                 {
#     #                     "type": "image_url",
#     #                     "image_url": {
#     #                         "url": f"data:image/jpeg;base64,{image_base64}"
#     #                     }
#     #                 }
#     #             ]
#     #         }
#     #     ],
#     #     temperature=0.1,
#     # )
#     image_tiles_base64 = _encode_and_scale_image_tiles(image_path, max_dimension=1568, quality=85)

#     content_blocks = [{"type": "text", "text": prompt_text}]
#     for tile_b64 in image_tiles_base64:
#         content_blocks.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{tile_b64}"}
#         })

#     response = client.chat.complete(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "user",
#                 "content": content_blocks
#             }
#         ],
#         temperature=0.1,
#     )
#     raw_result = response.choices[0].message.content

#     # Convert AI response to JSON
#     return _parse_ai_response_json(raw_result)

# # -----------------------------
# # Build Comic-Level Full Text
# # -----------------------------

# def build_comic_full_text(results: list) -> str:
#     """
#     Page-level OCR text ko page order mein combine karta hai.

#     Important:
#     - AI ko dobara entire comic dene ki zaroorat nahi.
#     - Existing page OCR ko hi combine kiya jata hai.
#     - Page number preserve hota hai.
#     """

#     full_text_parts = []

#     ordered_results = sorted(
#         results,
#         key=lambda x: x["page_number"]
#     )

#     for page in ordered_results:

#         if page.get("status") != "success":
#             continue

#         analysis = page.get("analysis") or {}

#         text_data = analysis.get(
#             "text",
#             {}
#         )

#         page_text = text_data.get(
#             "full_text",
#             ""
#         )

#         if not isinstance(page_text, str):
#             continue

#         page_text = page_text.strip()

#         if not page_text:
#             continue

#         full_text_parts.append(
#             f"PAGE {page['page_number']}\n"
#             f"{page_text}"
#         )

#     return "\n\n".join(
#         full_text_parts
#     )


# def analyze_pages(
#     pages: list,
#     max_workers: int = MAX_AI_WORKERS,
#     max_retries: int = MAX_AI_RETRIES,
#     on_page_complete: Optional[Callable[[dict, int, int], None]] = None
# ) -> list:

#     coordinator = RateLimitCoordinator(target_workers=max_workers)

#     def process_page(page):

#         page_number = page["page_number"]
#         image_path = page["image_path"]
#         page_start = time.perf_counter()
#         attempts = 0
#         retry_wait_total = 0.0

#         if not Path(image_path).exists():
#             page_duration = time.perf_counter() - page_start
#             logger.warning("[AI] Page image %s not found on local disk.", image_path)
#             return {
#                 "page_number": page_number,
#                 "filename": page.get("filename", Path(image_path).name),
#                 "image_path": image_path,
#                 "analysis": None,
#                 "metadata": {
#                     "page_number": page_number
#                 },
#                 "status": "error",
#                 "error": f"Image file not found: {image_path}",
#                 "perf": {
#                     "duration": page_duration,
#                     "attempts": 1,
#                     "retry_wait": 0.0
#                 }
#             }

#         for attempt in range(
#             max_retries + 1
#         ):
#             attempts += 1

#             # Request slot authorization from the coordinator
#             wait_time = coordinator.before_request(
#                 page_number=page_number,
#                 is_retry=(attempt > 0),
#                 attempt=attempt
#             )
#             retry_wait_total += wait_time

#             try:

#                 result = analyze_page(
#                     image_path
#                 )

#                 coordinator.on_success()

#                 page_duration = time.perf_counter() - page_start

#                 logger.info(
#                     "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs",
#                     page_number,
#                     page_duration,
#                     attempts,
#                     retry_wait_total
#                 )

#                 return {
#                     "page_number": page_number,
#                     "filename": page.get("filename", Path(image_path).name),
#                     "image_path": image_path,
#                     "analysis": result,
#                     "metadata": {
#                         "page_number": page_number,
#                         "has_text": bool(
#                             (
#                                 result.get("text", {})
#                                 .get("full_text", "")
#                                 if isinstance(
#                                     result.get("text", {}),
#                                     dict
#                                 )
#                                 else ""
#                             ).strip()
#                         )
#                     },
#                     "status": "success",
#                     "perf": {
#                         "duration": page_duration,
#                         "attempts": attempts,
#                         "retry_wait": retry_wait_total
#                     }
#                 }

#             except Exception as e:

#                 error_message = str(e)

#                 if is_rate_limit_error(e):

#                     if attempt >= max_retries:
#                         coordinator.on_other_error()
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED RATE LIMIT)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#                     # Register rate limit with coordinator and sleep with exponential backoff + jitter
#                     cooldown = coordinator.on_rate_limit(
#                         page_number=page_number,
#                         attempt=attempt,
#                         exc=e,
#                         max_retries=max_retries
#                     )
#                     time.sleep(cooldown)
#                     retry_wait_total += cooldown

#                 else:
#                     # Non-rate-limit error (e.g. transient network or JSON format error)
#                     coordinator.on_other_error()

#                     if attempt < max_retries:
#                         logger.warning(
#                             "Page %s encountered error (%s). Retrying in 1.5s (attempt %s/%s)...",
#                             page_number,
#                             error_message,
#                             attempt + 1,
#                             max_retries
#                         )
#                         retry_start = time.perf_counter()
#                         time.sleep(1.5 + random.uniform(0.1, 0.5))
#                         retry_wait_total += time.perf_counter() - retry_start
#                     else:
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED ERROR)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#     results = []
#     total_pages = len(pages)
#     ai_start = time.perf_counter()

#     # -----------------------------
#     # Parallel Processing
#     # -----------------------------

#     with ThreadPoolExecutor(
#         max_workers=max_workers
#     ) as executor:

#         futures = {
#             executor.submit(
#                 process_page,
#                 page
#             ): page["page_number"]

#             for page in pages
#         }

#         completed = 0

#         for future in as_completed(
#             futures
#         ):

#             result = future.result()

#             results.append(
#                 result
#             )

#             completed += 1

#             if result["status"] == "success":
#                 logger.info(
#                     "[%s/%s] Page %s analyzed successfully",
#                     completed,
#                     total_pages,
#                     result["page_number"]
#                 )
#             else:
#                 logger.error(
#                     "[%s/%s] Page %s analysis failed: %s",
#                     completed,
#                     total_pages,
#                     result["page_number"],
#                     result.get("error", "Unknown error")
#                 )

#             if on_page_complete is not None:
#                 try:
#                     on_page_complete(result, completed, total_pages)
#                 except Exception as cb_err:
#                     logger.warning("on_page_complete callback error: %s", str(cb_err))

#     total_ai_duration = time.perf_counter() - ai_start

#     # -----------------------------
#     # Performance Aggregate Stats
#     # -----------------------------
#     successful = sum(1 for r in results if r.get("status") == "success")
#     failed = sum(1 for r in results if r.get("status") == "error")
#     perf_list = [r.get("perf", {}) for r in results]
#     pages_with_retries = sum(1 for p in perf_list if p.get("attempts", 1) > 1)
#     total_retries = sum(max(0, p.get("attempts", 1) - 1) for p in perf_list)
#     total_retry_wait = sum(p.get("retry_wait", 0.0) for p in perf_list)
#     durations = [p.get("duration", 0.0) for p in perf_list]
#     avg_page_duration = (sum(durations) / len(durations)) if durations else 0.0
#     fastest_page = min(durations) if durations else 0.0
#     slowest_page = max(durations) if durations else 0.0

#     no_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) == 1]
#     avg_no_retry = (sum(no_retry_durations) / len(no_retry_durations)) if no_retry_durations else 0.0
#     with_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) > 1]
#     avg_with_retry = (sum(with_retry_durations) / len(with_retry_durations)) if with_retry_durations else 0.0

#     logger.info(
#         "\n"
#         "[PERF] ==================================================\n"
#         "[PERF] AI ANALYSIS PERFORMANCE\n"
#         "[PERF] Pages: %s\n"
#         "[PERF] Workers: %s\n"
#         "[PERF] Total duration: %.2fs\n"
#         "[PERF] Successful: %s\n"
#         "[PERF] Failed: %s\n"
#         "[PERF] Pages with retries: %s\n"
#         "[PERF] Total retries: %s\n"
#         "[PERF] Total retry wait: %.2fs\n"
#         "[PERF] Rate-limit events: %s\n"
#         "[PERF] Total provider cooldown time: %.2fs\n"
#         "[PERF] Average page duration: %.2fs\n"
#         "[PERF]   - No-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF]   - With-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF] Fastest page: %.2fs\n"
#         "[PERF] Slowest page: %.2fs\n"
#         "[PERF] ==================================================",
#         total_pages,
#         max_workers,
#         total_ai_duration,
#         successful,
#         failed,
#         pages_with_retries,
#         total_retries,
#         total_retry_wait,
#         coordinator.rate_limit_events,
#         coordinator.total_cooldown_wait,
#         avg_page_duration,
#         avg_no_retry,
#         len(no_retry_durations),
#         avg_with_retry,
#         len(with_retry_durations),
#         fastest_page,
#         slowest_page
#     )

#     # -----------------------------
#     # Restore Page Order
#     # -----------------------------

#     results.sort(
#         key=lambda x: x["page_number"]
#     )

#     return results


# import base64
# import io
# import json
# import random
# import re
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Callable, Optional
# from PIL import Image

# from mistralai.client import Mistral
# from mistralai.client.errors import MistralError

# from app.core.config import (
#     ENABLE_HYBRID_OCR,
#     MAX_AI_RETRIES,
#     MAX_AI_WORKERS,
#     MISTRAL_API_KEY,
#     MISTRAL_MODEL,
# )
# from app.core.logging import logger

# client = Mistral(
#     api_key=MISTRAL_API_KEY
# )



# MODEL_NAME = MISTRAL_MODEL

# # -----------------------------
# # Module-Level Analysis Prompt
# # -----------------------------

# COMIC_PAGE_ANALYSIS_PROMPT = """
# NOTE: The image(s) provided may be sequential top-to-bottom slices of a SINGLE tall comic page
# (with slight overlap between slices). If multiple images are given, treat them together as ONE
# continuous page — do not treat slices as separate pages or invent extra panels just because the
# page was sliced.

# You are a universal Comic Page Analyzer and Extraction Engine for a multimodal RAG pipeline.
# Your task is to accurately extract all text and describe the visual contents of the provided page image.

# This engine must process ANY genre, style, and sequential art format:
# - Western Comics & Graphic Novels (Superhero, Sci-Fi, Dark Fantasy, Crime/Noir, Indie)
# - Manga, Manhwa, & Webtoons (Shonen, Seinen, Shojo, Webcomics, vertical or horizontal layouts)
# - Multi-panel grids, full-page splash art, covers, chapter title pages, or silent action scenes.

# ==================================================
# 1. COMPLETE TEXT TRANSCRIPTION & OCR (HIGHEST PRIORITY)
# ==================================================
# - Scan EVERY panel across the entire page from top to bottom (following natural reading order: left-to-right or right-to-left as drawn).
# - Transcribe ALL text: speech bubbles, thought clouds, narration/monologue boxes, captions, signs, titles, and sound effects (SFX).
# - Do NOT skip any dialogue or text boxes. Capture all text present on the page faithfully.
# - Preserve exact wording, casing, and punctuation as visible.
# - If a page has no text, return empty string "" for full_text and empty list [] for dialogue_and_narration.

# ==================================================
# 2. PRECISE & GROUNDED VISUAL EXTRACTION (PANEL-SCOPED)
# ==================================================
# - Identify each distinct panel on the page separately (a full-page splash counts as ONE panel).
# - For EACH panel, describe ONLY the figures/objects/actions physically visible WITHIN that panel's borders.
# - NEVER borrow, merge, or transfer a detail (emotion, pose, clothing, identity) from one panel into your
#   description of a different panel, even if the figures look similar.
# - If the SAME figure (by consistent visual traits: hair, clothing, size) appears in multiple panels,
#   describe it separately in each panel's context — do NOT assume it is a different, new character just
#   because the panel changed, AND do NOT assume it is the same character unless traits genuinely match.
# - Do NOT invent a figure, pose, or emotional expression (e.g. "distressed", "looking up") unless it is
#   unambiguously visible in that specific panel. If uncertain, omit the detail rather than guessing.
# - Count actual panel borders/gutters visible in the image(s). Do NOT invent additional panels to match
#   the number of narration text blocks — text boxes and panel count are independent.
# - Describe the artwork EXACTLY as drawn — colors, poses, objects — based purely on visual pixels, even if
#   it seems to contradict, undershoot, or not directly illustrate the narration text. NEVER infer visual
#   details (powers, emotions, objects) purely from what the dialogue/narration text implies.

# ==================================================
# 3. OUTPUT FORMAT
# ==================================================
# Return ONLY a valid JSON object matching this exact schema:
# {
#     "page_summary": "Comprehensive narrative summary of the page's events, characters, visual action, and story progression.",
#     "panels_detected": 1,
#     "panels": [
#         {
#             "panel_index": 1,
#             "dialogue_and_narration": [
#                 "Exact transcribed dialogue or narration block belonging to THIS panel"
#             ],
#             "characters": [
#                 "Description of visible figure 1 in THIS panel only",
#                 "Description of visible figure 2 in THIS panel only"
#             ],
#             "actions": [
#                 "Description of visible action or interaction in THIS panel only"
#             ],
#             "environment": "Description of the physical setting or location in THIS panel only.",
#             "objects": [
#                 "Notable visible object or item in THIS panel only"
#             ]
#         }
#     ],
#     "text": {
#         "full_text": "Exact concatenated transcription of all readable text on this page in natural reading order.",
#         "dialogue_and_narration": [
#             "Exact transcribed dialogue or narration block 1",
#             "Exact transcribed dialogue or narration block 2"
#         ],
#         "sound_effects": [],
#         "signs_and_labels": []
#     },
#     "visual_description": {
#         "characters": [
#             "Description of visible character/figure 1",
#             "Description of visible character/figure 2"
#         ],
#         "actions": [
#             "Description of visible action or interaction 1",
#             "Description of visible action or interaction 2"
#         ],
#         "environment": "Description of the physical setting or location.",
#         "objects": [
#             "Notable visible object or item 1",
#             "Notable visible object or item 2"
#         ],
#         "background": "Description of background details, lighting, and mood.",
#         "other_details": ""
#     }
# }
# """


# def is_rate_limit_error(exc: Exception) -> bool:
#     """
#     Determines if an exception represents an HTTP 429 Rate Limit error.
#     Checks the Mistral SDK MistralError status_code, underlying response status_code,
#     or common rate limit error messages.
#     """
#     if isinstance(exc, MistralError) and getattr(exc, "status_code", None) == 429:
#         return True
#     if getattr(exc, "status_code", None) == 429:
#         return True
#     raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#     if raw_resp is not None and getattr(raw_resp, "status_code", None) == 429:
#         return True
#     err_str = str(exc).lower()
#     if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
#         return True
#     return False


# def get_retry_after(exc: Exception) -> float | None:
#     """
#     Extracts Retry-After header or provider delay duration from exception details.
#     """
#     headers = getattr(exc, "headers", None)
#     if headers is None:
#         raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#         if raw_resp is not None:
#             headers = getattr(raw_resp, "headers", None)
#     if headers:
#         retry_val = headers.get("retry-after") or headers.get("Retry-After")
#         if retry_val:
#             try:
#                 val = float(retry_val)
#                 if val > 0:
#                     return val
#             except (ValueError, TypeError):
#                 pass

#     # Extract delay from error message if available (e.g. 'try again in 12.4s')
#     err_str = str(exc)
#     match = re.search(r"(?:try again in|retry after|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", err_str, re.I)
#     if match:
#         try:
#             val = float(match.group(1))
#             if val > 0:
#                 return val
#         except (ValueError, TypeError):
#             pass

#     return None


# # -----------------------------
# # Rate Limit Coordinator (M-25)
# # -----------------------------

# class RateLimitCoordinator:
#     """
#     Centralized rate-limit and recovery coordination across worker threads.
#     - Implements slot reservation to prevent synchronized retry stampedes after 429s.
#     - Staggers recovery dispatches so workers resume gradually rather than in lockstep bursts.
#     - Maintains normal full concurrency of workers when API is healthy.
#     - Validates clearance immediately before dispatch to guarantee no collisions during extended cooldowns.
#     """

#     def __init__(
#         self,
#         target_workers: int = MAX_AI_WORKERS,
#         healthy_interval: float = 0.20,
#         recovery_interval: float = 1.80
#     ):
#         self.target_workers = target_workers
#         self.healthy_interval = healthy_interval
#         self.recovery_interval = recovery_interval

#         self._lock = threading.Lock()
#         self._cooldown_until = 0.0
#         self._recovery_until = 0.0
#         self._next_available_slot = 0.0
#         self._last_dispatch_time = 0.0
#         self._active_requests = 0
#         self._rate_limit_events = 0
#         self._total_cooldown_wait = 0.0

#     def before_request(self, page_number: int, is_retry: bool = False, attempt: int = 0) -> float:
#         """
#         Coordinates and reserves an authorized dispatch slot for a worker.
#         Handles both initial slot reservation and final pre-dispatch clearance,
#         ensuring workers never collide or fire during active global cooldowns.
#         Returns the total wait time in seconds.
#         """
#         wait_start = time.perf_counter()

#         while True:
#             # Phase 1: Slot Reservation
#             with self._lock:
#                 now = time.monotonic()
#                 earliest = max(now, self._cooldown_until)

#                 is_recovering = (now < self._recovery_until) or is_retry
#                 if is_recovering:
#                     # During rate-limit recovery: stagger dispatches smoothly
#                     interval = self.recovery_interval + random.uniform(0.1, 0.4)
#                 else:
#                     # Healthy concurrent throughput: gentle pacing
#                     interval = self.healthy_interval

#                 # Allocate slot at or after next available slot and cooldown
#                 slot = max(earliest, self._next_available_slot)
#                 self._next_available_slot = slot + interval

#             # Sleep outside the lock until our reserved slot arrives
#             delay = slot - time.monotonic()
#             if delay > 0:
#                 time.sleep(delay)

#             # Phase 2: Pre-Dispatch Verification
#             with self._lock:
#                 now = time.monotonic()
#                 # Check if a new 429 occurred while we were sleeping
#                 if now < self._cooldown_until:
#                     # Cooldown was extended by another event while we slept; loop and re-reserve
#                     continue

#                 # Clearance granted: record dispatch and increment active count
#                 self._last_dispatch_time = now
#                 self._active_requests += 1
#                 active = self._active_requests
#                 break

#         total_waited = time.perf_counter() - wait_start
#         if total_waited > 0.05:
#             with self._lock:
#                 self._total_cooldown_wait += total_waited

#         logger.info(
#             "[PERF] AI concurrency | active=%d | target=%d",
#             active,
#             self.target_workers
#         )

#         return total_waited

#     def on_success(self):
#         """Called when an API request completes successfully."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_other_error(self):
#         """Called when an API request fails with a non-rate-limit error."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_rate_limit(
#         self,
#         page_number: int,
#         attempt: int,
#         exc: Exception,
#         max_retries: int = 5
#     ) -> float:
#         """
#         Called when a worker receives an HTTP 429 rate-limit error.
#         Activates global cooldown and recovery mode across all workers.
#         Returns the computed cooldown duration in seconds.
#         """
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#             self._rate_limit_events += 1
#             now = time.monotonic()

#             retry_after = get_retry_after(exc)
#             if retry_after is not None and retry_after > 0:
#                 base_wait = retry_after
#             else:
#                 # Exponential backoff based on attempt with generous base
#                 base_wait = min(3 * (2 ** attempt), 40)

#             jitter = random.uniform(1.0, 3.0)
#             cooldown = base_wait + jitter

#             # Extend global cooldown and recovery window
#             self._cooldown_until = max(self._cooldown_until, now + base_wait)
#             self._recovery_until = max(
#                 self._recovery_until,
#                 self._cooldown_until + (self.recovery_interval * self.target_workers)
#             )

#             # Push next available slot beyond the new cooldown
#             self._next_available_slot = max(self._next_available_slot, self._cooldown_until)

#         logger.warning(
#             "[PERF] Rate limit detected | page=%s | attempt=%d | cooldown=%.2fs",
#             page_number,
#             attempt + 1,
#             cooldown
#         )

#         return cooldown

#     @property
#     def rate_limit_events(self) -> int:
#         with self._lock:
#             return self._rate_limit_events

#     @property
#     def total_cooldown_wait(self) -> float:
#         with self._lock:
#             return self._total_cooldown_wait


# def _parse_ai_response_json(raw_result: str) -> dict:
#     """
#     Robust JSON parser for AI page analysis responses.
#     Handles strict decoding, markdown code fences, embedded JSON objects, and unescaped newlines.
#     """
#     # 1. Direct parse with strict=False (allows unescaped control chars/newlines in strings)
#     try:
#         return json.loads(raw_result, strict=False)
#     except (json.JSONDecodeError, TypeError):
#         pass

#     # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
#     code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
#     if code_match:
#         try:
#             return json.loads(code_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 3. Extract outermost JSON object { ... }
#     brace_match = re.search(r"(\{.*\})", raw_result, re.DOTALL)
#     if brace_match:
#         try:
#             return json.loads(brace_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 4. Fallback cleanup: remove leading/trailing ```
#     cleaned = raw_result.strip()
#     if cleaned.startswith("```json"):
#         cleaned = cleaned[7:]
#     elif cleaned.startswith("```"):
#         cleaned = cleaned[3:]
#     if cleaned.endswith("```"):
#         cleaned = cleaned[:-3]
#     return json.loads(cleaned.strip(), strict=False)


# def _encode_and_scale_image(image_path: str, max_dimension: int = 1600, quality: int = 85) -> str:
#     """
#     Loads an image, resizes it so its longest dimension does not exceed max_dimension
#     (using LANCZOS resampling to preserve sharp text), and encodes to base64 JPEG (quality=85).
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         orig_w, orig_h = img.size
#         longest = max(orig_w, orig_h)

#         if longest > max_dimension:
#             scale = max_dimension / float(longest)
#             new_w = max(1, int(orig_w * scale))
#             new_h = max(1, int(orig_h * scale))
#             img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         buffer = io.BytesIO()
#         img.save(buffer, format="JPEG", quality=quality, optimize=True)
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")


# def _encode_and_scale_image_tiles(image_path: str, max_dimension: int = 1568, quality: int = 85) -> list[str]:
#     """
#     Splits a tall/wide comic page into near-square, overlapping tiles
#     (per Mistral's 1:1 aspect ratio recommendation for vision models) and
#     returns a list of base64-encoded JPEG tiles, ordered top-to-bottom.

#     Comic pages are typically very tall/portrait (aspect ratio far from 1:1).
#     Sending the whole page as one elongated image degrades Mistral's vision
#     accuracy on this model family and causes hallucinated/blended panel
#     content. Slicing into near-square, slightly-overlapping tiles keeps each
#     slice close to the recommended aspect ratio while preserving continuity
#     via the overlap.
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         w, h = img.size
#         aspect = h / w

#         # If already close to square, just encode as one tile (no slicing needed)
#         if aspect <= 1.3:
#             return [_encode_and_scale_image(image_path, max_dimension, quality)]

#         # Otherwise slice into near-square tiles with a small overlap for continuity
#         tile_height = w
#         overlap = int(tile_height * 0.15)
#         tiles = []
#         y = 0
#         while y < h:
#             box = (0, y, w, min(y + tile_height, h))
#             tile = img.crop(box)
#             buffer = io.BytesIO()

#             # Scale tile so its longest side <= max_dimension
#             tw, th = tile.size
#             longest = max(tw, th)
#             if longest > max_dimension:
#                 scale = max_dimension / float(longest)
#                 tile = tile.resize(
#                     (max(1, int(tw * scale)), max(1, int(th * scale))),
#                     Image.Resampling.LANCZOS
#                 )

#             tile.save(buffer, format="JPEG", quality=quality, optimize=True)
#             tiles.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

#             if y + tile_height >= h:
#                 break
#             y += tile_height - overlap

#         return tiles


# _easyocr_reader = None
# _easyocr_lock = threading.Lock()


# def get_easyocr_reader():
#     """Lazy-loads EasyOCR Reader instance thread-safely."""
#     global _easyocr_reader
#     if _easyocr_reader is None:
#         with _easyocr_lock:
#             if _easyocr_reader is None:
#                 try:
#                     import easyocr
#                     _easyocr_reader = easyocr.Reader(["en"], gpu=False)
#                 except Exception as e:
#                     logger.warning("EasyOCR initialization error: %s", e)
#                     _easyocr_reader = False
#     return _easyocr_reader if _easyocr_reader is not False else None


# def _extract_draft_ocr(image_path: str, max_dim: int = 1000) -> str:
#     """
#     Extracts fast preliminary draft OCR text using EasyOCR to assist Vision LLM.
#     """
#     reader = get_easyocr_reader()
#     if not reader:
#         return ""
#     try:
#         with Image.open(image_path) as img:
#             if img.mode not in ("RGB", "L"):
#                 img = img.convert("RGB")
#             w, h = img.size
#             if max(w, h) > max_dim:
#                 scale = max_dim / float(max(w, h))
#                 img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
#             buffer = io.BytesIO()
#             img.save(buffer, format="JPEG", quality=75)
#             buffer.seek(0)
#             results = reader.readtext(buffer.getvalue(), detail=0)
#             return "\n".join(results).strip()
#     except Exception as e:
#         logger.debug("Draft OCR error on %s: %s", image_path, e)
#         return ""


# # -----------------------------
# # Single Page Analysis
# # -----------------------------

# def analyze_page(image_path: str, draft_ocr: str = "") -> dict:

#     image_tiles_base64 = _encode_and_scale_image_tiles(
#         image_path,
#         max_dimension=1568,
#         quality=85
#     )

#     if ENABLE_HYBRID_OCR and not draft_ocr:
#         draft_ocr = _extract_draft_ocr(image_path)

#     if draft_ocr:
#         prompt_text = (
#             f"{COMIC_PAGE_ANALYSIS_PROMPT}\n\n"
#             f"==================================================\n"
#             f"DRAFT OCR TRANSCRIPTION (From local preprocessing):\n"
#             f"Verify, correct, complete, and structure this draft against the actual image:\n"
#             f'"""\n{draft_ocr}\n"""\n'
#         )
#     else:
#         prompt_text = COMIC_PAGE_ANALYSIS_PROMPT

#     content_blocks = [{"type": "text", "text": prompt_text}]
#     for tile_b64 in image_tiles_base64:
#         content_blocks.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{tile_b64}"
#             }
#         })

#     response = client.chat.complete(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "user",
#                 "content": content_blocks
#             }
#         ],
#         temperature=0.1,
#     )

#     raw_result = response.choices[0].message.content

#     # Convert AI response to JSON
#     return _parse_ai_response_json(raw_result)

# # -----------------------------
# # Build Comic-Level Full Text
# # -----------------------------

# def build_comic_full_text(results: list) -> str:
#     """
#     Page-level OCR text ko page order mein combine karta hai.

#     Important:
#     - AI ko dobara entire comic dene ki zaroorat nahi.
#     - Existing page OCR ko hi combine kiya jata hai.
#     - Page number preserve hota hai.
#     """

#     full_text_parts = []

#     ordered_results = sorted(
#         results,
#         key=lambda x: x["page_number"]
#     )

#     for page in ordered_results:

#         if page.get("status") != "success":
#             continue

#         analysis = page.get("analysis") or {}

#         text_data = analysis.get(
#             "text",
#             {}
#         )

#         page_text = text_data.get(
#             "full_text",
#             ""
#         )

#         if not isinstance(page_text, str):
#             continue

#         page_text = page_text.strip()

#         if not page_text:
#             continue

#         full_text_parts.append(
#             f"PAGE {page['page_number']}\n"
#             f"{page_text}"
#         )

#     return "\n\n".join(
#         full_text_parts
#     )


# def analyze_pages(
#     pages: list,
#     max_workers: int = MAX_AI_WORKERS,
#     max_retries: int = MAX_AI_RETRIES,
#     on_page_complete: Optional[Callable[[dict, int, int], None]] = None
# ) -> list:

#     coordinator = RateLimitCoordinator(target_workers=max_workers)

#     def process_page(page):

#         page_number = page["page_number"]
#         image_path = page["image_path"]
#         page_start = time.perf_counter()
#         attempts = 0
#         retry_wait_total = 0.0

#         if not Path(image_path).exists():
#             page_duration = time.perf_counter() - page_start
#             logger.warning("[AI] Page image %s not found on local disk.", image_path)
#             return {
#                 "page_number": page_number,
#                 "filename": page.get("filename", Path(image_path).name),
#                 "image_path": image_path,
#                 "analysis": None,
#                 "metadata": {
#                     "page_number": page_number
#                 },
#                 "status": "error",
#                 "error": f"Image file not found: {image_path}",
#                 "perf": {
#                     "duration": page_duration,
#                     "attempts": 1,
#                     "retry_wait": 0.0
#                 }
#             }

#         for attempt in range(
#             max_retries + 1
#         ):
#             attempts += 1

#             # Request slot authorization from the coordinator
#             wait_time = coordinator.before_request(
#                 page_number=page_number,
#                 is_retry=(attempt > 0),
#                 attempt=attempt
#             )
#             retry_wait_total += wait_time

#             try:

#                 result = analyze_page(
#                     image_path
#                 )

#                 coordinator.on_success()

#                 page_duration = time.perf_counter() - page_start

#                 logger.info(
#                     "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs",
#                     page_number,
#                     page_duration,
#                     attempts,
#                     retry_wait_total
#                 )

#                 return {
#                     "page_number": page_number,
#                     "filename": page.get("filename", Path(image_path).name),
#                     "image_path": image_path,
#                     "analysis": result,
#                     "metadata": {
#                         "page_number": page_number,
#                         "has_text": bool(
#                             (
#                                 result.get("text", {})
#                                 .get("full_text", "")
#                                 if isinstance(
#                                     result.get("text", {}),
#                                     dict
#                                 )
#                                 else ""
#                             ).strip()
#                         )
#                     },
#                     "status": "success",
#                     "perf": {
#                         "duration": page_duration,
#                         "attempts": attempts,
#                         "retry_wait": retry_wait_total
#                     }
#                 }

#             except Exception as e:

#                 error_message = str(e)

#                 if is_rate_limit_error(e):

#                     if attempt >= max_retries:
#                         coordinator.on_other_error()
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED RATE LIMIT)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#                     # Register rate limit with coordinator and sleep with exponential backoff + jitter
#                     cooldown = coordinator.on_rate_limit(
#                         page_number=page_number,
#                         attempt=attempt,
#                         exc=e,
#                         max_retries=max_retries
#                     )
#                     time.sleep(cooldown)
#                     retry_wait_total += cooldown

#                 else:
#                     # Non-rate-limit error (e.g. transient network or JSON format error)
#                     coordinator.on_other_error()

#                     if attempt < max_retries:
#                         logger.warning(
#                             "Page %s encountered error (%s). Retrying in 1.5s (attempt %s/%s)...",
#                             page_number,
#                             error_message,
#                             attempt + 1,
#                             max_retries
#                         )
#                         retry_start = time.perf_counter()
#                         time.sleep(1.5 + random.uniform(0.1, 0.5))
#                         retry_wait_total += time.perf_counter() - retry_start
#                     else:
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED ERROR)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#     results = []
#     total_pages = len(pages)
#     ai_start = time.perf_counter()

#     # -----------------------------
#     # Parallel Processing
#     # -----------------------------

#     with ThreadPoolExecutor(
#         max_workers=max_workers
#     ) as executor:

#         futures = {
#             executor.submit(
#                 process_page,
#                 page
#             ): page["page_number"]

#             for page in pages
#         }

#         completed = 0

#         for future in as_completed(
#             futures
#         ):

#             result = future.result()

#             results.append(
#                 result
#             )

#             completed += 1

#             if result["status"] == "success":
#                 logger.info(
#                     "[%s/%s] Page %s analyzed successfully",
#                     completed,
#                     total_pages,
#                     result["page_number"]
#                 )
#             else:
#                 logger.error(
#                     "[%s/%s] Page %s analysis failed: %s",
#                     completed,
#                     total_pages,
#                     result["page_number"],
#                     result.get("error", "Unknown error")
#                 )

#             if on_page_complete is not None:
#                 try:
#                     on_page_complete(result, completed, total_pages)
#                 except Exception as cb_err:
#                     logger.warning("on_page_complete callback error: %s", str(cb_err))

#     total_ai_duration = time.perf_counter() - ai_start

#     # -----------------------------
#     # Performance Aggregate Stats
#     # -----------------------------
#     successful = sum(1 for r in results if r.get("status") == "success")
#     failed = sum(1 for r in results if r.get("status") == "error")
#     perf_list = [r.get("perf", {}) for r in results]
#     pages_with_retries = sum(1 for p in perf_list if p.get("attempts", 1) > 1)
#     total_retries = sum(max(0, p.get("attempts", 1) - 1) for p in perf_list)
#     total_retry_wait = sum(p.get("retry_wait", 0.0) for p in perf_list)
#     durations = [p.get("duration", 0.0) for p in perf_list]
#     avg_page_duration = (sum(durations) / len(durations)) if durations else 0.0
#     fastest_page = min(durations) if durations else 0.0
#     slowest_page = max(durations) if durations else 0.0

#     no_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) == 1]
#     avg_no_retry = (sum(no_retry_durations) / len(no_retry_durations)) if no_retry_durations else 0.0
#     with_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) > 1]
#     avg_with_retry = (sum(with_retry_durations) / len(with_retry_durations)) if with_retry_durations else 0.0

#     logger.info(
#         "\n"
#         "[PERF] ==================================================\n"
#         "[PERF] AI ANALYSIS PERFORMANCE\n"
#         "[PERF] Pages: %s\n"
#         "[PERF] Workers: %s\n"
#         "[PERF] Total duration: %.2fs\n"
#         "[PERF] Successful: %s\n"
#         "[PERF] Failed: %s\n"
#         "[PERF] Pages with retries: %s\n"
#         "[PERF] Total retries: %s\n"
#         "[PERF] Total retry wait: %.2fs\n"
#         "[PERF] Rate-limit events: %s\n"
#         "[PERF] Total provider cooldown time: %.2fs\n"
#         "[PERF] Average page duration: %.2fs\n"
#         "[PERF]   - No-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF]   - With-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF] Fastest page: %.2fs\n"
#         "[PERF] Slowest page: %.2fs\n"
#         "[PERF] ==================================================",
#         total_pages,
#         max_workers,
#         total_ai_duration,
#         successful,
#         failed,
#         pages_with_retries,
#         total_retries,
#         total_retry_wait,
#         coordinator.rate_limit_events,
#         coordinator.total_cooldown_wait,
#         avg_page_duration,
#         avg_no_retry,
#         len(no_retry_durations),
#         avg_with_retry,
#         len(with_retry_durations),
#         fastest_page,
#         slowest_page
#     )

#     # -----------------------------
#     # Restore Page Order
#     # -----------------------------

#     results.sort(
#         key=lambda x: x["page_number"]
#     )

#     return results




# import base64
# import io
# import json
# import random
# import re
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Callable, Optional
# from PIL import Image

# from mistralai.client import Mistral
# from mistralai.client.errors import MistralError

# from app.core.config import (
#     ENABLE_HYBRID_OCR,
#     MAX_AI_RETRIES,
#     MAX_AI_WORKERS,
#     MISTRAL_API_KEY,
#     MISTRAL_MODEL,
# )
# from app.core.logging import logger

# # -----------------------------
# # NEW: Gemini config
# # -----------------------------
# # Add these to app/core/config.py (same pattern as the Mistral ones):
# #   GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# #   GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro")
# #   GEMINI_MAX_RPM = float(os.getenv("GEMINI_MAX_RPM", "5"))
# #
# # IMPORTANT: Pro models were removed from Gemini's free tier (~April 2026).
# # If your project has no billing attached, gemini-3.1-pro will likely
# # 429/RESOURCE_EXHAUSTED. Either enable billing (Tier 1, no min spend,
# # jumps to 150-300 RPM) or set GEMINI_MODEL to "gemini-3.5-flash" (10 RPM
# # free) or "gemini-3.1-flash-lite" (15 RPM free). Check your live quota at
# # Google AI Studio -> your project -> quota page, and set GEMINI_MAX_RPM
# # to match it (stay slightly under the real number for safety margin).
# try:
#     from app.core.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_RPM
# except ImportError:
#     import os
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6KfzpRrPpNUChRij9vls0B6HRVnuqEy7MHR8YWGFEPPvA")
#     GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
#     GEMINI_MAX_RPM = float(os.getenv("GEMINI_MAX_RPM", "5"))

# client = Mistral(
#     api_key=MISTRAL_API_KEY
# )



# MODEL_NAME = MISTRAL_MODEL

# # -----------------------------
# # Module-Level Analysis Prompts
# # -----------------------------
# # Split into two independent prompts so the visual-description model NEVER
# # sees dialogue/narration text in the same call. Mixing them previously
# # caused the model to invent fantastical visuals to match narration mood
# # (e.g. "demons" in the text -> an invented glowing/demonic figure in the
# # "visual" description that was not actually drawn on the page).

# VISUAL_ONLY_PROMPT = """
# NOTE: The image(s) provided may be sequential top-to-bottom slices of a SINGLE tall comic page
# (with slight overlap between slices). If multiple images are given, treat them together as ONE
# continuous page — do not treat slices as separate pages or invent extra panels just because the
# page was sliced.

# You are a Comic Page VISUAL ANALYZER for a multimodal RAG pipeline. You have NOT been given any
# dialogue, narration, or caption text — do not assume, guess, or invent any. Describe purely what
# you see in the pixels.

# This engine must process ANY genre, style, and sequential art format:
# - Western Comics & Graphic Novels (Superhero, Sci-Fi, Dark Fantasy, Crime/Noir, Indie)
# - Manga, Manhwa, & Webtoons (Shonen, Seinen, Shojo, Webcomics, vertical or horizontal layouts)
# - Multi-panel grids, full-page splash art, covers, chapter title pages, or silent action scenes.

# ==================================================
# CRITICAL GROUNDING RULES
# ==================================================
# 1. Describe ONLY figures, objects, and actions that are physically visible in the artwork.
# 2. Do NOT invent a creature, monster, demon, supernatural entity, magical effect, glowing object,
#    or fantastical element unless you can literally see it drawn (e.g. actual claws, horns, wings,
#    a glowing aura rendered in the art). If every figure you see is an ordinary human being in
#    ordinary clothing, say so plainly — do not add anything fantastical.
# 3. If a detail is unclear or you're not confident, write "[uncertain]" rather than guessing.
# 4. Do NOT name characters with real/known comic character names (e.g. do not write "Victor Von
#    Doom"). Describe figures visually (hair, clothing, build, pose) only.
# 5. Describe the artwork EXACTLY as drawn — colors, poses, objects — based purely on visual pixels.
#    NEVER infer visual details (powers, emotions, objects) from what narration text might imply,
#    since you have not been given any narration text.

# ==================================================
# PANEL-SCOPED EXTRACTION
# ==================================================
# - Identify each distinct panel on the page (a full-page splash counts as ONE panel).
# - Count ONLY panel borders/gutters you can actually see in the image. Do not guess a panel count
#   based on anything else (e.g. do not match panel count to a number of text blocks — you have no
#   text blocks here anyway).
# - For EACH panel, describe ONLY the figures/objects/actions physically visible WITHIN that panel's
#   borders. NEVER borrow, merge, or transfer a detail (emotion, pose, clothing, identity) from one
#   panel into your description of a different panel, even if the figures look similar.
# - If the SAME figure (by consistent visual traits: hair, clothing, size) appears in multiple
#   panels, describe it separately in each panel's context — do NOT assume it is a different, new
#   character just because the panel changed, AND do NOT assume it is the same character unless
#   traits genuinely match.

# ==================================================
# OUTPUT FORMAT
# ==================================================
# Return ONLY a valid JSON object matching this exact schema (no markdown fences, no commentary):
# {
#     "page_summary": "Neutral visual-only summary of what is drawn on the page.",
#     "panels_detected": 1,
#     "panels": [
#         {
#             "panel_index": 1,
#             "characters": [
#                 "Description of visible figure 1 in THIS panel only",
#                 "Description of visible figure 2 in THIS panel only"
#             ],
#             "actions": [
#                 "Description of visible action or interaction in THIS panel only"
#             ],
#             "environment": "Description of the physical setting or location in THIS panel only.",
#             "objects": [
#                 "Notable visible object or item in THIS panel only"
#             ]
#         }
#     ],
#     "visual_description": {
#         "characters": [
#             "Description of visible character/figure 1",
#             "Description of visible character/figure 2"
#         ],
#         "actions": [
#             "Description of visible action or interaction 1",
#             "Description of visible action or interaction 2"
#         ],
#         "environment": "Description of the physical setting or location.",
#         "objects": [
#             "Notable visible object or item 1",
#             "Notable visible object or item 2"
#         ],
#         "background": "Description of background details, lighting, and mood.",
#         "other_details": ""
#     }
# }
# """

# TEXT_ONLY_OCR_PROMPT = """
# NOTE: The image(s) provided may be sequential top-to-bottom slices of a SINGLE tall comic page
# (with slight overlap between slices). If multiple images are given, treat them together as ONE
# continuous page.

# You are a Comic Page TEXT / OCR EXTRACTOR for a multimodal RAG pipeline. Your ONLY job is to
# transcribe every piece of readable text on the page. Do NOT describe artwork, characters, or
# visuals — text only.

# ==================================================
# TEXT ACCURACY RULES
# ==================================================
# - Scan EVERY panel across the entire page from top to bottom (following natural reading order).
# - Transcribe ALL text: speech bubbles, thought bubbles, narration/monologue boxes, captions,
#   sound effects (SFX), signs, titles, credits.
# - Preserve exact wording, casing, and punctuation as visible. Do NOT paraphrase, summarize,
#   correct grammar, or correct spelling.
# - If a word cannot be confidently read, write [unclear]. Never guess an unreadable word from
#   story context.
# - If a page has no text, return empty string "" for full_text and empty list [] for
#   dialogue_and_narration.
# - If a DRAFT OCR transcription is provided below, verify and correct it against the actual image
#   rather than retyping from scratch.

# ==================================================
# OUTPUT FORMAT
# ==================================================
# Return ONLY a valid JSON object matching this exact schema (no markdown fences, no commentary):
# {
#     "text": {
#         "full_text": "Exact concatenated transcription of all readable text on this page in natural reading order.",
#         "dialogue_and_narration": [
#             "Exact transcribed dialogue or narration block 1",
#             "Exact transcribed dialogue or narration block 2"
#         ],
#         "sound_effects": [],
#         "signs_and_labels": []
#     }
# }
# """


# def is_rate_limit_error(exc: Exception) -> bool:
#     """
#     Determines if an exception represents an HTTP 429 Rate Limit error.
#     Checks the Mistral SDK MistralError status_code, underlying response status_code,
#     or common rate limit error messages.
#     """
#     if isinstance(exc, MistralError) and getattr(exc, "status_code", None) == 429:
#         return True
#     if getattr(exc, "status_code", None) == 429:
#         return True
#     raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#     if raw_resp is not None and getattr(raw_resp, "status_code", None) == 429:
#         return True
#     err_str = str(exc).lower()
#     if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
#         return True
#     return False


# def get_retry_after(exc: Exception) -> float | None:
#     """
#     Extracts Retry-After header or provider delay duration from exception details.
#     """
#     headers = getattr(exc, "headers", None)
#     if headers is None:
#         raw_resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
#         if raw_resp is not None:
#             headers = getattr(raw_resp, "headers", None)
#     if headers:
#         retry_val = headers.get("retry-after") or headers.get("Retry-After")
#         if retry_val:
#             try:
#                 val = float(retry_val)
#                 if val > 0:
#                     return val
#             except (ValueError, TypeError):
#                 pass

#     # Extract delay from error message if available (e.g. 'try again in 12.4s')
#     err_str = str(exc)
#     match = re.search(r"(?:try again in|retry after|wait)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", err_str, re.I)
#     if match:
#         try:
#             val = float(match.group(1))
#             if val > 0:
#                 return val
#         except (ValueError, TypeError):
#             pass

#     return None


# # -----------------------------
# # Rate Limit Coordinator (M-25) — Mistral text/OCR calls
# # -----------------------------

# class RateLimitCoordinator:
#     """
#     Centralized rate-limit and recovery coordination across worker threads.
#     - Implements slot reservation to prevent synchronized retry stampedes after 429s.
#     - Staggers recovery dispatches so workers resume gradually rather than in lockstep bursts.
#     - Maintains normal full concurrency of workers when API is healthy.
#     - Validates clearance immediately before dispatch to guarantee no collisions during extended cooldowns.
#     """

#     def __init__(
#         self,
#         target_workers: int = MAX_AI_WORKERS,
#         healthy_interval: float = 0.20,
#         recovery_interval: float = 1.80
#     ):
#         self.target_workers = target_workers
#         self.healthy_interval = healthy_interval
#         self.recovery_interval = recovery_interval

#         self._lock = threading.Lock()
#         self._cooldown_until = 0.0
#         self._recovery_until = 0.0
#         self._next_available_slot = 0.0
#         self._last_dispatch_time = 0.0
#         self._active_requests = 0
#         self._rate_limit_events = 0
#         self._total_cooldown_wait = 0.0

#     def before_request(self, page_number: int, is_retry: bool = False, attempt: int = 0) -> float:
#         """
#         Coordinates and reserves an authorized dispatch slot for a worker.
#         Handles both initial slot reservation and final pre-dispatch clearance,
#         ensuring workers never collide or fire during active global cooldowns.
#         Returns the total wait time in seconds.
#         """
#         wait_start = time.perf_counter()

#         while True:
#             # Phase 1: Slot Reservation
#             with self._lock:
#                 now = time.monotonic()
#                 earliest = max(now, self._cooldown_until)

#                 is_recovering = (now < self._recovery_until) or is_retry
#                 if is_recovering:
#                     # During rate-limit recovery: stagger dispatches smoothly
#                     interval = self.recovery_interval + random.uniform(0.1, 0.4)
#                 else:
#                     # Healthy concurrent throughput: gentle pacing
#                     interval = self.healthy_interval

#                 # Allocate slot at or after next available slot and cooldown
#                 slot = max(earliest, self._next_available_slot)
#                 self._next_available_slot = slot + interval

#             # Sleep outside the lock until our reserved slot arrives
#             delay = slot - time.monotonic()
#             if delay > 0:
#                 time.sleep(delay)

#             # Phase 2: Pre-Dispatch Verification
#             with self._lock:
#                 now = time.monotonic()
#                 # Check if a new 429 occurred while we were sleeping
#                 if now < self._cooldown_until:
#                     # Cooldown was extended by another event while we slept; loop and re-reserve
#                     continue

#                 # Clearance granted: record dispatch and increment active count
#                 self._last_dispatch_time = now
#                 self._active_requests += 1
#                 active = self._active_requests
#                 break

#         total_waited = time.perf_counter() - wait_start
#         if total_waited > 0.05:
#             with self._lock:
#                 self._total_cooldown_wait += total_waited

#         logger.info(
#             "[PERF] AI concurrency | active=%d | target=%d",
#             active,
#             self.target_workers
#         )

#         return total_waited

#     def on_success(self):
#         """Called when an API request completes successfully."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_other_error(self):
#         """Called when an API request fails with a non-rate-limit error."""
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#     def on_rate_limit(
#         self,
#         page_number: int,
#         attempt: int,
#         exc: Exception,
#         max_retries: int = 5
#     ) -> float:
#         """
#         Called when a worker receives an HTTP 429 rate-limit error.
#         Activates global cooldown and recovery mode across all workers.
#         Returns the computed cooldown duration in seconds.
#         """
#         with self._lock:
#             if self._active_requests > 0:
#                 self._active_requests -= 1

#             self._rate_limit_events += 1
#             now = time.monotonic()

#             retry_after = get_retry_after(exc)
#             if retry_after is not None and retry_after > 0:
#                 base_wait = retry_after
#             else:
#                 # Exponential backoff based on attempt with generous base
#                 base_wait = min(3 * (2 ** attempt), 40)

#             jitter = random.uniform(1.0, 3.0)
#             cooldown = base_wait + jitter

#             # Extend global cooldown and recovery window
#             self._cooldown_until = max(self._cooldown_until, now + base_wait)
#             self._recovery_until = max(
#                 self._recovery_until,
#                 self._cooldown_until + (self.recovery_interval * self.target_workers)
#             )

#             # Push next available slot beyond the new cooldown
#             self._next_available_slot = max(self._next_available_slot, self._cooldown_until)

#         logger.warning(
#             "[PERF] Rate limit detected | page=%s | attempt=%d | cooldown=%.2fs",
#             page_number,
#             attempt + 1,
#             cooldown
#         )

#         return cooldown

#     @property
#     def rate_limit_events(self) -> int:
#         with self._lock:
#             return self._rate_limit_events

#     @property
#     def total_cooldown_wait(self) -> float:
#         with self._lock:
#             return self._total_cooldown_wait


# # -----------------------------
# # Gemini Rate Limiter — visual analysis calls
# # -----------------------------
# # Free-tier Gemini RPM is very low (5-15 depending on model). Running
# # MAX_AI_WORKERS threads in parallel against it WILL blow through the limit
# # instantly. This limiter does two things:
# #   1. Forces ALL Gemini calls (from every worker thread) through a single
# #      shared gate, spaced (60 / GEMINI_MAX_RPM) seconds apart. Concurrency
# #      doesn't help at 5-15 RPM anyway — it just causes bursts + 429s.
# #   2. On a 429/RESOURCE_EXHAUSTED response, backs off exponentially with
# #      jitter and pushes the shared cooldown forward for every other
# #      waiting worker too, so they don't all retry at once.
# # This is completely independent from RateLimitCoordinator above (which
# # still paces the Mistral text/OCR calls exactly as before).

# class GeminiRateLimiter:
#     def __init__(self, max_rpm: float = GEMINI_MAX_RPM):
#         self.min_interval = 60.0 / max_rpm
#         self._lock = threading.Lock()
#         self._next_allowed_time = 0.0

#     def wait_for_slot(self):
#         with self._lock:
#             now = time.monotonic()
#             slot = max(now, self._next_allowed_time)
#             self._next_allowed_time = slot + self.min_interval
#         delay = slot - time.monotonic()
#         if delay > 0:
#             time.sleep(delay)

#     def push_back(self, extra_wait: float):
#         """Call after a 429 — extends the shared cooldown for everyone waiting."""
#         with self._lock:
#             self._next_allowed_time = max(
#                 self._next_allowed_time,
#                 time.monotonic() + extra_wait
#             )


# _gemini_limiter = GeminiRateLimiter()
# _gemini_client = None
# _gemini_client_lock = threading.Lock()


# def _get_gemini_client():
#     """Lazy-init the Gemini client (thread-safe singleton)."""
#     global _gemini_client
#     if _gemini_client is None:
#         with _gemini_client_lock:
#             if _gemini_client is None:
#                 from google import genai
#                 _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
#     return _gemini_client


# def _is_gemini_rate_limit_error(exc: Exception) -> bool:
#     err_str = str(exc).lower()
#     return (
#         "429" in err_str
#         or "resource_exhausted" in err_str
#         or "rate limit" in err_str
#         or "quota" in err_str
#     )


# def _get_gemini_retry_delay(exc: Exception) -> float | None:
#     """Gemini sometimes reports a structured retry delay in the error details."""
#     err_str = str(exc)
#     match = re.search(r"retry.{0,20}?([0-9]+(?:\.[0-9]+)?)\s*s", err_str, re.I)
#     if match:
#         try:
#             val = float(match.group(1))
#             if val > 0:
#                 return val
#         except (ValueError, TypeError):
#             pass
#     return None


# def _call_gemini_with_backoff(fn, max_retries: int = 5):
#     """
#     Wraps a Gemini API call with:
#     - rate-limiter slot wait BEFORE every attempt (including retries)
#     - exponential backoff + jitter on 429/quota errors
#     - honors any retry-delay the API itself reports
#     """
#     for attempt in range(max_retries + 1):
#         _gemini_limiter.wait_for_slot()
#         try:
#             return fn()
#         except Exception as e:
#             if _is_gemini_rate_limit_error(e) and attempt < max_retries:
#                 reported_delay = _get_gemini_retry_delay(e)
#                 if reported_delay is not None:
#                     wait_time = reported_delay + random.uniform(0.5, 1.5)
#                 else:
#                     wait_time = min(4 * (2 ** attempt), 60) + random.uniform(0.5, 1.5)
#                 logger.warning(
#                     "[GEMINI RATE LIMIT] attempt %d/%d — backing off %.1fs",
#                     attempt + 1, max_retries + 1, wait_time
#                 )
#                 _gemini_limiter.push_back(wait_time)
#                 time.sleep(wait_time)
#             else:
#                 raise


# def _parse_ai_response_json(raw_result: str) -> dict:
#     """
#     Robust JSON parser for AI page analysis responses.
#     Handles strict decoding, markdown code fences, embedded JSON objects, and unescaped newlines.
#     """
#     # 1. Direct parse with strict=False (allows unescaped control chars/newlines in strings)
#     try:
#         return json.loads(raw_result, strict=False)
#     except (json.JSONDecodeError, TypeError):
#         pass

#     # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
#     code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
#     if code_match:
#         try:
#             return json.loads(code_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 3. Extract outermost JSON object { ... }
#     brace_match = re.search(r"(\{.*\})", raw_result, re.DOTALL)
#     if brace_match:
#         try:
#             return json.loads(brace_match.group(1).strip(), strict=False)
#         except (json.JSONDecodeError, TypeError):
#             pass

#     # 4. Fallback cleanup: remove leading/trailing ```
#     cleaned = raw_result.strip()
#     if cleaned.startswith("```json"):
#         cleaned = cleaned[7:]
#     elif cleaned.startswith("```"):
#         cleaned = cleaned[3:]
#     if cleaned.endswith("```"):
#         cleaned = cleaned[:-3]
#     return json.loads(cleaned.strip(), strict=False)


# def _encode_and_scale_image(image_path: str, max_dimension: int = 1600, quality: int = 85) -> str:
#     """
#     Loads an image, resizes it so its longest dimension does not exceed max_dimension
#     (using LANCZOS resampling to preserve sharp text), and encodes to base64 JPEG (quality=85).
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         orig_w, orig_h = img.size
#         longest = max(orig_w, orig_h)

#         if longest > max_dimension:
#             scale = max_dimension / float(longest)
#             new_w = max(1, int(orig_w * scale))
#             new_h = max(1, int(orig_h * scale))
#             img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         buffer = io.BytesIO()
#         img.save(buffer, format="JPEG", quality=quality, optimize=True)
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")


# def _encode_and_scale_image_tiles(image_path: str, max_dimension: int = 1568, quality: int = 85) -> list[str]:
#     """
#     Splits a tall/wide comic page into near-square, overlapping tiles
#     (per Mistral's 1:1 aspect ratio recommendation for vision models) and
#     returns a list of base64-encoded JPEG tiles, ordered top-to-bottom.

#     Comic pages are typically very tall/portrait (aspect ratio far from 1:1).
#     Sending the whole page as one elongated image degrades vision model
#     accuracy and causes hallucinated/blended panel content. Slicing into
#     near-square, slightly-overlapping tiles keeps each slice close to the
#     recommended aspect ratio while preserving continuity via the overlap.
#     """
#     with Image.open(image_path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")

#         w, h = img.size
#         aspect = h / w

#         # If already close to square, just encode as one tile (no slicing needed)
#         if aspect <= 1.3:
#             return [_encode_and_scale_image(image_path, max_dimension, quality)]

#         # Otherwise slice into near-square tiles with a small overlap for continuity
#         tile_height = w
#         overlap = int(tile_height * 0.15)
#         tiles = []
#         y = 0
#         while y < h:
#             box = (0, y, w, min(y + tile_height, h))
#             tile = img.crop(box)
#             buffer = io.BytesIO()

#             # Scale tile so its longest side <= max_dimension
#             tw, th = tile.size
#             longest = max(tw, th)
#             if longest > max_dimension:
#                 scale = max_dimension / float(longest)
#                 tile = tile.resize(
#                     (max(1, int(tw * scale)), max(1, int(th * scale))),
#                     Image.Resampling.LANCZOS
#                 )

#             tile.save(buffer, format="JPEG", quality=quality, optimize=True)
#             tiles.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

#             if y + tile_height >= h:
#                 break
#             y += tile_height - overlap

#         return tiles


# _easyocr_reader = None
# _easyocr_lock = threading.Lock()


# def get_easyocr_reader():
#     """Lazy-loads EasyOCR Reader instance thread-safely."""
#     global _easyocr_reader
#     if _easyocr_reader is None:
#         with _easyocr_lock:
#             if _easyocr_reader is None:
#                 try:
#                     import easyocr
#                     _easyocr_reader = easyocr.Reader(["en"], gpu=False)
#                 except Exception as e:
#                     logger.warning("EasyOCR initialization error: %s", e)
#                     _easyocr_reader = False
#     return _easyocr_reader if _easyocr_reader is not False else None


# def _extract_draft_ocr(image_path: str, max_dim: int = 1000) -> str:
#     """
#     Extracts fast preliminary draft OCR text using EasyOCR to assist Vision LLM.
#     """
#     reader = get_easyocr_reader()
#     if not reader:
#         return ""
#     try:
#         with Image.open(image_path) as img:
#             if img.mode not in ("RGB", "L"):
#                 img = img.convert("RGB")
#             w, h = img.size
#             if max(w, h) > max_dim:
#                 scale = max_dim / float(max(w, h))
#                 img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
#             buffer = io.BytesIO()
#             img.save(buffer, format="JPEG", quality=75)
#             buffer.seek(0)
#             results = reader.readtext(buffer.getvalue(), detail=0)
#             return "\n".join(results).strip()
#     except Exception as e:
#         logger.debug("Draft OCR error on %s: %s", image_path, e)
#         return ""


# # -----------------------------
# # Visual Analysis — Gemini (rate-limited)
# # -----------------------------

# def analyze_page_visual_gemini(image_tiles_base64: list[str]) -> dict:
#     """
#     Pure visual description via Gemini. Never sees any dialogue/narration text,
#     so it cannot invent visuals to match narration mood. Rate-limited so any
#     number of worker threads funnel through one shared, spaced-out gate.
#     """
#     from google.genai import types

#     if not GEMINI_API_KEY:
#         raise RuntimeError("GEMINI_API_KEY is not configured.")

#     gclient = _get_gemini_client()

#     parts = []
#     for tile_b64 in image_tiles_base64:
#         parts.append(
#             types.Part.from_bytes(data=base64.b64decode(tile_b64), mime_type="image/jpeg")
#         )
#     parts.append(VISUAL_ONLY_PROMPT)

#     def _do_call():
#         return gclient.models.generate_content(
#             model=GEMINI_MODEL,
#             contents=parts,
#             config=types.GenerateContentConfig(
#                 temperature=0.1,
#                 response_mime_type="application/json",
#             ),
#         )

#     response = _call_gemini_with_backoff(_do_call)
#     return _parse_ai_response_json(response.text)


# # -----------------------------
# # Text/OCR Analysis — Mistral
# # -----------------------------

# def analyze_page_text_mistral(image_tiles_base64: list[str], draft_ocr: str = "") -> dict:
#     """Pure OCR/text transcription via Mistral — never asked about visuals."""
#     prompt_text = TEXT_ONLY_OCR_PROMPT
#     if draft_ocr:
#         prompt_text += f'\n\nDRAFT OCR (verify against image):\n"""\n{draft_ocr}\n"""\n'

#     content_blocks = [{"type": "text", "text": prompt_text}]
#     for tile_b64 in image_tiles_base64:
#         content_blocks.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{tile_b64}"
#             }
#         })

#     response = client.chat.complete(
#         model=MODEL_NAME,
#         messages=[
#             {
#                 "role": "user",
#                 "content": content_blocks
#             }
#         ],
#         temperature=0.1,
#     )

#     raw_result = response.choices[0].message.content
#     return _parse_ai_response_json(raw_result)


# # -----------------------------
# # Single Page Analysis (merged)
# # -----------------------------

# def analyze_page(image_path: str, draft_ocr: str = "") -> dict:
#     """
#     Full page analysis = tiling + 2 independent calls merged:
#       - VISUAL description -> Gemini (rate-limited, see GeminiRateLimiter)
#       - TEXT/OCR transcription -> Mistral (unchanged, already accurate)
#     Keeping them in separate calls/providers prevents text from leaking into
#     (and contaminating) the visual description, and lets us use whichever
#     provider is actually strong at each sub-task.
#     """
#     image_tiles_base64 = _encode_and_scale_image_tiles(
#         image_path,
#         max_dimension=1568,
#         quality=85
#     )

#     if ENABLE_HYBRID_OCR and not draft_ocr:
#         draft_ocr = _extract_draft_ocr(image_path)

#     visual_json = analyze_page_visual_gemini(image_tiles_base64)
#     text_json = analyze_page_text_mistral(image_tiles_base64, draft_ocr=draft_ocr)

#     return {
#         "page_summary": visual_json.get("page_summary", ""),
#         "panels_detected": visual_json.get("panels_detected", 1),
#         "panels": visual_json.get("panels", []),
#         "text": text_json.get("text", {}),
#         "visual_description": visual_json.get("visual_description", {}),
#     }

# # -----------------------------
# # Build Comic-Level Full Text
# # -----------------------------

# def build_comic_full_text(results: list) -> str:
#     """
#     Page-level OCR text ko page order mein combine karta hai.

#     Important:
#     - AI ko dobara entire comic dene ki zaroorat nahi.
#     - Existing page OCR ko hi combine kiya jata hai.
#     - Page number preserve hota hai.
#     """

#     full_text_parts = []

#     ordered_results = sorted(
#         results,
#         key=lambda x: x["page_number"]
#     )

#     for page in ordered_results:

#         if page.get("status") != "success":
#             continue

#         analysis = page.get("analysis") or {}

#         text_data = analysis.get(
#             "text",
#             {}
#         )

#         page_text = text_data.get(
#             "full_text",
#             ""
#         )

#         if not isinstance(page_text, str):
#             continue

#         page_text = page_text.strip()

#         if not page_text:
#             continue

#         full_text_parts.append(
#             f"PAGE {page['page_number']}\n"
#             f"{page_text}"
#         )

#     return "\n\n".join(
#         full_text_parts
#     )


# def analyze_pages(
#     pages: list,
#     max_workers: int = MAX_AI_WORKERS,
#     max_retries: int = MAX_AI_RETRIES,
#     on_page_complete: Optional[Callable[[dict, int, int], None]] = None
# ) -> list:

#     coordinator = RateLimitCoordinator(target_workers=max_workers)

#     def process_page(page):

#         page_number = page["page_number"]
#         image_path = page["image_path"]
#         page_start = time.perf_counter()
#         attempts = 0
#         retry_wait_total = 0.0

#         if not Path(image_path).exists():
#             page_duration = time.perf_counter() - page_start
#             logger.warning("[AI] Page image %s not found on local disk.", image_path)
#             return {
#                 "page_number": page_number,
#                 "filename": page.get("filename", Path(image_path).name),
#                 "image_path": image_path,
#                 "analysis": None,
#                 "metadata": {
#                     "page_number": page_number
#                 },
#                 "status": "error",
#                 "error": f"Image file not found: {image_path}",
#                 "perf": {
#                     "duration": page_duration,
#                     "attempts": 1,
#                     "retry_wait": 0.0
#                 }
#             }

#         for attempt in range(
#             max_retries + 1
#         ):
#             attempts += 1

#             # Request slot authorization from the coordinator (paces the
#             # Mistral text/OCR call; the Gemini visual call inside
#             # analyze_page() has its own independent GeminiRateLimiter gate)
#             wait_time = coordinator.before_request(
#                 page_number=page_number,
#                 is_retry=(attempt > 0),
#                 attempt=attempt
#             )
#             retry_wait_total += wait_time

#             try:

#                 result = analyze_page(
#                     image_path
#                 )

#                 coordinator.on_success()

#                 page_duration = time.perf_counter() - page_start

#                 logger.info(
#                     "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs",
#                     page_number,
#                     page_duration,
#                     attempts,
#                     retry_wait_total
#                 )

#                 return {
#                     "page_number": page_number,
#                     "filename": page.get("filename", Path(image_path).name),
#                     "image_path": image_path,
#                     "analysis": result,
#                     "metadata": {
#                         "page_number": page_number,
#                         "has_text": bool(
#                             (
#                                 result.get("text", {})
#                                 .get("full_text", "")
#                                 if isinstance(
#                                     result.get("text", {}),
#                                     dict
#                                 )
#                                 else ""
#                             ).strip()
#                         )
#                     },
#                     "status": "success",
#                     "perf": {
#                         "duration": page_duration,
#                         "attempts": attempts,
#                         "retry_wait": retry_wait_total
#                     }
#                 }

#             except Exception as e:

#                 error_message = str(e)

#                 if is_rate_limit_error(e):

#                     if attempt >= max_retries:
#                         coordinator.on_other_error()
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED RATE LIMIT)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#                     # Register rate limit with coordinator and sleep with exponential backoff + jitter
#                     cooldown = coordinator.on_rate_limit(
#                         page_number=page_number,
#                         attempt=attempt,
#                         exc=e,
#                         max_retries=max_retries
#                     )
#                     time.sleep(cooldown)
#                     retry_wait_total += cooldown

#                 else:
#                     # Non-rate-limit error (e.g. transient network or JSON format error)
#                     coordinator.on_other_error()

#                     if attempt < max_retries:
#                         logger.warning(
#                             "Page %s encountered error (%s). Retrying in 1.5s (attempt %s/%s)...",
#                             page_number,
#                             error_message,
#                             attempt + 1,
#                             max_retries
#                         )
#                         retry_start = time.perf_counter()
#                         time.sleep(1.5 + random.uniform(0.1, 0.5))
#                         retry_wait_total += time.perf_counter() - retry_start
#                     else:
#                         page_duration = time.perf_counter() - page_start
#                         logger.error(
#                             "[PERF] Page %s | duration=%.2fs | attempts=%s | retry_wait=%.2fs (FAILED ERROR)",
#                             page_number,
#                             page_duration,
#                             attempts,
#                             retry_wait_total
#                         )

#                         return {
#                             "page_number": page_number,
#                             "filename": page.get("filename", Path(image_path).name),
#                             "image_path": image_path,
#                             "analysis": None,
#                             "metadata": {
#                                 "page_number": page_number
#                             },
#                             "status": "error",
#                             "error": error_message,
#                             "perf": {
#                                 "duration": page_duration,
#                                 "attempts": attempts,
#                                 "retry_wait": retry_wait_total
#                             }
#                         }

#     results = []
#     total_pages = len(pages)
#     ai_start = time.perf_counter()

#     # -----------------------------
#     # Parallel Processing
#     # -----------------------------

#     with ThreadPoolExecutor(
#         max_workers=max_workers
#     ) as executor:

#         futures = {
#             executor.submit(
#                 process_page,
#                 page
#             ): page["page_number"]

#             for page in pages
#         }

#         completed = 0

#         for future in as_completed(
#             futures
#         ):

#             result = future.result()

#             results.append(
#                 result
#             )

#             completed += 1

#             if result["status"] == "success":
#                 logger.info(
#                     "[%s/%s] Page %s analyzed successfully",
#                     completed,
#                     total_pages,
#                     result["page_number"]
#                 )
#             else:
#                 logger.error(
#                     "[%s/%s] Page %s analysis failed: %s",
#                     completed,
#                     total_pages,
#                     result["page_number"],
#                     result.get("error", "Unknown error")
#                 )

#             if on_page_complete is not None:
#                 try:
#                     on_page_complete(result, completed, total_pages)
#                 except Exception as cb_err:
#                     logger.warning("on_page_complete callback error: %s", str(cb_err))

#     total_ai_duration = time.perf_counter() - ai_start

#     # -----------------------------
#     # Performance Aggregate Stats
#     # -----------------------------
#     successful = sum(1 for r in results if r.get("status") == "success")
#     failed = sum(1 for r in results if r.get("status") == "error")
#     perf_list = [r.get("perf", {}) for r in results]
#     pages_with_retries = sum(1 for p in perf_list if p.get("attempts", 1) > 1)
#     total_retries = sum(max(0, p.get("attempts", 1) - 1) for p in perf_list)
#     total_retry_wait = sum(p.get("retry_wait", 0.0) for p in perf_list)
#     durations = [p.get("duration", 0.0) for p in perf_list]
#     avg_page_duration = (sum(durations) / len(durations)) if durations else 0.0
#     fastest_page = min(durations) if durations else 0.0
#     slowest_page = max(durations) if durations else 0.0

#     no_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) == 1]
#     avg_no_retry = (sum(no_retry_durations) / len(no_retry_durations)) if no_retry_durations else 0.0
#     with_retry_durations = [p.get("duration", 0.0) for p in perf_list if p.get("attempts", 1) > 1]
#     avg_with_retry = (sum(with_retry_durations) / len(with_retry_durations)) if with_retry_durations else 0.0

#     logger.info(
#         "\n"
#         "[PERF] ==================================================\n"
#         "[PERF] AI ANALYSIS PERFORMANCE\n"
#         "[PERF] Pages: %s\n"
#         "[PERF] Workers: %s\n"
#         "[PERF] Total duration: %.2fs\n"
#         "[PERF] Successful: %s\n"
#         "[PERF] Failed: %s\n"
#         "[PERF] Pages with retries: %s\n"
#         "[PERF] Total retries: %s\n"
#         "[PERF] Total retry wait: %.2fs\n"
#         "[PERF] Rate-limit events: %s\n"
#         "[PERF] Total provider cooldown time: %.2fs\n"
#         "[PERF] Average page duration: %.2fs\n"
#         "[PERF]   - No-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF]   - With-retry pages avg: %.2fs (%s pages)\n"
#         "[PERF] Fastest page: %.2fs\n"
#         "[PERF] Slowest page: %.2fs\n"
#         "[PERF] ==================================================",
#         total_pages,
#         max_workers,
#         total_ai_duration,
#         successful,
#         failed,
#         pages_with_retries,
#         total_retries,
#         total_retry_wait,
#         coordinator.rate_limit_events,
#         coordinator.total_cooldown_wait,
#         avg_page_duration,
#         avg_no_retry,
#         len(no_retry_durations),
#         avg_with_retry,
#         len(with_retry_durations),
#         fastest_page,
#         slowest_page
#     )

#     # -----------------------------
#     # Restore Page Order
#     # -----------------------------

#     results.sort(
#         key=lambda x: x["page_number"]
#     )

#     return results

    # EDEN_KEY_1 = os.getenv("EDEN_KEY_1", "sk-eden-live-nlQJ9hMh0QdIsis1wCn2_hUzri4BFkWFnCw2U2RheNI1491a31a")
    # EDEN_KEY_2 = os.getenv("EDEN_KEY_2", "sk-eden-live-jI8otyOpgJuOzGWXW5F2Ga4qPFgVn7fVlmFk_uIr6IE60e2f9f2")
    # EDEN_KEY_3 = os.getenv("EDEN_KEY_3", "sk-eden-live-DlDN3RUAsx4JVxXXbWRIy3rfEcM65eJIP17FN02MO7Aeb792aa1")
    # EDEN_KEY_4 = os.getenv("EDEN_KEY_4", "sk-eden-live-U4UwbCWbkbZ0NIN7BV3FS8LAjhiuq8Q2qLYbxBok-44ec03f19b")



import base64
import io
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image


from app.core.config import MAX_AI_RETRIES, MAX_AI_WORKERS
from app.core.logging import logger

# -----------------------------
# EdenAI configuration
# -----------------------------
import os

EDEN_API_URL = os.getenv(
    "EDEN_API_URL",
    "https://api.edenai.run/v3/chat/completions",
)
EDEN_MODEL = os.getenv(
    "EDEN_MODEL",
    "google/gemma-4-26b-a4b-it",
)
EDEN_MAX_RETRIES = int(os.getenv("EDEN_MAX_RETRIES", "4"))
EDEN_TIMEOUT = float(os.getenv("EDEN_TIMEOUT", "200"))

EDEN_KEY_1 = os.getenv("EDEN_KEY_1", "sk-eden-live-nlQJ9hMh0QdIsis1wCn2_hUzri4BFkWFnCw2U2RheNI1491a31a")
EDEN_KEY_2 = os.getenv("EDEN_KEY_2", "sk-eden-live-jI8otyOpgJuOzGWXW5F2Ga4qPFgVn7fVlmFk_uIr6IE60e2f9f2")
EDEN_KEY_3 = os.getenv("EDEN_KEY_3", "sk-eden-live-DlDN3RUAsx4JVxXXbWRIy3rfEcM65eJIP17FN02MO7Aeb792aa1")
EDEN_KEY_4 = os.getenv("EDEN_KEY_4", "sk-eden-live-U4UwbCWbkbZ0NIN7BV3FS8LAjhiuq8Q2qLYbxBok-44ec03f19b")

EDEN_COMIC_PROMPT = """
You are a universal Comic Page Analyzer and Extraction Engine for a multimodal RAG pipeline.
Your task is to accurately extract all text and describe the visual contents of the provided page image.

==================================================
0. Follow The rule
==================================================
CRITICAL BORDER AWARENESS: Count every single fully enclosed frame as a separate panel. If a black or white dividing line (gutter) separates two scenes side-by-side, they MUST be counted as TWO distinct panels, even if they show the exact same characters in the exact same room. DO NOT MERGE adjacent panels.

==================================================
1. COMPLETE TEXT TRANSCRIPTION & OCR (HIGHEST PRIORITY)
==================================================
- Extract ALL text visible on the page (dialogue, narration, sound effects, signs).
- Ensure absolute accuracy in spelling, punctuation, and reading order.

==================================================
2. PRECISE & GROUNDED VISUAL EXTRACTION (PANEL-SCOPED)
==================================================
- Identify each distinct panel on the page separately (a full-page splash counts as ONE panel).
- CRITICAL PANEL RULE: Count visual image frames as panels, NOT text boxes. A single image panel may contain multiple dialogue or narration boxes. Do NOT split one continuous image into multiple panels just because it has multiple text bubbles.
- STRICT CATEGORIZATION: The "characters" array MUST ONLY contain living beings (people, animals, creatures). Do NOT put inanimate objects, landscapes, or background details (like wagons, mountains, skies) in the characters array. Map those correctly to "objects" or "environment".
- For EACH panel, describe ONLY the figures/objects/actions physically visible WITHIN that panel's borders.
- NEVER borrow, merge, or transfer a detail (emotion, pose, clothing, identity) from one panel into your description of a different panel, even if the figures look similar.
- If the SAME figure (by consistent visual traits: hair, clothing, size) appears in multiple panels, describe it separately in each panel's context — do NOT assume it is a different, new character just because the panel changed, AND do NOT assume it is the same character unless traits genuinely match.
- Do NOT invent a figure, pose, or emotional expression (e.g. "distressed", "looking up") unless it is unambiguously visible in that specific panel. If uncertain, omit the detail rather than guessing.

==================================================
3. OUTPUT FORMAT
==================================================
Return ONLY a valid JSON object matching this exact schema:
{
    "page_summary": "...",
    "panels_detected": 0,
    "panels": [
        {
            "panel_index": 1,
            "dialogue_and_narration": ["..."],
            "characters": ["Description of visible figure 1 in THIS panel only", "..."],
            "actions": ["..."],
            "environment": "...",
            "objects": ["..."]
        }
    ],
    "text": {
        "full_text": "...",
        "dialogue_and_narration": [...],
        "sound_effects": [],
        "signs_and_labels": []
    },
    "visual_description": {
        "characters": [...],
        "actions": [...],
        "environment": "...",
        "objects": [...],
        "background": "...",
        "other_details": ""
    }
}
"""


# -----------------------------
# EdenAI helpers
# -----------------------------

_eden_keys = tuple(
    key for key in (EDEN_KEY_1, EDEN_KEY_2, EDEN_KEY_3, EDEN_KEY_4)
    if key
)
_eden_key_lock = threading.Lock()
_eden_key_index = 0
_eden_key_semaphores = {
    key: threading.Semaphore(2) for key in _eden_keys
}


def _get_next_eden_key() -> str:
    global _eden_key_index

    if not _eden_keys:
        raise RuntimeError(
            "No EdenAI API key configured. "
            "Set EDEN_KEY_1 and optionally EDEN_KEY_2..EDEN_KEY_4."
        )

    with _eden_key_lock:
        key = _eden_keys[_eden_key_index % len(_eden_keys)]
        _eden_key_index += 1
        return key


def _normalize_page_analysis(data: Any) -> dict:
    """
    Ensure the analysis result is ALWAYS a valid dictionary conforming to the comic schema.
    Handles:
    - Dictionary directly (standard case)
    - List of 1 dictionary: [ { "page_summary": ... } ]
    - List of panel dictionaries: [ { "panel_index": 1, ... }, ... ]
    - List of text/dialogue items
    - Non-dict or malformed items
    """
    if isinstance(data, list):
        if not data:
            return {
                "page_summary": "No visual content detected on this page.",
                "panels_detected": 0,
                "panels": [],
                "text": {"full_text": "", "dialogue_and_narration": [], "sound_effects": [], "signs_and_labels": []},
                "visual_description": {"characters": [], "actions": [], "environment": "", "objects": [], "background": "", "other_details": ""}
            }

        first = data[0]
        if isinstance(first, dict):
            if "page_summary" in first or "text" in first or "visual_description" in first:
                data = first
            else:
                # The model returned a list of panels: [ { "panel_index": 1, ... }, ... ]
                panels = []
                all_text_parts = []
                dialogues = []
                all_characters = []
                all_actions = []
                all_objects = []

                for idx, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        p_idx = item.get("panel_index", idx)
                        p_dial = item.get("dialogue_and_narration") or item.get("text") or []
                        if isinstance(p_dial, str):
                            p_dial = [p_dial] if p_dial.strip() else []
                        elif not isinstance(p_dial, list):
                            p_dial = []

                        for d in p_dial:
                            if isinstance(d, str) and d.strip():
                                all_text_parts.append(d.strip())
                                dialogues.append(d.strip())

                        chars = item.get("characters") or []
                        if isinstance(chars, str):
                            chars = [chars] if chars.strip() else []
                        elif isinstance(chars, list):
                            all_characters.extend([str(c) for c in chars if c])

                        acts = item.get("actions") or []
                        if isinstance(acts, str):
                            acts = [acts] if acts.strip() else []
                        elif isinstance(acts, list):
                            all_actions.extend([str(a) for a in acts if a])

                        objs = item.get("objects") or []
                        if isinstance(objs, str):
                            objs = [objs] if objs.strip() else []
                        elif isinstance(objs, list):
                            all_objects.extend([str(o) for o in objs if o])

                        panels.append({
                            "panel_index": p_idx,
                            "dialogue_and_narration": p_dial,
                            "characters": chars if isinstance(chars, list) else [],
                            "actions": acts if isinstance(acts, list) else [],
                            "environment": str(item.get("environment") or ""),
                            "objects": objs if isinstance(objs, list) else []
                        })

                full_text = "\n".join(all_text_parts)
                data = {
                    "page_summary": f"Comic page containing {len(panels)} panels.",
                    "panels_detected": len(panels),
                    "panels": panels,
                    "text": {
                        "full_text": full_text,
                        "dialogue_and_narration": dialogues,
                        "sound_effects": [],
                        "signs_and_labels": []
                    },
                    "visual_description": {
                        "characters": list(dict.fromkeys(all_characters)),
                        "actions": list(dict.fromkeys(all_actions)),
                        "environment": "",
                        "objects": list(dict.fromkeys(all_objects)),
                        "background": "",
                        "other_details": ""
                    }
                }
        else:
            text_combined = "\n".join(str(x) for x in data if x)
            data = {
                "page_summary": text_combined[:300] if text_combined else "Comic page.",
                "panels_detected": 1,
                "panels": [],
                "text": {"full_text": text_combined, "dialogue_and_narration": [text_combined] if text_combined else [], "sound_effects": [], "signs_and_labels": []},
                "visual_description": {"characters": [], "actions": [], "environment": "", "objects": [], "background": "", "other_details": ""}
            }

    if not isinstance(data, dict):
        data = {
            "page_summary": str(data) if data else "",
            "panels_detected": 0,
            "panels": [],
            "text": {"full_text": "", "dialogue_and_narration": [], "sound_effects": [], "signs_and_labels": []},
            "visual_description": {"characters": [], "actions": [], "environment": "", "objects": [], "background": "", "other_details": ""}
        }

    # Ensure required sub-keys exist and are of proper type
    if "text" not in data or not isinstance(data["text"], dict):
        data["text"] = {
            "full_text": "",
            "dialogue_and_narration": [],
            "sound_effects": [],
            "signs_and_labels": []
        }
    else:
        if "full_text" not in data["text"] or not isinstance(data["text"]["full_text"], str):
            dial = data["text"].get("dialogue_and_narration")
            if isinstance(dial, list):
                data["text"]["full_text"] = "\n".join(str(d) for d in dial if d)
            elif isinstance(dial, str):
                data["text"]["full_text"] = dial
            else:
                data["text"]["full_text"] = ""

    if "visual_description" not in data or not isinstance(data["visual_description"], dict):
        data["visual_description"] = {
            "characters": [],
            "actions": [],
            "environment": "",
            "objects": [],
            "background": "",
            "other_details": ""
        }

    if "panels" not in data or not isinstance(data["panels"], list):
        data["panels"] = []

    if "panels_detected" not in data:
        data["panels_detected"] = len(data["panels"])

    if "page_summary" not in data or not isinstance(data["page_summary"], str):
        data["page_summary"] = ""

    return data


def _parse_ai_response_json(raw_result: Any) -> dict:
    """Parse EdenAI JSON, including fenced JSON responses and lists, returning a guaranteed valid dict."""
    if isinstance(raw_result, (dict, list)):
        return _normalize_page_analysis(raw_result)

    if not isinstance(raw_result, str):
        raw_result = str(raw_result or "")

    raw_result = raw_result.strip()

    try:
        parsed = json.loads(raw_result, strict=False)
        return _normalize_page_analysis(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    code_match = re.search(
        r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```",
        raw_result,
        re.DOTALL,
    )
    if code_match:
        try:
            parsed = json.loads(code_match.group(1).strip(), strict=False)
            return _normalize_page_analysis(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    brace_match = re.search(r"(\{.*\})", raw_result, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(1).strip(), strict=False)
            return _normalize_page_analysis(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    bracket_match = re.search(r"(\[.*\])", raw_result, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(1).strip(), strict=False)
            return _normalize_page_analysis(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    if raw_result:
        return _normalize_page_analysis({
            "page_summary": raw_result[:300],
            "panels_detected": 1,
            "panels": [],
            "text": {
                "full_text": raw_result,
                "dialogue_and_narration": [raw_result],
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
        })

    raise ValueError("EdenAI returned invalid JSON.")


def _eden_request(image_tiles_base64: list[str]) -> dict:
    """Send one EdenAI request for complete OCR + visual page analysis."""
    key = _get_next_eden_key()
    semaphore = _eden_key_semaphores[key]

    content = [
        {
            "type": "text",
            "text": EDEN_COMIC_PROMPT,
        }
    ]

    for tile_b64 in image_tiles_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{tile_b64}"
                },
            }
        )

    payload = {
        "model": EDEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(EDEN_MAX_RETRIES):
        try:
            with semaphore:
                request = urllib.request.Request(
                    EDEN_API_URL,
                    data=body,
                    headers=headers,
                    method="POST",
                )

                with urllib.request.urlopen(
                    request,
                    timeout=EDEN_TIMEOUT,
                ) as response:
                    raw_data = json.loads(
                        response.read().decode("utf-8")
                    )

            message_content = raw_data["choices"][0]["message"]["content"]

            if isinstance(message_content, list):
                parts = []
                for item in message_content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", "") or item.get("content", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                    else:
                        parts.append(str(item))
                message_content = "".join(parts)

            return _parse_ai_response_json(message_content)

        except urllib.error.HTTPError as exc:
            last_error = exc

            if (
                exc.code not in (429, 500, 502, 503, 504)
                or attempt >= EDEN_MAX_RETRIES - 1
            ):
                raise RuntimeError(
                    f"EdenAI HTTP {exc.code}: {exc.reason}"
                ) from exc

            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_time = max(1.0, float(retry_after))
                except (ValueError, TypeError):
                    wait_time = 5.0
            else:
                wait_time = min(5.0 * (2 ** attempt), 60.0)

            logger.warning(
                "[EDEN] HTTP %s; retrying in %.1fs "
                "(attempt %d/%d)",
                exc.code,
                wait_time,
                attempt + 1,
                EDEN_MAX_RETRIES,
            )
            time.sleep(wait_time)

        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc

            if attempt >= EDEN_MAX_RETRIES - 1:
                raise

            wait_time = min(4.0 * (2 ** attempt) + random.uniform(0.5, 2.0), 30.0)
            logger.warning(
                "[EDEN] transient/parse error: %s; "
                "retrying in %.1fs (attempt %d/%d)",
                exc,
                wait_time,
                attempt + 1,
                EDEN_MAX_RETRIES,
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"EdenAI failed after {EDEN_MAX_RETRIES} attempts: {last_error}"
    )


# -----------------------------
# Image encoding
# -----------------------------

def _encode_and_scale_image(
    image_path: str,
    max_dimension: int = 1600,
    quality: int = 85,
) -> str:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        longest = max(orig_w, orig_h)

        if longest > max_dimension:
            scale = max_dimension / float(longest)
            img = img.resize(
                (
                    max(1, int(orig_w * scale)),
                    max(1, int(orig_h * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        img.save(
            buffer,
            format="JPEG",
            quality=quality,
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


# -----------------------------
# Single Page Analysis
# -----------------------------

def analyze_page(image_path: str, draft_ocr: str = "") -> dict:
    """
    Complete page analysis through EdenAI only.

    EdenAI receives the exact universal comic-analysis prompt and
    performs OCR, panel detection, text extraction, and visual
    extraction in one request.
    """
    image_b64 = _encode_and_scale_image(
        image_path,
        max_dimension=1600,
        quality=85,
    )

    result = _eden_request([image_b64])
    return _normalize_page_analysis(result)


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
        key=lambda x: x.get("page_number", 0) if isinstance(x, dict) else 0
    )

    for page in ordered_results:
        if not isinstance(page, dict) or page.get("status") != "success":
            continue

        raw_analysis = page.get("analysis")
        analysis = _normalize_page_analysis(raw_analysis) if raw_analysis else {}

        text_data = analysis.get("text", {})
        if not isinstance(text_data, dict):
            text_data = {}

        page_text = text_data.get("full_text", "")
        if not isinstance(page_text, str):
            page_text = str(page_text or "")

        page_text = page_text.strip()
        if not page_text:
            continue

        p_num = page.get("page_number", 0)
        full_text_parts.append(
            f"PAGE {p_num}\n"
            f"{page_text}"
        )

    return "\n\n".join(full_text_parts)


def analyze_pages(
    pages: list,
    max_workers: int = MAX_AI_WORKERS,
    max_retries: int = MAX_AI_RETRIES,
    on_page_complete: Optional[Callable[[dict, int, int], None]] = None,
) -> list:
    """Analyze comic pages in parallel using EdenAI only."""

    def process_page(page):
        page_number = page["page_number"]
        image_path = page["image_path"]
        page_start = time.perf_counter()
        attempts = 0
        retry_wait_total = 0.0

        if not Path(image_path).exists():
            duration = time.perf_counter() - page_start
            return {
                "page_number": page_number,
                "filename": page.get("filename", Path(image_path).name),
                "image_path": image_path,
                "analysis": None,
                "metadata": {"page_number": page_number},
                "status": "error",
                "error": f"Image file not found: {image_path}",
                "perf": {
                    "duration": duration,
                    "attempts": 1,
                    "retry_wait": 0.0,
                },
            }

        last_error = None

        for attempt in range(max_retries + 1):
            attempts += 1
            retry_start = time.perf_counter()

            try:
                result = analyze_page(image_path)
                result = _normalize_page_analysis(result)

                duration = time.perf_counter() - page_start
                text_data = result.get("text", {})
                if not isinstance(text_data, dict):
                    text_data = {}

                full_text = text_data.get("full_text", "")
                if not isinstance(full_text, str):
                    full_text = str(full_text or "")

                return {
                    "page_number": page_number,
                    "filename": page.get(
                        "filename",
                        Path(image_path).name,
                    ),
                    "image_path": image_path,
                    "analysis": result,
                    "metadata": {
                        "page_number": page_number,
                        "has_text": bool(full_text.strip()),
                    },
                    "status": "success",
                    "perf": {
                        "duration": duration,
                        "attempts": attempts,
                        "retry_wait": retry_wait_total,
                    },
                }

            except Exception as exc:
                last_error = exc

                if attempt >= max_retries:
                    break

                wait_time = min(
                    2.0 * (2 ** attempt) + random.uniform(0.1, 0.5),
                    30.0,
                )
                time.sleep(wait_time)
                retry_wait_total += (
                    time.perf_counter() - retry_start
                )

                logger.warning(
                    "[EDEN] Page %s failed: %s. "
                    "Retrying in %.1fs (attempt %d/%d)...",
                    page_number,
                    exc,
                    wait_time,
                    attempt + 1,
                    max_retries + 1,
                )

        duration = time.perf_counter() - page_start

        return {
            "page_number": page_number,
            "filename": page.get(
                "filename",
                Path(image_path).name,
            ),
            "image_path": image_path,
            "analysis": None,
            "metadata": {"page_number": page_number},
            "status": "error",
            "error": str(last_error),
            "perf": {
                "duration": duration,
                "attempts": attempts,
                "retry_wait": retry_wait_total,
            },
        }

    results = []
    total_pages = len(pages)
    ai_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_page, page): page["page_number"]
            for page in pages
        }

        completed = 0

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if result["status"] == "success":
                logger.info(
                    "[%s/%s] Page %s analyzed successfully by EdenAI",
                    completed,
                    total_pages,
                    result["page_number"],
                )
            else:
                logger.error(
                    "[%s/%s] Page %s analysis failed: %s",
                    completed,
                    total_pages,
                    result["page_number"],
                    result.get("error", "Unknown error"),
                )

            if on_page_complete is not None:
                try:
                    on_page_complete(
                        result,
                        completed,
                        total_pages,
                    )
                except Exception as callback_error:
                    logger.warning(
                        "on_page_complete callback error: %s",
                        callback_error,
                    )

    total_ai_duration = time.perf_counter() - ai_start

    successful = sum(
        1 for result in results
        if result.get("status") == "success"
    )
    failed = len(results) - successful

    durations = [
        result.get("perf", {}).get("duration", 0.0)
        for result in results
    ]
    attempts_list = [
        result.get("perf", {}).get("attempts", 1)
        for result in results
    ]

    logger.info(
        "\n"
        "[PERF] ==================================================\n"
        "[PERF] EDENAI ANALYSIS PERFORMANCE\n"
        "[PERF] Pages: %s\n"
        "[PERF] Workers: %s\n"
        "[PERF] Total duration: %.2fs\n"
        "[PERF] Successful: %s\n"
        "[PERF] Failed: %s\n"
        "[PERF] Total retries: %s\n"
        "[PERF] Average page duration: %.2fs\n"
        "[PERF] ==================================================",
        total_pages,
        max_workers,
        total_ai_duration,
        successful,
        failed,
        sum(max(0, value - 1) for value in attempts_list),
        (
            sum(durations) / len(durations)
            if durations else 0.0
        ),
    )

    results.sort(key=lambda item: item["page_number"])
    return results

