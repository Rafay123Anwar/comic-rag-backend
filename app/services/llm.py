# # """
# # LLM Service

# # Generates grounded comic answers using Mistral AI, enforcing strict grounding rules,
# # pinned fallback phrase, and optional conversation memory context.
# # """
# # import asyncio
# # import random
# # import re
# # import time
# # from mistralai.client import Mistral

# # from deep_translator import GoogleTranslator

# # from app.core.config import LLM_MODEL, MISTRAL_API_KEY, USE_LIBRARY_TRANSLATION

# # # -----------------------------
# # # Mistral Client
# # # -----------------------------

# # client = Mistral(
# #     api_key=MISTRAL_API_KEY
# # )


# # def _safe_chat_complete(
# #     messages: list[dict],
# #     model: str = LLM_MODEL,
# #     temperature: float = 0.3,
# #     max_tokens: int = 1500,
# #     max_retries: int = 5,
# #     **kwargs
# # ):
# #     """
# #     Executes client.chat.complete with exponential backoff on 429 rate limit errors.
# #     """
# #     for attempt in range(max_retries + 1):
# #         try:
# #             return client.chat.complete(
# #                 model=model,
# #                 messages=messages,
# #                 temperature=temperature,
# #                 max_tokens=max_tokens,
# #                 **kwargs
# #             )
# #         except Exception as e:
# #             err_str = str(e).lower()
# #             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
# #                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
# #                 time.sleep(wait_time)
# #             else:
# #                 raise


# # def _safe_chat_stream(
# #     messages: list[dict],
# #     model: str = LLM_MODEL,
# #     temperature: float = 0.3,
# #     max_tokens: int = 1800,
# #     max_retries: int = 5,
# #     **kwargs
# # ):
# #     """
# #     Executes client.chat.stream with exponential backoff on 429 rate limit errors.
# #     """
# #     for attempt in range(max_retries + 1):
# #         try:
# #             return client.chat.stream(
# #                 model=model,
# #                 messages=messages,
# #                 temperature=temperature,
# #                 max_tokens=max_tokens,
# #                 **kwargs
# #             )
# #         except Exception as e:
# #             err_str = str(e).lower()
# #             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
# #                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
# #                 time.sleep(wait_time)
# #             else:
# #                 raise


# # async def _safe_chat_stream_async(
# #     messages: list[dict],
# #     model: str = LLM_MODEL,
# #     temperature: float = 0.3,
# #     max_tokens: int = 1800,
# #     max_retries: int = 5,
# #     **kwargs
# # ):
# #     """
# #     Executes client.chat.stream_async with exponential backoff on 429 rate limit errors.
# #     """
# #     for attempt in range(max_retries + 1):
# #         try:
# #             return await client.chat.stream_async(
# #                 model=model,
# #                 messages=messages,
# #                 temperature=temperature,
# #                 max_tokens=max_tokens,
# #                 **kwargs
# #             )
# #         except Exception as e:
# #             err_str = str(e).lower()
# #             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
# #                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
# #                 await asyncio.sleep(wait_time)
# #             else:
# #                 raise


# # ENGLISH_FUNCTION_WORDS = {
# #     "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
# #     "is", "are", "was", "were", "am", "be", "been", "being",
# #     "do", "does", "did", "done",
# #     "have", "has", "had", "having",
# #     "can", "could", "would", "should", "will", "shall", "might", "must",
# #     "the", "a", "an",
# #     "in", "on", "at", "to", "for", "from", "with", "about", "by", "of", "into", "through", "after", "before",
# #     "and", "or", "but", "if", "because", "as", "than", "so",
# #     "he", "she", "it", "they", "him", "her", "his", "their", "them", "my", "your", "our", "its",
# #     "tell", "me", "explain", "give", "show", "describe", "summarize", "find", "list",
# #     "this", "that", "these", "those", "there", "here", "any", "some", "all",
# #     "plot", "story", "character", "characters", "page", "pages", "book", "comic"
# # }

# # NON_ENGLISH_MARKERS = {
# #     # Roman Urdu / Hindi question words & markers
# #     "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
# #     "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
# #     "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
# #     "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
# #     "mera", "meri", "mere", "apna", "apni", "apne",
# #     "mujhe", "tum", "aap", "humein", "hum",
# #     "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
# #     "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
# #     "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
# #     # Spanish / French / German / other common European markers
# #     "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
# #     "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
# # }


# # def is_english_query(query: str) -> bool:
# #     """
# #     Lightweight heuristic check to detect if a query is English.
# #     Returns True if the query appears to be English, False otherwise.
# #     """
# #     if not query or not query.strip():
# #         return True

# #     # 1. Non-Latin script check (Urdu, Arabic, Hindi/Devanagari, CJK, Cyrillic, etc.)
# #     non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
# #     if non_latin:
# #         return False

# #     words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
# #     if not words:
# #         return True

# #     # 2. Check for explicit non-English indicator tokens
# #     for w in words:
# #         if w in NON_ENGLISH_MARKERS:
# #             return False

# #     # 3. Count English vocabulary words
# #     english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
# #     if english_word_count > 0:
# #         return True

# #     return True


# # def is_urdu_script_query(query: str) -> bool:
# #     """Checks if a query contains Arabic / Urdu script characters."""
# #     if not query:
# #         return False
# #     return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


# # def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
# #     """
# #     Extracts proper nouns / character names from English text and replaces them with
# #     protected placeholder tokens (__PROPER_NAME_X__) so translation tools do not alter them.
# #     """
# #     if not text:
# #         return text, {}
# #     stopwords = {
# #         "On", "In", "The", "A", "An", "This", "That", "These", "Those", "Page", "Based",
# #         "According", "When", "If", "While", "After", "Before", "He", "She", "They", "It",
# #         "His", "Her", "Their", "As", "At", "By", "For", "From", "With", "To", "Comic", "Context",
# #         "Current", "Question", "Answer", "Yes", "No", "Not", "All", "Some", "Any"
# #     }
# #     pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:Von|De|La|van|von|der|of)\s+[A-Z][a-zA-Z]*|\s+[A-Z][a-zA-Z]*)*\b"
# #     matches = list(re.finditer(pattern, text))
# #     entities = []
# #     for m in matches:
# #         ent = m.group(0).strip()
# #         if ent in stopwords or ent.lower() in {"page", "comic", "context", "chapter"}:
# #             continue
# #         if ent not in entities:
# #             entities.append(ent)
# #     entities.sort(key=len, reverse=True)
# #     mapping = {}
# #     protected = text
# #     for i, ent in enumerate(entities):
# #         placeholder = f"__PROPER_NAME_{i}__"
# #         mapping[placeholder] = ent
# #         protected = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected)
# #     return protected, mapping


# # def restore_proper_names(text: str, mapping: dict[str, str]) -> str:
# #     """Restores protected placeholder tokens back to their original exact proper names."""
# #     if not text or not mapping:
# #         return text
# #     restored = text
# #     for placeholder, original in mapping.items():
# #         restored = restored.replace(placeholder, original)
# #     return restored


# # def transliterate_to_roman_urdu(urdu_text: str) -> str:
# #     """
# #     Lightweight, fast transliteration of Urdu script into Roman Urdu using Latin alphabet (A-Z).
# #     """
# #     if not urdu_text or not urdu_text.strip():
# #         return urdu_text

# #     prompt = (
# #         "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
# #         "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only (e.g. 'Kahani ek aise shakhs ke baare mein...').\n"
# #         "RULES:\n"
# #         "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
# #         "2. Preserve all character names, proper nouns (e.g. Cynthia, Werner Von Doom, Victor, Baron), and placeholder tokens (__PROPER_NAME_*) EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
# #         "3. Transliterate the entire text completely from beginning to end without stopping early or adding commentary.\n"
# #         "4. Output ONLY the transliterated text."
# #     )

# #     response = _safe_chat_complete(
# #         model=LLM_MODEL,
# #         messages=[
# #             {
# #                 "role": "system",
# #                 "content": prompt
# #             },
# #             {
# #                 "role": "user",
# #                 "content": urdu_text
# #             }
# #         ],
# #         temperature=0.1,
# #         max_tokens=1500
# #     )
# #     return response.choices[0].message.content.strip()


# # def translate_with_library(english_text: str, original_question: str) -> str:
# #     """
# #     Fast translation using deep-translator (GoogleTranslator) with Roman Urdu transliteration,
# #     strict character name protection, and automatic fallback to LLM translation if needed.
# #     """
# #     if not english_text or not english_text.strip():
# #         return english_text

# #     try:
# #         protected_text, name_map = extract_and_protect_proper_names(english_text)

# #         # Case 1: Urdu Script query -> Direct English to Urdu script translation
# #         if is_urdu_script_query(original_question):
# #             translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
# #             restored_ur = restore_proper_names(translated_ur, name_map)
# #             return clean_llm_response(restored_ur, question=original_question)

# #         # Case 2: Roman Urdu query -> English to Urdu script via library, then lightweight transliteration to Latin letters
# #         urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
# #         roman_urdu = transliterate_to_roman_urdu(urdu_script)
# #         restored_roman = restore_proper_names(roman_urdu, name_map)
# #         return clean_llm_response(restored_roman, question=original_question)

# #     except Exception as e:
# #         print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
# #         return translate_answer(english_text, original_question)


# # def translate_answer(english_answer: str, original_question: str) -> str:
# #     """
# #     Step 2 translation: Translates an English grounded answer into the language,
# #     script, and style of the user's original question using Mistral.
# #     """
# #     translation_system_prompt = (
# #         "You are a precise, natural translator. Translate the text faithfully "
# #         "matching the exact language, script/alphabet, and style of the user's question."
# #     )

# #     translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.

# # CRITICAL SCRIPT & LANGUAGE RULES:
# # 1. SCRIPT MATCHING:
# #    - If the USER'S ORIGINAL QUESTION is written in Latin/English letters (A-Z, e.g. Roman Urdu, Hinglish, Spanish, French, German, Indonesian, Romaji, etc.), your translation MUST be written entirely in Latin/English letters (A-Z). For Roman Urdu questions (e.g. "story kiya hai"), write in natural Roman Urdu using Latin alphabet — NEVER use Arabic/Urdu script (اردو).
# #    - If the USER'S ORIGINAL QUESTION is written in a non-Latin script (e.g. Arabic/Urdu script like اردو, Devanagari, Japanese Kana/Kanji, Cyrillic, etc.), write in that exact script.
# # 2. ACCURACY: Preserve all facts, character names, events, and details EXACTLY as stated in the English text — do not invent, add, or omit any details.
# # 3. PRESERVE PROPER NOUNS & CHARACTER NAMES: Keep all character names (e.g. 'Cynthia', 'Werner Von Doom', 'Victor Von Doom', 'Baron', 'Latveria') EXACTLY in their original English spelling. NEVER translate, alter, or phonetically transliterate character names into Urdu sound-approximations like 'Santhya', 'Santhiya', 'Varner', etc.
# # 4. NATURAL PHRASING: Use natural, conversational grammar as written by native speakers.
# # 5. FORMATTING: Keep proper nouns, character names, and markdown formatting intact.
# # 6. NO COMMENTARY: Output ONLY the translated text. Do not add intros, notes, or explanations.

# # USER'S ORIGINAL QUESTION (for language & script reference):
# # {original_question}

# # ENGLISH TEXT TO TRANSLATE:
# # {english_answer}"""

# #     response = _safe_chat_complete(
# #         model=LLM_MODEL,
# #         messages=[
# #             {
# #                 "role": "system",
# #                 "content": translation_system_prompt
# #             },
# #             {
# #                 "role": "user",
# #                 "content": translation_user_prompt
# #             }
# #         ],
# #         temperature=0.2,
# #         max_tokens=1500
# #     )

# #     translated = response.choices[0].message.content.strip()
# #     return clean_llm_response(translated, question=original_question)


# # def clean_llm_response(text: str, question: str = "") -> str:
# #     """
# #     Post-processes the LLM output to enforce formatting and grounding compliance:
# #     - Strips meta-commentary parentheticals (e.g. '(no outside knowledge used)')
# #     - Strips unrequested 'Missing Details' sections
# #     - Strips excessive bolding (max 1-2 bold terms total, unbolds section headers and bullet labels)
# #     """
# #     if not text or not text.strip():
# #         return text

# #     cleaned = text.strip()

# #     # 1. Strip meta-commentary parentheticals/brackets
# #     cleaned = re.sub(
# #         r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context|no external knowledge)[^\)]*\)",
# #         "",
# #         cleaned,
# #         flags=re.IGNORECASE
# #     )
# #     cleaned = re.sub(
# #         r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context|no external knowledge)[^\]]*\]",
# #         "",
# #         cleaned,
# #         flags=re.IGNORECASE
# #     )

# #     # 2. Strip 'Missing Details' / 'Missing Information' sections unless explicitly requested
# #     q_lower = question.lower()
# #     if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
# #         missing_section_pattern = re.compile(
# #             r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing / Unclear Details|Unclear Information|Gaps in Information|Missing aspects|Things not mentioned)(?:\*\*)?:?.*$",
# #             re.IGNORECASE | re.DOTALL
# #         )
# #         cleaned = missing_section_pattern.sub("", cleaned)

# #     # 3. Unbold section headers, category titles, and bullet labels (e.g. **Summary:** -> Summary:, **Key Events:** -> Key Events:)
# #     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
# #     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)

# #     # 4. Enforce the BOLD FORMATTING RULE (max 1-2 bolded terms in the entire response)
# #     bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", cleaned))
# #     if len(bold_matches) > 2:
# #         count = 0
# #         def unbold_excess(match):
# #             nonlocal count
# #             count += 1
# #             if count <= 2:
# #                 return match.group(0)
# #             return match.group(1)
# #         cleaned = re.sub(r"\*\*(.*?)\*\*", unbold_excess, cleaned)

# #     return cleaned.strip()


# # BROAD_SUMMARY_REGEX = re.compile(
# #     r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|what is happening|describe this page|explain this page|tell me about this page)\b|"
# #     r"\b(kya hua|kya ho raha|khulasa|poori kahani|puri kahani|sari kahani|sab batao|is page pe|ye page pe|yeh page pe)\b",
# #     re.IGNORECASE
# # )

# # SHORT_REQUEST_REGEX = re.compile(
# #     r"\b(short|shortly|brief|briefly|in short|quick|quickly|summarize quickly|one line|two lines|just a line|few words|concise|tldr)\b|"
# #     r"\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|kam\s+(shabdon|alfaaz|words)\s+(mein|mai|me)|chhoti|choti|short\s+karke|aik\s+line|do\s+line)\b",
# #     re.IGNORECASE
# # )

# # DETAIL_REQUEST_REGEX = re.compile(
# #     r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details|comprehensive)\b|"
# #     r"\b(poora\s+batao|pura\s+batao|puri\s+detail|poori\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao|tafseel)\b",
# #     re.IGNORECASE
# # )


# # def is_explicit_short_query(question: str) -> bool:
# #     """Detect if the user explicitly requested a short/brief answer."""
# #     return bool(question and SHORT_REQUEST_REGEX.search(question))


# # def is_explicit_detail_query(question: str) -> bool:
# #     """Detect if the user explicitly requested a detailed/elaborated answer."""
# #     return bool(question and DETAIL_REQUEST_REGEX.search(question))


# # def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
# #     """
# #     Returns an optimized token budget based on query intent:
# #     - 250 tokens for explicit short answer requests ('short mai batao', 'briefly', etc.)
# #     - 1800 tokens for explicit detailed breakdown requests ('in detail', 'poora batao')
# #     - 1500 tokens for broad summary / page overview requests
# #     - 700 tokens for short factual / character lookups
# #     """
# #     if is_explicit_short_query(question):
# #         return 250
# #     if is_explicit_detail_query(question):
# #         return 1800
# #     if current_page is not None:
# #         return 1500
# #     if question and BROAD_SUMMARY_REGEX.search(question):
# #         return 1500
# #     return 700


# # # -----------------------------
# # # Generate Answer
# # # -----------------------------

# # def generate_answer(
# #     question: str,
# #     context: str,
# #     conversation_history: list[dict] | None = None,
# #     current_page: int | None = None
# # ) -> str:
# #     """
# #     Generates an answer strictly grounded in comic context using Mistral AI.
# #     - Responds in clear English by default, or Roman Urdu if asked in Roman Urdu.
# #     - Strictly scopes response to active page when current_page is provided.
# #     - Enforces BOLD FORMATTING RULE (1-2 bold terms max) and complete punctuation.
# #     - Dynamically allocates token budget (250 for short, 700 for factual, 1500 for broad summary).
# #     """
# #     if not question or not question.strip():
# #         raise ValueError("Question cannot be empty.")

# #     if not context or not context.strip():
# #         return "I could not find relevant information in the comic."

# #     system_prompt, user_prompt = _build_streaming_prompts(
# #         question=question,
# #         context=context,
# #         conversation_history=conversation_history,
# #         current_page=current_page
# #     )

# #     token_budget = get_dynamic_max_tokens(question, current_page)

# #     print("=" * 60)
# #     print("LLM DEBUG (GROUNDED QA)")
# #     print("LLM MODEL:", LLM_MODEL)
# #     print("QUESTION:", question)
# #     print("CURRENT PAGE:", current_page)
# #     print("DYNAMIC MAX TOKENS:", token_budget)
# #     print("CONTEXT LENGTH:", len(context))
# #     print("=" * 60)

# #     response = _safe_chat_complete(
# #         model=LLM_MODEL,
# #         messages=[
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ],
# #         temperature=0.2,
# #         max_tokens=token_budget,
# #         frequency_penalty=0.3,
# #         presence_penalty=0.2
# #     )

# #     raw_answer = response.choices[0].message.content.strip()
# #     cleaned = clean_llm_response(raw_answer, question=question)

# #     print("=" * 60)
# #     print("LLM DEBUG RESULT:")
# #     print(cleaned)
# #     print("=" * 60)

# #     return cleaned


# # # -----------------------------
# # # Stream Generate Answer
# # # -----------------------------

# # def _build_streaming_prompts(
# #     question: str,
# #     context: str,
# #     conversation_history: list[dict] | None = None,
# #     current_page: int | None = None
# # ) -> tuple[str, str]:
# #     """Helper to assemble system and user prompts for grounded streaming QA."""
# #     is_urdu_script = is_urdu_script_query(question)
# #     is_english = is_english_query(question) and not is_urdu_script
# #     is_roman_urdu = not is_english and not is_urdu_script

# #     if is_urdu_script:
# #         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
# #     elif is_roman_urdu:
# #         lang_reminder = (
# #             f"\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu: '{question}'. "
# #             "You MUST write your entire response in authentic, natural Roman Urdu using the Latin alphabet (A-Z). "
# #             "Do NOT answer in English. Do NOT output Arabic/Urdu script (اردو).)"
# #         )
# #     else:
# #         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

# #     if is_explicit_short_query(question):
# #         formatting_reminder = (
# #             "\n\n(CRITICAL LENGTH CONSTRAINT: The user explicitly demanded a SHORT answer. "
# #             "Provide a SHORT, punchy response of EXACTLY 1-3 sentences in a single brief paragraph. "
# #             "Use markdown bold for AT MOST 1-2 key terms. Finish sentences completely.)"
# #         )
# #     elif is_explicit_detail_query(question):
# #         formatting_reminder = (
# #             "\n\n(LENGTH CONSTRAINT: The user requested a DETAILED breakdown. "
# #             "Provide a comprehensive, rich narrative covering all relevant details from the comic context. "
# #             "Use markdown bold for AT MOST 1-2 key terms. Finish sentences completely.)"
# #         )
# #     else:
# #         formatting_reminder = (
# #             "\n\n(FORMATTING & STYLE CONSTRAINTS: Write with genuine comic fan excitement and cinematic drama. "
# #             "Use markdown bold for AT MOST 1-2 key terms in your entire answer. "
# #             "Prefer a flowing narrative paragraph over bullet lists unless an explicit list was requested. Finish all thoughts and sentences completely.)"
# #         )

# #     page_hint = ""
# #     if current_page is not None:
# #         try:
# #             from app.services.rag_qa import is_page_scoped_query
# #             is_page_specific = is_page_scoped_query(question, current_page)
# #         except Exception:
# #             is_page_specific = False

# #         if is_page_specific:
# #             page_hint = (
# #                 f"\nCURRENT USER VIEWING PAGE: Page {current_page}\n"
# #                 f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on Page {current_page} evidence.)\n"
# #             )
# #         else:
# #             page_hint = (
# #                 f"\n(Note: The user is currently viewing Page {current_page}, but this question is asking about the comic's overall story, events, or characters. Synthesize the answer from all provided comic context across pages.)\n"
# #             )

# #     if conversation_history:
# #         history_lines = []
# #         for msg in conversation_history:
# #             role_label = "User" if msg.get("role") == "user" else "Assistant"
# #             content = msg.get("content", "").strip()
# #             if content:
# #                 history_lines.append(f"{role_label}: {content}")
# #         history_text = "\n".join(history_lines) if history_lines else "None"
# #         user_prompt = f"""CONVERSATION HISTORY:
# # {history_text}
# # {page_hint}
# # COMIC CONTEXT:
# # {context}

# # CURRENT QUESTION:
# # {question}{lang_reminder}{formatting_reminder}
# # """
# #     else:
# #         user_prompt = f"""{page_hint}COMIC CONTEXT:
# # {context}

# # CURRENT QUESTION:
# # {question}{lang_reminder}{formatting_reminder}
# # """

# #     system_prompt = (
# #         "You are a real comic book enthusiast and expert companion talking directly to another comic fan.\n"
# #         "Your ONLY factual grounding source is the provided COMIC CONTEXT.\n\n"
# #         "==================================================\n"
# #         "AUTHENTIC COMIC READER VOICE (NO AI BOT VIBE)\n"
# #         "==================================================\n"
# #         "1. TALK LIKE A REAL COMIC READER, NOT AN AI BOT:\n"
# #         "   - Speak naturally, directly, and vividly about the characters, panels, action, dialogue, and lore.\n"
# #         "   - NEVER leak internal technical jargon: NEVER say words like 'chunk', 'chunks', 'pehle chunk mein', 'retrieval', 'context', or 'database'. Speak strictly in terms of 'panel', 'scene', 'page', 'artwork', 'background', or 'characters'.\n"
# #         "   - NEVER use robotic meta-commentary like 'Yeh comic dastaan sunata hai...', 'This narrative illustrates...', or 'Reader ko yeh feel hota hai...'.\n"
# #         "   - NEVER be argumentative or pedantic: When a user asks about a character by an action or appearance (e.g., 'jo larki dance kar rahi hai', 'who is the person with the sword'), directly identify WHO that character is (e.g., Cynthia Von Doom / Victor's mother) and explain their role across the panels with warmth and lore expertise.\n"
# #         "   - In Roman Urdu: Sound like a passionate comic fan having a real conversation (e.g., 'Page ke top panel mein jo larki aag ke samne dance kar rahi hai, woh **Cynthia Von Doom** hai—Victor ki maa...').\n"
# #         "   - In English: Deliver punchy, engaging comic storytelling with character focus and dramatic energy.\n\n"
# #         "==================================================\n"
# #         "UNIVERSAL COMIC CONTEXT UNDERSTANDING\n"
# #         "==================================================\n"
# #         "1. ANY COMIC FORMAT (Superhero, Manga, Sci-Fi, Dark Fantasy, Indie):\n"
# #         "   - Combine dialogue/OCR text, panel visual descriptions, character emotions, and environmental cues from the COMIC CONTEXT to understand the exact story beats.\n"
# #         "   - Accurately identify who is speaking, who is fighting, what powers/items are used, and the narrative stakes.\n"
# #         "2. SINGLE-PAGE OR MULTI-PAGE COMICS:\n"
# #         "   - If the comic is 1 page or a single excerpt, narrate the scene and story unfolding across that page's panels from top to bottom without saying 'On the first page'.\n"
# #         "   - If multi-page, synthesize the complete story arc across all pages clearly.\n\n"
# #         "==================================================\n"
# #         "LANGUAGE & SCRIPT RULES\n"
# #         "==================================================\n"
# #         "1. MATCH USER LANGUAGE EXACTLY:\n"
# #         "   - If the question is in Roman Urdu ('story kiya hai', 'kya hua yahan', 'batao', 'on hai', 'kon hai', 'dance kar rahi hai'), answer 100% in natural Roman Urdu using Latin alphabet (A-Z).\n"
# #         "   - If the question is in English, answer in English.\n"
# #         "2. NO ARABIC/URDU SCRIPT: NEVER output Arabic/Urdu script (اردو) unless the user wrote their question in Arabic/Urdu script.\n"
# #         "3. CHARACTER NAMES: Preserve character names (e.g. 'Victor Von Doom', 'Cynthia', 'Batman', 'Peter Parker') in standard spelling.\n\n"
# #         "==================================================\n"
# #         "GROUNDING & FORMATTING RULES\n"
# #         "==================================================\n"
# #         "1. STRICT FACTUAL GROUNDING: Rely ONLY on the COMIC CONTEXT. Never invent unseen events or outside lore not in the context. If not found, reply: 'I could not find relevant information in the comic.'\n"
# #         "2. LENGTH: For short requests ('short mai', 'in short'), give 1-3 crisp sentences. For story overviews, give 1-2 rich flowing paragraphs.\n"
# #         "3. MINIMAL BOLDING: Use markdown bold for AT MOST 1-2 key character or story terms across the entire reply (e.g. **Cynthia Von Doom**).\n"
# #         "4. COMPLETION: Always complete every sentence properly with full punctuation."
# #     )

# #     return system_prompt, user_prompt


# # async def stream_generate_answer_async(
# #     question: str,
# #     context: str,
# #     conversation_history: list[dict] | None = None,
# #     current_page: int | None = None
# # ):
# #     """
# #     Asynchronous Grounded Question Answering Streaming Generator:
# #     - If question is English: Streams grounded tokens from Mistral LLM directly to client.
# #     - If question is Non-English: Streams grounded tokens directly in the target language/script matching question style,
# #       enforcing strict grounding in comic context, minimal bolding, and no truncation.
# #     """
# #     if not question or not question.strip():
# #         yield "Please provide a valid question."
# #         return

# #     if not context or not context.strip():
# #         yield "I could not find relevant information in the comic."
# #         return

# #     system_prompt, user_prompt = _build_streaming_prompts(
# #         question=question,
# #         context=context,
# #         conversation_history=conversation_history,
# #         current_page=current_page
# #     )

# #     token_budget = get_dynamic_max_tokens(question, current_page)

# #     stream_resp = await _safe_chat_stream_async(
# #         model=LLM_MODEL,
# #         messages=[
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ],
# #         temperature=0.2,
# #         max_tokens=token_budget,
# #         frequency_penalty=0.3,
# #         presence_penalty=0.2
# #     )

# #     async for chunk in stream_resp:
# #         delta = chunk.data.choices[0].delta.content
# #         if delta:
# #             yield delta


# # def stream_generate_answer(
# #     question: str,
# #     context: str,
# #     conversation_history: list[dict] | None = None,
# #     current_page: int | None = None
# # ):
# #     """
# #     Synchronous Grounded Question Answering Streaming Generator:
# #     - If question is English: Streams grounded tokens from Mistral LLM directly to caller.
# #     - If question is Non-English: Streams grounded tokens directly in the target language and script matching question style.
# #     """
# #     if not question or not question.strip():
# #         yield "Please provide a valid question."
# #         return

# #     if not context or not context.strip():
# #         yield "I could not find relevant information in the comic."
# #         return

# #     system_prompt, user_prompt = _build_streaming_prompts(
# #         question=question,
# #         context=context,
# #         conversation_history=conversation_history,
# #         current_page=current_page
# #     )

# #     token_budget = get_dynamic_max_tokens(question, current_page)

# #     stream_resp = _safe_chat_stream(
# #         model=LLM_MODEL,
# #         messages=[
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ],
# #         temperature=0.2,
# #         max_tokens=token_budget,
# #         frequency_penalty=0.3,
# #         presence_penalty=0.2
# #     )

# #     for chunk in stream_resp:
# #         delta = chunk.data.choices[0].delta.content
# #         if delta:
# #             yield delta

# """
# LLM Service

# Generates grounded comic answers using Mistral AI, enforcing strict grounding rules,
# pinned fallback phrase, and optional conversation memory context.
# Updated with highly optimized prompts to perfectly read nested JSON comic data.
# """
# import asyncio
# import random
# import re
# import time
# from mistralai.client import Mistral

# from deep_translator import GoogleTranslator

# from app.core.config import LLM_MODEL, MISTRAL_API_KEY, USE_LIBRARY_TRANSLATION

# # -----------------------------
# # Mistral Client
# # -----------------------------

# client = Mistral(
#     api_key=MISTRAL_API_KEY
# )


# def _safe_chat_complete(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1500,
#     max_retries: int = 5,
#     **kwargs
# ):
#     """
#     Executes client.chat.complete with exponential backoff on 429 rate limit errors.
#     """
#     for attempt in range(max_retries + 1):
#         try:
#             return client.chat.complete(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 raise


# def _safe_chat_stream(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     """
#     Executes client.chat.stream with exponential backoff on 429 rate limit errors.
#     """
#     for attempt in range(max_retries + 1):
#         try:
#             return client.chat.stream(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 raise


# async def _safe_chat_stream_async(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     """
#     Executes client.chat.stream_async with exponential backoff on 429 rate limit errors.
#     """
#     for attempt in range(max_retries + 1):
#         try:
#             return await client.chat.stream_async(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise


# ENGLISH_FUNCTION_WORDS = {
#     "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
#     "is", "are", "was", "were", "am", "be", "been", "being",
#     "do", "does", "did", "done",
#     "have", "has", "had", "having",
#     "can", "could", "would", "should", "will", "shall", "might", "must",
#     "the", "a", "an",
#     "in", "on", "at", "to", "for", "from", "with", "about", "by", "of", "into", "through", "after", "before",
#     "and", "or", "but", "if", "because", "as", "than", "so",
#     "he", "she", "it", "they", "him", "her", "his", "their", "them", "my", "your", "our", "its",
#     "tell", "me", "explain", "give", "show", "describe", "summarize", "find", "list",
#     "this", "that", "these", "those", "there", "here", "any", "some", "all",
#     "plot", "story", "character", "characters", "page", "pages", "book", "comic"
# }

# NON_ENGLISH_MARKERS = {
#     "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
#     "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
#     "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
#     "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
#     "mera", "meri", "mere", "apna", "apni", "apne",
#     "mujhe", "tum", "aap", "humein", "hum",
#     "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
#     "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
#     "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
#     "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
#     "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
# }


# def is_english_query(query: str) -> bool:
#     if not query or not query.strip():
#         return True
#     non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
#     if non_latin:
#         return False
#     words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
#     if not words:
#         return True
#     for w in words:
#         if w in NON_ENGLISH_MARKERS:
#             return False
#     english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
#     if english_word_count > 0:
#         return True
#     return True


# def is_urdu_script_query(query: str) -> bool:
#     if not query:
#         return False
#     return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


# def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
#     if not text:
#         return text, {}
#     stopwords = {
#         "On", "In", "The", "A", "An", "This", "That", "These", "Those", "Page", "Based",
#         "According", "When", "If", "While", "After", "Before", "He", "She", "They", "It",
#         "His", "Her", "Their", "As", "At", "By", "For", "From", "With", "To", "Comic", "Context",
#         "Current", "Question", "Answer", "Yes", "No", "Not", "All", "Some", "Any"
#     }
#     pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:Von|De|La|van|von|der|of)\s+[A-Z][a-zA-Z]*|\s+[A-Z][a-zA-Z]*)*\b"
#     matches = list(re.finditer(pattern, text))
#     entities = []
#     for m in matches:
#         ent = m.group(0).strip()
#         if ent in stopwords or ent.lower() in {"page", "comic", "context", "chapter"}:
#             continue
#         if ent not in entities:
#             entities.append(ent)
#     entities.sort(key=len, reverse=True)
#     mapping = {}
#     protected = text
#     for i, ent in enumerate(entities):
#         placeholder = f"__PROPER_NAME_{i}__"
#         mapping[placeholder] = ent
#         protected = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected)
#     return protected, mapping


# def restore_proper_names(text: str, mapping: dict[str, str]) -> str:
#     if not text or not mapping:
#         return text
#     restored = text
#     for placeholder, original in mapping.items():
#         restored = restored.replace(placeholder, original)
#     return restored


# def transliterate_to_roman_urdu(urdu_text: str) -> str:
#     if not urdu_text or not urdu_text.strip():
#         return urdu_text
#     prompt = (
#         "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
#         "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only.\n"
#         "RULES:\n"
#         "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
#         "2. Preserve all character names, proper nouns EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
#         "3. Output ONLY the transliterated text."
#     )
#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[{"role": "system", "content": prompt}, {"role": "user", "content": urdu_text}],
#         temperature=0.1,
#         max_tokens=1500
#     )
#     return response.choices[0].message.content.strip()


# def translate_with_library(english_text: str, original_question: str) -> str:
#     if not english_text or not english_text.strip():
#         return english_text
#     try:
#         protected_text, name_map = extract_and_protect_proper_names(english_text)
#         if is_urdu_script_query(original_question):
#             translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
#             restored_ur = restore_proper_names(translated_ur, name_map)
#             return clean_llm_response(restored_ur, question=original_question)
#         urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
#         roman_urdu = transliterate_to_roman_urdu(urdu_script)
#         restored_roman = restore_proper_names(roman_urdu, name_map)
#         return clean_llm_response(restored_roman, question=original_question)
#     except Exception as e:
#         print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
#         return translate_answer(english_text, original_question)


# def translate_answer(english_answer: str, original_question: str) -> str:
#     translation_system_prompt = (
#         "You are a precise, natural translator. Translate the text faithfully "
#         "matching the exact language, script/alphabet, and style of the user's question."
#     )
#     translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.
# CRITICAL RULES:
# 1. SCRIPT MATCHING: If the USER'S ORIGINAL QUESTION is in Roman Urdu, write entirely in Latin letters (A-Z). NEVER use Arabic/Urdu script (اردو).
# 2. PRESERVE NAMES: Keep all character names (e.g. 'Victor', 'Cynthia') EXACTLY in their English spelling.
# 3. NO COMMENTARY: Output ONLY the translated text.

# USER'S ORIGINAL QUESTION:
# {original_question}

# ENGLISH TEXT TO TRANSLATE:
# {english_answer}"""
#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[{"role": "system", "content": translation_system_prompt}, {"role": "user", "content": translation_user_prompt}],
#         temperature=0.2,
#         max_tokens=1500
#     )
#     translated = response.choices[0].message.content.strip()
#     return clean_llm_response(translated, question=original_question)


# def clean_llm_response(text: str, question: str = "") -> str:
#     if not text or not text.strip():
#         return text
#     cleaned = text.strip()
#     cleaned = re.sub(r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
#     cleaned = re.sub(r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
#     q_lower = question.lower()
#     if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
#         missing_section_pattern = re.compile(r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing aspects)(?:\*\*)?:?.*$", re.IGNORECASE | re.DOTALL)
#         cleaned = missing_section_pattern.sub("", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)
#     bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", cleaned))
#     if len(bold_matches) > 2:
#         count = 0
#         def unbold_excess(match):
#             nonlocal count
#             count += 1
#             if count <= 2:
#                 return match.group(0)
#             return match.group(1)
#         cleaned = re.sub(r"\*\*(.*?)\*\*", unbold_excess, cleaned)
#     return cleaned.strip()


# BROAD_SUMMARY_REGEX = re.compile(r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|describe this page|explain this page)\b|\b(kya hua|kya ho raha|poori kahani|puri kahani|sari kahani|sab batao)\b", re.IGNORECASE)
# SHORT_REQUEST_REGEX = re.compile(r"\b(short|shortly|brief|briefly|in short|quick|quickly|one line|two lines|tldr)\b|\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|aik\s+line|do\s+line)\b", re.IGNORECASE)
# DETAIL_REQUEST_REGEX = re.compile(r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details)\b|\b(poora\s+batao|pura\s+batao|puri\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao)\b", re.IGNORECASE)


# def is_explicit_short_query(question: str) -> bool:
#     return bool(question and SHORT_REQUEST_REGEX.search(question))

# def is_explicit_detail_query(question: str) -> bool:
#     return bool(question and DETAIL_REQUEST_REGEX.search(question))

# def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
#     if is_explicit_short_query(question):
#         return 250
#     if is_explicit_detail_query(question):
#         return 1800
#     if current_page is not None:
#         return 1500
#     if question and BROAD_SUMMARY_REGEX.search(question):
#         return 1500
#     return 700


# # -----------------------------
# # Generate Answer
# # -----------------------------

# def generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> str:
#     if not question or not question.strip():
#         raise ValueError("Question cannot be empty.")
#     if not context or not context.strip():
#         return "I could not find relevant information in the comic."

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     raw_answer = response.choices[0].message.content.strip()
#     cleaned = clean_llm_response(raw_answer, question=question)
#     return cleaned


# # -----------------------------
# # Prompt Builder
# # -----------------------------

# def _build_streaming_prompts(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> tuple[str, str]:
#     """Helper to assemble system and user prompts optimized for structured JSON data."""
#     is_urdu_script = is_urdu_script_query(question)
#     is_english = is_english_query(question) and not is_urdu_script
#     is_roman_urdu = not is_english and not is_urdu_script

#     if is_urdu_script:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
#     elif is_roman_urdu:
#         lang_reminder = (
#             f"\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu: '{question}'. "
#             "You MUST write your entire response in authentic, natural Roman Urdu using the Latin alphabet (A-Z). "
#             "Do NOT answer in English. Do NOT output Arabic/Urdu script (اردو).)"
#         )
#     else:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

#     if is_explicit_short_query(question):
#         formatting_reminder = (
#             "\n\n(CRITICAL FORMATTING: The user explicitly demanded a SHORT answer. "
#             "Provide a brief, punchy response of EXACTLY 1-3 sentences. Use markdown bold for AT MOST 1-2 key terms.)"
#         )
#     elif is_explicit_detail_query(question):
#         formatting_reminder = (
#             "\n\n(FORMATTING: The user requested a DETAILED breakdown. "
#             "Provide a rich narrative covering all relevant details (characters, environment, actions) from the context. "
#             "Use markdown bold for AT MOST 1-2 key terms.)"
#         )
#     else:
#         formatting_reminder = (
#             "\n\n(FORMATTING: Write with genuine comic fan excitement. "
#             "Prefer a flowing narrative paragraph over rigid bullet lists. "
#             "Use markdown bold for AT MOST 1-2 key terms in your entire answer. Finish all thoughts completely.)"
#         )

#     page_hint = ""
#     if current_page is not None:
#         try:
#             from app.services.rag_qa import is_page_scoped_query
#             is_page_specific = is_page_scoped_query(question, current_page)
#         except Exception:
#             is_page_specific = False

#         if is_page_specific:
#             page_hint = (
#                 f"\nCURRENT USER VIEWING PAGE: Page {current_page}\n"
#                 f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on evidence from Page {current_page}.)\n"
#             )
#         else:
#             page_hint = (
#                 f"\n(Note: The user is currently viewing Page {current_page}, but synthesizing the answer from all provided context.)\n"
#             )

#     if conversation_history:
#         history_lines = []
#         for msg in conversation_history:
#             role_label = "User" if msg.get("role") == "user" else "Assistant"
#             content = msg.get("content", "").strip()
#             if content:
#                 history_lines.append(f"{role_label}: {content}")
#         history_text = "\n".join(history_lines) if history_lines else "None"
#         user_prompt = f"""CONVERSATION HISTORY:
# {history_text}
# {page_hint}
# RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """
#     else:
#         user_prompt = f"""{page_hint}RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """

#     # YAHAN SYSTEM PROMPT UPDATE KIYA HAI: Taake wo JSON structures ko effectively read kare
#     system_prompt = (
#         "You are an expert comic book companion. Your ONLY factual grounding source is the provided RICH COMIC CONTEXT.\n"
#         "The context you receive is extracted from highly detailed JSON structures. It contains:\n"
#         " - PAGE SUMMARIES: A high-level overview of the page's events.\n"
#         " - PANEL BREAKDOWNS: Specific details for every panel (Characters, Actions, Environment, Objects).\n"
#         " - TEXT/DIALOGUE: The exact OCR text, dialogue, and narration.\n\n"
#         "==================================================\n"
#         "HOW TO READ THE DATA & ANSWER:\n"
#         "==================================================\n"
#         "1. SYNTHESIZE, DON'T RECITE:\n"
#         "   - Do NOT just spit back the JSON keys to the user. Do not say 'In the Characters array it says...' or 'According to Panel 3 actions...'.\n"
#         "   - Instead, weave the data into a natural story. For example, if Panel 3 characters array says 'Victor' and actions array says 'splashing water', you say: 'Victor is seen playfully splashing water in the third panel.'\n"
#         "2. MATCH VISUALS WITH DIALOGUE:\n"
#         "   - If a user asks what a character is doing while saying a specific line, cross-reference the 'dialogue_and_narration' with the 'actions' and 'environment' for that specific panel.\n"
#         "3. IDENTIFY CHARACTERS EXPERTLY:\n"
#         "   - If the user asks 'who is the guy in the blue shirt', look at the character descriptions and confidently identify them based on the context.\n\n"
#         "==================================================\n"
#         "AUTHENTIC COMIC READER VOICE\n"
#         "==================================================\n"
#         "1. Speak naturally, directly, and vividly about the art, characters, and lore.\n"
#         "2. NEVER use robotic phrases like 'This text indicates...' or 'The visual data shows...'.\n"
#         "3. In Roman Urdu: Sound like a passionate desi comic fan (e.g., 'Page ke teesre panel mein jo larki red dress mein hai, woh Valeria hai...').\n\n"
#         "==================================================\n"
#         "STRICT GROUNDING RULE\n"
#         "==================================================\n"
#         "Rely ONLY on the provided context. If the answer is not in the context, reply exactly: 'I could not find relevant information in the comic.'"
#     )

#     return system_prompt, user_prompt


# async def stream_generate_answer_async(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = await _safe_chat_stream_async(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     async for chunk in stream_resp:
#         delta = chunk.data.choices[0].delta.content
#         if delta:
#             yield delta


# def stream_generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = _safe_chat_stream(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     for chunk in stream_resp:
#         delta = chunk.data.choices[0].delta.content
#         if delta:
#             yield delta

# """
# LLM Service

# Generates grounded comic answers using Mistral AI, enforcing strict grounding rules,
# pinned fallback phrase, and optional conversation memory context.
# Updated to fix the "over-excited fanboy" tone and make it natural and concise.
# """
# import asyncio
# import random
# import re
# import time
# from mistralai.client import Mistral

# from deep_translator import GoogleTranslator

# from app.core.config import LLM_MODEL, MISTRAL_API_KEY, USE_LIBRARY_TRANSLATION

# # -----------------------------
# # Mistral Client
# # -----------------------------

# client = Mistral(
#     api_key=MISTRAL_API_KEY
# )


# def _safe_chat_complete(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1500,
#     max_retries: int = 5,
#     **kwargs
# ):
#     for attempt in range(max_retries + 1):
#         try:
#             return client.chat.complete(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 raise


# def _safe_chat_stream(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     for attempt in range(max_retries + 1):
#         try:
#             return client.chat.stream(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 raise


# async def _safe_chat_stream_async(
#     messages: list[dict],
#     model: str = LLM_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     for attempt in range(max_retries + 1):
#         try:
#             return await client.chat.stream_async(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 **kwargs
#             )
#         except Exception as e:
#             err_str = str(e).lower()
#             if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise


# ENGLISH_FUNCTION_WORDS = {
#     "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
#     "is", "are", "was", "were", "am", "be", "been", "being",
#     "do", "does", "did", "done",
#     "have", "has", "had", "having",
#     "can", "could", "would", "should", "will", "shall", "might", "must",
#     "the", "a", "an",
#     "in", "on", "at", "to", "for", "from", "with", "about", "by", "of", "into", "through", "after", "before",
#     "and", "or", "but", "if", "because", "as", "than", "so",
#     "he", "she", "it", "they", "him", "her", "his", "their", "them", "my", "your", "our", "its",
#     "tell", "me", "explain", "give", "show", "describe", "summarize", "find", "list",
#     "this", "that", "these", "those", "there", "here", "any", "some", "all",
#     "plot", "story", "character", "characters", "page", "pages", "book", "comic"
# }

# NON_ENGLISH_MARKERS = {
#     "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
#     "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
#     "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
#     "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
#     "mera", "meri", "mere", "apna", "apni", "apne",
#     "mujhe", "tum", "aap", "humein", "hum",
#     "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
#     "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
#     "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
#     "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
#     "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
# }


# def is_english_query(query: str) -> bool:
#     if not query or not query.strip():
#         return True
#     non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
#     if non_latin:
#         return False
#     words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
#     if not words:
#         return True
#     for w in words:
#         if w in NON_ENGLISH_MARKERS:
#             return False
#     english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
#     if english_word_count > 0:
#         return True
#     return True


# def is_urdu_script_query(query: str) -> bool:
#     if not query:
#         return False
#     return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


# def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
#     if not text:
#         return text, {}
#     stopwords = {
#         "On", "In", "The", "A", "An", "This", "That", "These", "Those", "Page", "Based",
#         "According", "When", "If", "While", "After", "Before", "He", "She", "They", "It",
#         "His", "Her", "Their", "As", "At", "By", "For", "From", "With", "To", "Comic", "Context",
#         "Current", "Question", "Answer", "Yes", "No", "Not", "All", "Some", "Any"
#     }
#     pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:Von|De|La|van|von|der|of)\s+[A-Z][a-zA-Z]*|\s+[A-Z][a-zA-Z]*)*\b"
#     matches = list(re.finditer(pattern, text))
#     entities = []
#     for m in matches:
#         ent = m.group(0).strip()
#         if ent in stopwords or ent.lower() in {"page", "comic", "context", "chapter"}:
#             continue
#         if ent not in entities:
#             entities.append(ent)
#     entities.sort(key=len, reverse=True)
#     mapping = {}
#     protected = text
#     for i, ent in enumerate(entities):
#         placeholder = f"__PROPER_NAME_{i}__"
#         mapping[placeholder] = ent
#         protected = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected)
#     return protected, mapping


# def restore_proper_names(text: str, mapping: dict[str, str]) -> str:
#     if not text or not mapping:
#         return text
#     restored = text
#     for placeholder, original in mapping.items():
#         restored = restored.replace(placeholder, original)
#     return restored


# def transliterate_to_roman_urdu(urdu_text: str) -> str:
#     if not urdu_text or not urdu_text.strip():
#         return urdu_text
#     prompt = (
#         "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
#         "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only.\n"
#         "RULES:\n"
#         "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
#         "2. Preserve all character names, proper nouns EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
#         "3. Output ONLY the transliterated text."
#     )
#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[{"role": "system", "content": prompt}, {"role": "user", "content": urdu_text}],
#         temperature=0.1,
#         max_tokens=1500
#     )
#     return response.choices[0].message.content.strip()


# def translate_with_library(english_text: str, original_question: str) -> str:
#     if not english_text or not english_text.strip():
#         return english_text
#     try:
#         protected_text, name_map = extract_and_protect_proper_names(english_text)
#         if is_urdu_script_query(original_question):
#             translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
#             restored_ur = restore_proper_names(translated_ur, name_map)
#             return clean_llm_response(restored_ur, question=original_question)
#         urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
#         roman_urdu = transliterate_to_roman_urdu(urdu_script)
#         restored_roman = restore_proper_names(roman_urdu, name_map)
#         return clean_llm_response(restored_roman, question=original_question)
#     except Exception as e:
#         print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
#         return translate_answer(english_text, original_question)


# def translate_answer(english_answer: str, original_question: str) -> str:
#     translation_system_prompt = (
#         "You are a precise, natural translator. Translate the text faithfully "
#         "matching the exact language, script/alphabet, and style of the user's question."
#     )
#     translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.
# CRITICAL RULES:
# 1. SCRIPT MATCHING: If the USER'S ORIGINAL QUESTION is in Roman Urdu, write entirely in Latin letters (A-Z). NEVER use Arabic/Urdu script (اردو).
# 2. PRESERVE NAMES: Keep all character names (e.g. 'Victor', 'Cynthia') EXACTLY in their English spelling.
# 3. NO COMMENTARY: Output ONLY the translated text.

# USER'S ORIGINAL QUESTION:
# {original_question}

# ENGLISH TEXT TO TRANSLATE:
# {english_answer}"""
#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[{"role": "system", "content": translation_system_prompt}, {"role": "user", "content": translation_user_prompt}],
#         temperature=0.2,
#         max_tokens=1500
#     )
#     translated = response.choices[0].message.content.strip()
#     return clean_llm_response(translated, question=original_question)


# def clean_llm_response(text: str, question: str = "") -> str:
#     if not text or not text.strip():
#         return text
#     cleaned = text.strip()
#     cleaned = re.sub(r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
#     cleaned = re.sub(r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
#     q_lower = question.lower()
#     if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
#         missing_section_pattern = re.compile(r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing aspects)(?:\*\*)?:?.*$", re.IGNORECASE | re.DOTALL)
#         cleaned = missing_section_pattern.sub("", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)
#     bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", cleaned))
#     if len(bold_matches) > 2:
#         count = 0
#         def unbold_excess(match):
#             nonlocal count
#             count += 1
#             if count <= 2:
#                 return match.group(0)
#             return match.group(1)
#         cleaned = re.sub(r"\*\*(.*?)\*\*", unbold_excess, cleaned)
#     return cleaned.strip()


# BROAD_SUMMARY_REGEX = re.compile(r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|describe this page|explain this page)\b|\b(kya hua|kya ho raha|poori kahani|puri kahani|sari kahani|sab batao)\b", re.IGNORECASE)
# SHORT_REQUEST_REGEX = re.compile(r"\b(short|shortly|brief|briefly|in short|quick|quickly|one line|two lines|tldr)\b|\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|aik\s+line|do\s+line)\b", re.IGNORECASE)
# DETAIL_REQUEST_REGEX = re.compile(r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details)\b|\b(poora\s+batao|pura\s+batao|puri\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao)\b", re.IGNORECASE)


# def is_explicit_short_query(question: str) -> bool:
#     return bool(question and SHORT_REQUEST_REGEX.search(question))

# def is_explicit_detail_query(question: str) -> bool:
#     return bool(question and DETAIL_REQUEST_REGEX.search(question))

# def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
#     if is_explicit_short_query(question):
#         return 250
#     if is_explicit_detail_query(question):
#         return 1800
#     if current_page is not None:
#         return 1500
#     if question and BROAD_SUMMARY_REGEX.search(question):
#         return 1500
#     return 700


# # -----------------------------
# # Generate Answer
# # -----------------------------

# def generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> str:
#     if not question or not question.strip():
#         raise ValueError("Question cannot be empty.")
#     if not context or not context.strip():
#         return "I could not find relevant information in the comic."

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     response = _safe_chat_complete(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     raw_answer = response.choices[0].message.content.strip()
#     cleaned = clean_llm_response(raw_answer, question=question)
#     return cleaned


# # -----------------------------
# # Prompt Builder
# # -----------------------------
# def _build_streaming_prompts(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> tuple[str, str]:
#     """Helper to assemble system and user prompts optimized for structured JSON data."""
#     is_urdu_script = is_urdu_script_query(question)
#     is_english = is_english_query(question) and not is_urdu_script
#     is_roman_urdu = not is_english and not is_urdu_script

#     if is_urdu_script:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
#     elif is_roman_urdu:
#         lang_reminder = (
#             "\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu. "
#             "You MUST write your entire response in authentic, conversational Pakistani Roman Urdu. "
#             "Speak like a normal human explaining a story. Use proper grammar and flowing sentences. "
#             "Do NOT use Hindi words. Do NOT write blunt one-liners.)"
#         )
#     else:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

#     if is_explicit_short_query(question):
#         formatting_reminder = (
#             "\n\n(CRITICAL FORMATTING: Provide a brief, direct response of EXACTLY 1-2 sentences. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )
#     elif is_explicit_detail_query(question):
#         formatting_reminder = (
#             "\n\n(FORMATTING: Provide a clear, detailed narrative. "
#             "Explain the specific moment beautifully based on the context. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )
#     else:
#         formatting_reminder = (
#             "\n\n(FORMATTING: Write a natural, conversational response (about 2 to 3 sentences). "
#             "Do not just give a blunt one-liner. Explain the context naturally like a storyteller. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )

#     page_hint = ""
#     if current_page is not None:
#         try:
#             from app.services.rag_qa import is_page_scoped_query
#             is_page_specific = is_page_scoped_query(question, current_page)
#         except Exception:
#             is_page_specific = False

#         if is_page_specific:
#             page_hint = (
#                 f"\nCURRENT USER VIEWING PAGE: Page {current_page}\n"
#                 f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on evidence from Page {current_page}.)\n"
#             )
#         else:
#             page_hint = (
#                 f"\n(Note: The user is currently viewing Page {current_page}, but synthesizing the answer from all provided context.)\n"
#             )

#     if conversation_history:
#         history_lines = []
#         for msg in conversation_history:
#             role_label = "User" if msg.get("role") == "user" else "Assistant"
#             content = msg.get("content", "").strip()
#             if content:
#                 history_lines.append(f"{role_label}: {content}")
#         history_text = "\n".join(history_lines) if history_lines else "None"
#         user_prompt = f"""CONVERSATION HISTORY:
# {history_text}
# {page_hint}
# RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """
#     else:
#         user_prompt = f"""{page_hint}RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """

#     # Yahan "Good vs Bad" example diya gaya hai taake AI theek se seekh jaye
#     system_prompt = (
#         "Strickly follow the text "
#         "You are a helpful comic book companion. Your ONLY factual grounding source is the provided RICH COMIC CONTEXT.\n\n"
#         "==================================================\n"
#         "CORE ANSWERING RULES:\n"
#         "==================================================\n"
#         "1. STRICT RELEVANCE: Answer ONLY what is specifically asked. Do not bring up later events or other panels unnecessarily (e.g., if asked about splashing water, do NOT talk about the sea serpent unless asked).\n"
#         "2. CONCISE & DIRECT: Keep answers focused in 1 to 2 clear sentences. Do not drag out the answer.\n"
#         "3. NO ROBOTIC/INVENTED PHRASES: Never use broken phrases like 'jhoot ghoomne' or wrong grammar like 'Victor aur Valeria ne khel rahe the'. Use correct Urdu: 'jhoot bolne ka socha', 'Victor aur Valeria khel rahe the'.\n"
#         "4. NO ASTERISKS: Never use markdown bold (**) or asterisks anywhere in the output.\n\n"
#         "==================================================\n"
#         "STRICT GROUNDING RULE\n"
#         "==================================================\n"
#         "Rely ONLY on the provided context. If the answer is not in the context, reply exactly: 'I could not find relevant information in the comic.'"
#     )


#     return system_prompt, user_prompt


# async def stream_generate_answer_async(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = await _safe_chat_stream_async(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     async for chunk in stream_resp:
#         delta = chunk.data.choices[0].delta.content
#         if delta:
#             yield delta


# def stream_generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = _safe_chat_stream(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget,
#         frequency_penalty=0.3,
#         presence_penalty=0.2
#     )

#     for chunk in stream_resp:
#         delta = chunk.data.choices[0].delta.content
#         if delta:
#             yield delta

# """
# LLM Service - EDEN AI VERSION

# Generates grounded comic answers using Eden AI (Gemini), enforcing strict grounding rules,
# pinned fallback phrase, and optional conversation memory context.
# Updated to fix the "over-excited fanboy" tone and make it natural and concise.
# """
# import asyncio
# import random
# import re
# import time
# import requests
# import aiohttp
# import json

# from deep_translator import GoogleTranslator

# # Agar aapke paas USE_LIBRARY_TRANSLATION wagerah config mein hai toh theek hai, 
# # warna aap Mistral wali cheezein config se hata sakte hain.
# from app.core.config import USE_LIBRARY_TRANSLATION

# # ==========================================
# # EDEN AI SETTINGS
# # ==========================================
# EDEN_API_KEY = "sk-eden-live-nlQJ9hMh0QdIsis1wCn2_hUzri4BFkWFnCw2U2RheNI1491a31a"  # YAHAN APNI EDEN AI KEY DAALAIN
# EDEN_URL = "https://api.edenai.run/v3/chat/completions"
# EDEN_MODEL = "google/gemma-4-31b-it"  


# def _safe_chat_complete(
#     messages: list[dict],
#     model: str = EDEN_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 5000,
#     max_retries: int = 5,
#     **kwargs
# ) -> str:
#     """
#     Executes a standard POST request to Eden AI chat completions with retry logic.
#     Returns the text response directly.
#     """
#     headers = {
#         "Authorization": f"Bearer {EDEN_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "stream": False
#     }
    
#     for attempt in range(max_retries + 1):
#         try:
#             resp = requests.post(EDEN_URL, headers=headers, json=payload, timeout=60)
#             resp.raise_for_status()
#             data = resp.json()
#             return data["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             if attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 print(f"Eden API Error (Sync): {e}")
#                 return "I could not find relevant information in the comic."


# def _safe_chat_stream(
#     messages: list[dict],
#     model: str = EDEN_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     """
#     Executes a synchronous streaming request to Eden AI.
#     Yields chunks of text.
#     """
#     headers = {
#         "Authorization": f"Bearer {EDEN_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "stream": True
#     }
    
#     for attempt in range(max_retries + 1):
#         try:
#             with requests.post(EDEN_URL, headers=headers, json=payload, stream=True, timeout=60) as response:
#                 response.raise_for_status()
#                 for line in response.iter_lines():
#                     if line:
#                         line = line.decode('utf-8').strip()
#                         if line.startswith("data: ") and line != "data: [DONE]":
#                             try:
#                                 data = json.loads(line[6:])
#                                 delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
#                                 if delta:
#                                     yield delta
#                             except json.JSONDecodeError:
#                                 pass
#             break
#         except Exception as e:
#             if attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 time.sleep(wait_time)
#             else:
#                 raise e


# async def _safe_chat_stream_async(
#     messages: list[dict],
#     model: str = EDEN_MODEL,
#     temperature: float = 0.3,
#     max_tokens: int = 1800,
#     max_retries: int = 5,
#     **kwargs
# ):
#     """
#     Executes an asynchronous streaming request to Eden AI.
#     Yields chunks of text.
#     """
#     headers = {
#         "Authorization": f"Bearer {EDEN_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "stream": True
#     }
    
#     for attempt in range(max_retries + 1):
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(EDEN_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
#                     response.raise_for_status()
#                     async for line in response.content:
#                         line = line.decode('utf-8').strip()
#                         if line.startswith("data: ") and line != "data: [DONE]":
#                             try:
#                                 data = json.loads(line[6:])
#                                 delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
#                                 if delta:
#                                     yield delta
#                             except json.JSONDecodeError:
#                                 pass
#             break
#         except Exception as e:
#             if attempt < max_retries:
#                 wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise e


# ENGLISH_FUNCTION_WORDS = {
#     "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
#     "is", "are", "was", "were", "am", "be", "been", "being",
#     "do", "does", "did", "done",
#     "have", "has", "had", "having",
#     "can", "could", "would", "should", "will", "shall", "might", "must",
#     "the", "a", "an",
#     "in", "on", "at", "to", "for", "from", "with", "about", "by", "of", "into", "through", "after", "before",
#     "and", "or", "but", "if", "because", "as", "than", "so",
#     "he", "she", "it", "they", "him", "her", "his", "their", "them", "my", "your", "our", "its",
#     "tell", "me", "explain", "give", "show", "describe", "summarize", "find", "list",
#     "this", "that", "these", "those", "there", "here", "any", "some", "all",
#     "plot", "story", "character", "characters", "page", "pages", "book", "comic"
# }

# NON_ENGLISH_MARKERS = {
#     "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
#     "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
#     "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
#     "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
#     "mera", "meri", "mere", "apna", "apni", "apne",
#     "mujhe", "tum", "aap", "humein", "hum",
#     "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
#     "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
#     "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
#     "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
#     "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
# }


# def is_english_query(query: str) -> bool:
#     if not query or not query.strip():
#         return True
#     non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
#     if non_latin:
#         return False
#     words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
#     if not words:
#         return True
#     for w in words:
#         if w in NON_ENGLISH_MARKERS:
#             return False
#     english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
#     if english_word_count > 0:
#         return True
#     return True


# def is_urdu_script_query(query: str) -> bool:
#     if not query:
#         return False
#     return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


# def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
#     if not text:
#         return text, {}
#     stopwords = {
#         "On", "In", "The", "A", "An", "This", "That", "These", "Those", "Page", "Based",
#         "According", "When", "If", "While", "After", "Before", "He", "She", "They", "It",
#         "His", "Her", "Their", "As", "At", "By", "For", "From", "With", "To", "Comic", "Context",
#         "Current", "Question", "Answer", "Yes", "No", "Not", "All", "Some", "Any"
#     }
#     pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:Von|De|La|van|von|der|of)\s+[A-Z][a-zA-Z]*|\s+[A-Z][a-zA-Z]*)*\b"
#     matches = list(re.finditer(pattern, text))
#     entities = []
#     for m in matches:
#         ent = m.group(0).strip()
#         if ent in stopwords or ent.lower() in {"page", "comic", "context", "chapter"}:
#             continue
#         if ent not in entities:
#             entities.append(ent)
#     entities.sort(key=len, reverse=True)
#     mapping = {}
#     protected = text
#     for i, ent in enumerate(entities):
#         placeholder = f"__PROPER_NAME_{i}__"
#         mapping[placeholder] = ent
#         protected = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected)
#     return protected, mapping


# def restore_proper_names(text: str, mapping: dict[str, str]) -> str:
#     if not text or not mapping:
#         return text
#     restored = text
#     for placeholder, original in mapping.items():
#         restored = restored.replace(placeholder, original)
#     return restored


# def transliterate_to_roman_urdu(urdu_text: str) -> str:
#     if not urdu_text or not urdu_text.strip():
#         return urdu_text
#     prompt = (
#         "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
#         "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only.\n"
#         "RULES:\n"
#         "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
#         "2. Preserve all character names, proper nouns EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
#         "3. Output ONLY the transliterated text."
#     )
#     # Eden API returns a string now
#     response_text = _safe_chat_complete(
#         model=EDEN_MODEL,
#         messages=[{"role": "system", "content": prompt}, {"role": "user", "content": urdu_text}],
#         temperature=0.1,
#         max_tokens=1500
#     )
#     return response_text


# def translate_with_library(english_text: str, original_question: str) -> str:
#     if not english_text or not english_text.strip():
#         return english_text
#     try:
#         protected_text, name_map = extract_and_protect_proper_names(english_text)
#         if is_urdu_script_query(original_question):
#             translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
#             restored_ur = restore_proper_names(translated_ur, name_map)
#             return clean_llm_response(restored_ur, question=original_question)
#         urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
#         roman_urdu = transliterate_to_roman_urdu(urdu_script)
#         restored_roman = restore_proper_names(roman_urdu, name_map)
#         return clean_llm_response(restored_roman, question=original_question)
#     except Exception as e:
#         print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
#         return translate_answer(english_text, original_question)


# def translate_answer(english_answer: str, original_question: str) -> str:
#     translation_system_prompt = (
#         "You are a precise, natural translator. Translate the text faithfully "
#         "matching the exact language, script/alphabet, and style of the user's question."
#     )
#     translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.
# CRITICAL RULES:
# 1. SCRIPT MATCHING: If the USER'S ORIGINAL QUESTION is in Roman Urdu, write entirely in Latin letters (A-Z). NEVER use Arabic/Urdu script (اردو).
# 2. PRESERVE NAMES: Keep all character names (e.g. 'Victor', 'Cynthia') EXACTLY in their English spelling.
# 3. NO COMMENTARY: Output ONLY the translated text.

# USER'S ORIGINAL QUESTION:
# {original_question}

# ENGLISH TEXT TO TRANSLATE:
# {english_answer}"""
    
#     # Eden API returns a string now
#     translated_text = _safe_chat_complete(
#         model=EDEN_MODEL,
#         messages=[{"role": "system", "content": translation_system_prompt}, {"role": "user", "content": translation_user_prompt}],
#         temperature=0.2,
#         max_tokens=1500
#     )
#     return clean_llm_response(translated_text, question=original_question)


# def clean_llm_response(text: str, question: str = "") -> str:
#     if not text or not text.strip():
#         return text
#     cleaned = text.strip()
#     cleaned = re.sub(r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
#     cleaned = re.sub(r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
#     q_lower = question.lower()
#     if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
#         missing_section_pattern = re.compile(r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing aspects)(?:\*\*)?:?.*$", re.IGNORECASE | re.DOTALL)
#         cleaned = missing_section_pattern.sub("", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
#     cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)
#     bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", cleaned))
#     if len(bold_matches) > 2:
#         count = 0
#         def unbold_excess(match):
#             nonlocal count
#             count += 1
#             if count <= 2:
#                 return match.group(0)
#             return match.group(1)
#         cleaned = re.sub(r"\*\*(.*?)\*\*", unbold_excess, cleaned)
#     return cleaned.strip()


# BROAD_SUMMARY_REGEX = re.compile(r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|describe this page|explain this page)\b|\b(kya hua|kya ho raha|poori kahani|puri kahani|sari kahani|sab batao)\b", re.IGNORECASE)
# SHORT_REQUEST_REGEX = re.compile(r"\b(short|shortly|brief|briefly|in short|quick|quickly|one line|two lines|tldr)\b|\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|aik\s+line|do\s+line)\b", re.IGNORECASE)
# DETAIL_REQUEST_REGEX = re.compile(r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details)\b|\b(poora\s+batao|pura\s+batao|puri\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao)\b", re.IGNORECASE)


# def is_explicit_short_query(question: str) -> bool:
#     return bool(question and SHORT_REQUEST_REGEX.search(question))

# def is_explicit_detail_query(question: str) -> bool:
#     return bool(question and DETAIL_REQUEST_REGEX.search(question))

# def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
#     if is_explicit_short_query(question):
#         return 250
#     if is_explicit_detail_query(question):
#         return 1800
#     if current_page is not None:
#         return 1500
#     if question and BROAD_SUMMARY_REGEX.search(question):
#         return 1500
#     return 700


# # -----------------------------
# # Generate Answer
# # -----------------------------

# def generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> str:
#     if not question or not question.strip():
#         raise ValueError("Question cannot be empty.")
#     if not context or not context.strip():
#         return "I could not find relevant information in the comic."

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     raw_answer = _safe_chat_complete(
#         model=EDEN_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget
#     )

#     cleaned = clean_llm_response(raw_answer, question=question)
#     return cleaned


# # -----------------------------
# # Prompt Builder
# # -----------------------------
# def _build_streaming_prompts(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ) -> tuple[str, str]:
#     """Helper to assemble system and user prompts optimized for structured JSON data."""
#     is_urdu_script = is_urdu_script_query(question)
#     is_english = is_english_query(question) and not is_urdu_script
#     is_roman_urdu = not is_english and not is_urdu_script

#     if is_urdu_script:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
#     elif is_roman_urdu:
#         lang_reminder = (
#             "\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu. "
#             "You MUST write your entire response in authentic, conversational Pakistani Roman Urdu. "
#             "Speak like a normal human explaining a story. Use proper grammar and flowing sentences. "
#             "Do NOT use Hindi words. Do NOT write blunt one-liners.)"
#         )
#     else:
#         lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

#     if is_explicit_short_query(question):
#         formatting_reminder = (
#             "\n\n(CRITICAL FORMATTING: Provide a brief, direct response of EXACTLY 1-2 sentences. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )
#     elif is_explicit_detail_query(question):
#         formatting_reminder = (
#             "\n\n(FORMATTING: Provide a clear, detailed narrative. "
#             "Explain the specific moment beautifully based on the context. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )
#     else:
#         formatting_reminder = (
#             "\n\n(FORMATTING: Write a natural, conversational response (about 2 to 3 sentences). "
#             "Do not just give a blunt one-liner. Explain the context naturally like a storyteller. "
#             "DO NOT use markdown, asterisks (**), or bold text formatting.)"
#         )

#     page_hint = ""
#     if current_page is not None:
#         try:
#             from app.services.rag_qa import is_page_scoped_query
#             is_page_specific = is_page_scoped_query(question, current_page)
#         except Exception:
#             is_page_specific = False

#         if is_page_specific:
#             page_hint = (
#                 f"\nCURRENT USER VIEWING PAGE: Page {current_page}\n"
#                 f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on evidence from Page {current_page}.)\n"
#             )
#         else:
#             page_hint = (
#                 f"\n(Note: The user is currently viewing Page {current_page}, but synthesizing the answer from all provided context.)\n"
#             )

#     if conversation_history:
#         history_lines = []
#         for msg in conversation_history:
#             role_label = "User" if msg.get("role") == "user" else "Assistant"
#             content = msg.get("content", "").strip()
#             if content:
#                 history_lines.append(f"{role_label}: {content}")
#         history_text = "\n".join(history_lines) if history_lines else "None"
#         user_prompt = f"""CONVERSATION HISTORY:
# {history_text}
# {page_hint}
# RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """
#     else:
#         user_prompt = f"""{page_hint}RICH COMIC CONTEXT (Extracted JSON Data):
# {context}

# CURRENT QUESTION:
# {question}{lang_reminder}{formatting_reminder}
# """

#     system_prompt = (
#         "Strickly follow the text "
#         "You are a helpful comic book companion. Your ONLY factual grounding source is the provided RICH COMIC CONTEXT.\n\n"
#         "==================================================\n"
#         "CORE ANSWERING RULES:\n"
#         "==================================================\n"
#         "1. STRICT RELEVANCE: Answer ONLY what is specifically asked. Do not bring up later events or other panels unnecessarily (e.g., if asked about splashing water, do NOT talk about the sea serpent unless asked).\n"
#         "2. CONCISE & DIRECT: Keep answers focused in 1 to 2 clear sentences. Do not drag out the answer.\n"
#         "3. NO ROBOTIC/INVENTED PHRASES: Never use broken phrases like 'jhoot ghoomne' or wrong grammar like 'Victor aur Valeria ne khel rahe the'. Use correct Urdu: 'jhoot bolne ka socha', 'Victor aur Valeria khel rahe the'.\n"
#         "4. NO ASTERISKS: Never use markdown bold (**) or asterisks anywhere in the output.\n\n"
#         "==================================================\n"
#         "STRICT GROUNDING RULE\n"
#         "==================================================\n"
#         "Rely ONLY on the provided context. If the answer is not in the context, reply exactly: 'I could not find relevant information in the comic.'"
#     )

#     return system_prompt, user_prompt


# async def stream_generate_answer_async(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = _safe_chat_stream_async(
#         model=EDEN_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget
#     )

#     # _safe_chat_stream_async yields text directly now
#     async for chunk in stream_resp:
#         yield chunk


# def stream_generate_answer(
#     question: str,
#     context: str,
#     conversation_history: list[dict] | None = None,
#     current_page: int | None = None
# ):
#     if not question or not question.strip():
#         yield "Please provide a valid question."
#         return
#     if not context or not context.strip():
#         yield "I could not find relevant information in the comic."
#         return

#     system_prompt, user_prompt = _build_streaming_prompts(
#         question=question,
#         context=context,
#         conversation_history=conversation_history,
#         current_page=current_page
#     )
#     token_budget = get_dynamic_max_tokens(question, current_page)

#     stream_resp = _safe_chat_stream(
#         model=EDEN_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.2,
#         max_tokens=token_budget
#     )

#     # _safe_chat_stream yields text directly now
#     for chunk in stream_resp:
#         yield chunk


"""
LLM Service - EDEN AI VERSION

Generates grounded comic answers using Eden AI (Gemini), enforcing strict grounding rules,
pinned fallback phrase, and optional conversation memory context.
Updated to fix the "over-excited fanboy" tone and make it natural and concise.
"""
import asyncio
import random
import re
import time
import requests
import aiohttp
import json
import os

from deep_translator import GoogleTranslator

# Agar aapke paas USE_LIBRARY_TRANSLATION wagerah config mein hai toh theek hai, 
# warna aap Mistral wali cheezein config se hata sakte hain.
from app.core.config import USE_LIBRARY_TRANSLATION

# ==========================================
# EDEN AI SETTINGS
# ==========================================
EDEN_KEYS = [
    os.getenv(f"EDEN_KEY_{i}") for i in range(1, 5)
    if os.getenv(f"EDEN_KEY_{i}")
]
if not EDEN_KEYS:
    fallback_k = os.getenv("EDEN_API_KEY", "sk-eden-live-nlQJ9hMh0QdIsis1wCn2_hUzri4BFkWFnCw2U2RheNI1491a31a")
    EDEN_KEYS = [fallback_k]

_eden_key_idx = 0

def get_active_eden_key() -> str:
    global _eden_key_idx
    return EDEN_KEYS[_eden_key_idx % len(EDEN_KEYS)]

def rotate_eden_key() -> str:
    global _eden_key_idx
    _eden_key_idx = (_eden_key_idx + 1) % len(EDEN_KEYS)
    return get_active_eden_key()

EDEN_URL = os.getenv("EDEN_API_URL", "https://api.edenai.run/v3/chat/completions")
EDEN_MODEL = os.getenv("EDEN_MODEL", "google/gemma-4-26b-a4b-it")


def _safe_chat_complete(
    messages: list[dict],
    model: str = EDEN_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_retries: int = 4,
    **kwargs
) -> str:
    """
    Executes a standard POST request to Eden AI chat completions with retry logic.
    Returns the text response directly.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    for attempt in range(max_retries + 1):
        key = get_active_eden_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(EDEN_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices and isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get("message") if isinstance(choices[0], dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if content is not None:
                    return str(content).strip()
            return ""
        except Exception as e:
            if attempt < max_retries:
                rotate_eden_key()
                wait_time = (2 ** attempt) * 1.5 + random.uniform(0.3, 0.8)
                time.sleep(wait_time)
            else:
                print(f"Eden API Error (Sync): {e}")
                return ""


def _safe_chat_stream(
    messages: list[dict],
    model: str = EDEN_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_retries: int = 4,
    **kwargs
):
    """
    Executes a synchronous streaming request to Eden AI.
    Yields chunks of text in real time.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    for attempt in range(max_retries + 1):
        key = get_active_eden_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        try:
            with requests.post(EDEN_URL, headers=headers, json=payload, stream=True, timeout=60) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            json_str = line_str[6:].strip()
                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
            break
        except Exception as e:
            if attempt < max_retries:
                rotate_eden_key()
                wait_time = (2 ** attempt) * 1.5 + random.uniform(0.3, 0.8)
                time.sleep(wait_time)
            else:
                raise e


async def _safe_chat_stream_async(
    messages: list[dict],
    model: str = EDEN_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_retries: int = 4,
    **kwargs
):
    """
    Executes an asynchronous streaming request to Eden AI using httpx.AsyncClient.
    Yields text content tokens immediately in real time as they arrive.
    """
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    for attempt in range(max_retries + 1):
        key = get_active_eden_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)) as client:
                async with client.stream("POST", EDEN_URL, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            json_str = line_str[6:].strip()
                            try:
                                data = json.loads(json_str)
                                choices = data.get("choices", [])
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except (json.JSONDecodeError, KeyError):
                                pass
            return
        except Exception as e:
            if attempt < max_retries:
                rotate_eden_key()
                wait_time = (2 ** attempt) * 1.5 + random.uniform(0.3, 0.8)
                print(f"[EDEN STREAM ASYNC] Retrying ({attempt+1}/{max_retries}) with next key in {wait_time:.1f}s due to {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"[EDEN STREAM ASYNC] Error after retries: {e}")
                raise e

ENGLISH_FUNCTION_WORDS = {
    "what", "who", "whom", "whose", "where", "when", "why", "which", "how",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had", "having",
    "can", "could", "would", "should", "will", "shall", "might", "must",
    "the", "a", "an",
    "in", "on", "at", "to", "for", "from", "with", "about", "by", "of", "into", "through", "after", "before",
    "and", "or", "but", "if", "because", "as", "than", "so",
    "he", "she", "it", "they", "him", "her", "his", "their", "them", "my", "your", "our", "its",
    "tell", "me", "explain", "give", "show", "describe", "summarize", "find", "list",
    "this", "that", "these", "those", "there", "here", "any", "some", "all",
    "plot", "story", "character", "characters", "page", "pages", "book", "comic"
}

NON_ENGLISH_MARKERS = {
    "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
    "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
    "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
    "mera", "meri", "mere", "apna", "apni", "apne",
    "mujhe", "tum", "aap", "humein", "hum",
    "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
    "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
    "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
    "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
}


def is_english_query(query: str) -> bool:
    if not query or not query.strip():
        return True
    non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
    if non_latin:
        return False
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
    if not words:
        return True
    for w in words:
        if w in NON_ENGLISH_MARKERS:
            return False
    english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    if english_word_count > 0:
        return True
    return True


def is_urdu_script_query(query: str) -> bool:
    if not query:
        return False
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
    if not text:
        return text, {}
    stopwords = {
        "On", "In", "The", "A", "An", "This", "That", "These", "Those", "Page", "Based",
        "According", "When", "If", "While", "After", "Before", "He", "She", "They", "It",
        "His", "Her", "Their", "As", "At", "By", "For", "From", "With", "To", "Comic", "Context",
        "Current", "Question", "Answer", "Yes", "No", "Not", "All", "Some", "Any"
    }
    pattern = r"\b[A-Z][a-zA-Z]*(?:\s+(?:Von|De|La|van|von|der|of)\s+[A-Z][a-zA-Z]*|\s+[A-Z][a-zA-Z]*)*\b"
    matches = list(re.finditer(pattern, text))
    entities = []
    for m in matches:
        ent = m.group(0).strip()
        if ent in stopwords or ent.lower() in {"page", "comic", "context", "chapter"}:
            continue
        if ent not in entities:
            entities.append(ent)
    entities.sort(key=len, reverse=True)
    mapping = {}
    protected = text
    for i, ent in enumerate(entities):
        placeholder = f"__PROPER_NAME_{i}__"
        mapping[placeholder] = ent
        protected = re.sub(r"\b" + re.escape(ent) + r"\b", placeholder, protected)
    return protected, mapping


def restore_proper_names(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    restored = text
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    return restored


def transliterate_to_roman_urdu(urdu_text: str) -> str:
    if not urdu_text or not urdu_text.strip():
        return urdu_text
    prompt = (
        "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
        "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only.\n"
        "RULES:\n"
        "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
        "2. Preserve all character names, proper nouns EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
        "3. Output ONLY the transliterated text."
    )
    response_text = _safe_chat_complete(
        model=EDEN_MODEL,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": urdu_text}],
        temperature=0.1,
        max_tokens=1500
    )
    return response_text


def translate_with_library(english_text: str, original_question: str) -> str:
    if not english_text or not english_text.strip():
        return english_text
    try:
        protected_text, name_map = extract_and_protect_proper_names(english_text)
        if is_urdu_script_query(original_question):
            translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
            restored_ur = restore_proper_names(translated_ur, name_map)
            return clean_llm_response(restored_ur, question=original_question)
        urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
        roman_urdu = transliterate_to_roman_urdu(urdu_script)
        restored_roman = restore_proper_names(roman_urdu, name_map)
        return clean_llm_response(restored_roman, question=original_question)
    except Exception as e:
        print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
        return translate_answer(english_text, original_question)


def translate_answer(english_answer: str, original_question: str) -> str:
    translation_system_prompt = (
        "You are a precise, natural translator. Translate the text faithfully "
        "matching the exact language, script/alphabet, and style of the user's question."
    )
    translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.
CRITICAL RULES:
1. SCRIPT MATCHING: If the USER'S ORIGINAL QUESTION is in Roman Urdu, write entirely in Latin letters (A-Z). NEVER use Arabic/Urdu script (اردو).
2. PRESERVE NAMES: Keep all character names (e.g. 'Victor', 'Cynthia') EXACTLY in their English spelling.
3. NO COMMENTARY: Output ONLY the translated text.

USER'S ORIGINAL QUESTION:
{original_question}

ENGLISH TEXT TO TRANSLATE:
{english_answer}"""
    
    translated_text = _safe_chat_complete(
        model=EDEN_MODEL,
        messages=[{"role": "system", "content": translation_system_prompt}, {"role": "user", "content": translation_user_prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    return clean_llm_response(translated_text, question=original_question)


def clean_llm_response(text: str, question: str = "") -> str:
    if not text or not text.strip():
        return text
    cleaned = text.strip()
    cleaned = re.sub(r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    q_lower = question.lower()
    if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
        missing_section_pattern = re.compile(r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing aspects)(?:\*\*)?:?.*$", re.IGNORECASE | re.DOTALL)
        cleaned = missing_section_pattern.sub("", cleaned)
    cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)
    bold_matches = list(re.finditer(r"\*\*(.*?)\*\*", cleaned))
    if len(bold_matches) > 2:
        count = 0
        def unbold_excess(match):
            nonlocal count
            count += 1
            if count <= 2:
                return match.group(0)
            return match.group(1)
        cleaned = re.sub(r"\*\*(.*?)\*\*", unbold_excess, cleaned)
    return cleaned.strip()


BROAD_SUMMARY_REGEX = re.compile(r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|describe this page|explain this page)\b|\b(kya hua|kya ho raha|poori kahani|puri kahani|sari kahani|sab batao)\b", re.IGNORECASE)
SHORT_REQUEST_REGEX = re.compile(r"\b(short|shortly|brief|briefly|in short|quick|quickly|one line|two lines|tldr)\b|\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|aik\s+line|do\s+line)\b", re.IGNORECASE)
DETAIL_REQUEST_REGEX = re.compile(r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details)\b|\b(poora\s+batao|pura\s+batao|puri\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao)\b", re.IGNORECASE)


def is_explicit_short_query(question: str) -> bool:
    return bool(question and SHORT_REQUEST_REGEX.search(question))

def is_explicit_detail_query(question: str) -> bool:
    return bool(question and DETAIL_REQUEST_REGEX.search(question))

# def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
#     if is_explicit_short_query(question):
#         return 250
#     if is_explicit_detail_query(question):
#         return 1800
#     if current_page is not None:
#         return 1500
#     if question and BROAD_SUMMARY_REGEX.search(question):
#         return 1500
#     return 700

def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
    # Model ki reasoning phase ke liye tokens barha diye gaye hain
    if is_explicit_short_query(question):
        return 800  # Pehle 250 tha, ab reasoning ke liye space di hai
    if is_explicit_detail_query(question):
        return 2500 # Pehle 1800 tha
    if current_page is not None:
        return 2000
    if question and BROAD_SUMMARY_REGEX.search(question):
        return 2000
    return 1500


# -----------------------------
# Query Reformulation (Coreference Resolution)
# -----------------------------

QUERY_OPTIMIZER_SYSTEM_PROMPT = (
    "You are a Query Reformulator, NOT an answering assistant. DO NOT answer the user's question. "
    "Your ONLY job is to resolve pronouns using the chat history and rewrite the question. "
    "If no pronouns exist or no rewrite is needed, you MUST output the exact original question. "
    "NEVER apologize, NEVER say 'I cannot answer', just output the query."
)


def reformulate_query(query: str, chat_history: str) -> str:
    """
    Rewrites user query to resolve pronouns and coreferences into a clear, standalone question
    grounded in the provided conversation history.
    """
    if not query or not query.strip():
        return query

    original_user_query = query.strip()
    if not chat_history or not chat_history.strip():
        return original_user_query

    user_prompt = f"Chat History:\n{chat_history.strip()}\n\nUser's Latest Question:\n{original_user_query}"
    reformulated_query = original_user_query

    try:
        raw_response = _safe_chat_complete(
            model=EDEN_MODEL,
            messages=[
                {"role": "system", "content": QUERY_OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        # Safe extraction & check if text before calling strip()
        text = raw_response
        if isinstance(text, dict):
            text = text.get("text") or text.get("content") or ""

        if text:
            cleaned = str(text).strip().strip('"`\'')
            if cleaned:
                reformulated_query = cleaned
    except Exception as e:
        print(f"[QUERY REFORMULATION] Failed to reformulate query, using original: {e}")
        reformulated_query = original_user_query

    # Output Validation (Sanity Check):
    bad_phrases = ["i could not find", "i cannot", "i don't know", "i am sorry", "does not contain"]
    if any(phrase in reformulated_query.lower() for phrase in bad_phrases):
        print(f"[QUERY REFORMULATION] Detected bad phrase in reformulated query '{reformulated_query}', falling back to original.")
        reformulated_query = original_user_query

    if reformulated_query != original_user_query:
        print(f"[QUERY REFORMULATION] '{original_user_query}' -> '{reformulated_query}'")

    return reformulated_query


async def reformulate_query_async(query: str, chat_history: str) -> str:
    """
    Asynchronous version of reformulate_query for use in async streaming endpoints.
    """
    if not query or not query.strip():
        return query

    clean_query = query.strip()
    if not chat_history or not chat_history.strip():
        return clean_query

    try:
        import anyio
        return await anyio.to_thread.run_sync(reformulate_query, clean_query, chat_history)
    except Exception:
        import asyncio
        return await asyncio.to_thread(reformulate_query, clean_query, chat_history)


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not context or not context.strip():
        return "I could not find relevant information in the comic."

    system_prompt, user_prompt = _build_streaming_prompts(
        question=question,
        context=context,
        conversation_history=conversation_history,
        current_page=current_page
    )
    token_budget = get_dynamic_max_tokens(question, current_page)

    raw_answer = _safe_chat_complete(
        model=EDEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=token_budget
    )

    cleaned = clean_llm_response(raw_answer, question=question)
    return cleaned


# -----------------------------
# Prompt Builder
# -----------------------------
def _build_streaming_prompts(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
) -> tuple[str, str]:
    """Helper to assemble system and user prompts optimized for structured JSON data."""
    is_urdu_script = is_urdu_script_query(question)
    is_english = is_english_query(question) and not is_urdu_script
    is_roman_urdu = not is_english and not is_urdu_script
    
    # [NEW FIX] Protect against Context Window Overload
    # If the context is massive (like all 26 pages), cap it so the LLM doesn't crash silently
    MAX_CONTEXT_CHARS = 35000 
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[Context truncated due to length limits]..."

    if is_urdu_script:
        lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
    elif is_roman_urdu:
        lang_reminder = (
            "\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu. "
            "You MUST write your entire response in authentic, conversational Pakistani Roman Urdu. "
            "Speak like a normal human explaining a story. Use proper grammar and flowing sentences. "
            "Do NOT use Hindi words. Do NOT write blunt one-liners.)"
        )
    else:
        lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

    if is_explicit_short_query(question):
        formatting_reminder = (
            "\n\n(CRITICAL FORMATTING: Provide a brief, direct response of EXACTLY 1-2 sentences. "
            "DO NOT use markdown, asterisks (**), or bold text formatting.)"
        )
    elif is_explicit_detail_query(question):
        formatting_reminder = (
            "\n\n(FORMATTING: Provide a clear, detailed narrative. "
            "Explain the specific moment beautifully based on the context. "
            "DO NOT use markdown, asterisks (**), or bold text formatting.)"
        )
    else:
        formatting_reminder = (
            "\n\n(FORMATTING: Write a natural, conversational response (about 2 to 3 sentences). "
            "Do not just give a blunt one-liner. Explain the context naturally like a storyteller. "
            "DO NOT use markdown, asterisks (**), or bold text formatting.)"
        )

    page_hint = ""
    if current_page is not None:
        try:
            from app.services.rag_qa import is_page_scoped_query
            is_page_specific = is_page_scoped_query(question, current_page)
        except Exception:
            is_page_specific = False

        if is_page_specific:
            page_hint = (
                f"\nCURRENT USER VIEWING PAGE: Page {current_page}\n"
                f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on evidence from Page {current_page}.)\n"
            )
        else:
            page_hint = (
                f"\n(Note: The user is currently viewing Page {current_page}, but synthesizing the answer from all provided context.)\n"
            )

    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"{role_label}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "None"
        user_prompt = f"""CONVERSATION HISTORY:
{history_text}
{page_hint}
RICH COMIC CONTEXT (Extracted JSON Data):
{context}

CURRENT QUESTION:
{question}{lang_reminder}{formatting_reminder}
"""
    else:
        user_prompt = f"""{page_hint}RICH COMIC CONTEXT (Extracted JSON Data):
{context}

CURRENT QUESTION:
{question}{lang_reminder}{formatting_reminder}
"""

    system_prompt = (
        "Strickly follow the text "
        "You are a helpful comic book companion. Your ONLY factual grounding source is the provided RICH COMIC CONTEXT.\n\n"
        "==================================================\n"
        "CORE ANSWERING RULES:\n"
        "==================================================\n"
        "1. STRICT RELEVANCE: Answer ONLY what is specifically asked. Do not bring up later events or other panels unnecessarily (e.g., if asked about splashing water, do NOT talk about the sea serpent unless asked).\n"
        "2. CONCISE & DIRECT: Keep answers focused in 1 to 2 clear sentences. Do not drag out the answer.\n"
        "3. NO ROBOTIC/INVENTED PHRASES: Never use broken phrases like 'jhoot ghoomne' or wrong grammar like 'Victor aur Valeria ne khel rahe the'. Use correct Urdu: 'jhoot bolne ka socha', 'Victor aur Valeria khel rahe the'.\n"
        "4. NO ASTERISKS: Never use markdown bold (**) or asterisks anywhere in the output.\n\n"
        "==================================================\n"
        "STRICT GROUNDING RULE\n"
        "==================================================\n"
        "Rely ONLY on the provided context. If the answer is not in the context, reply exactly: 'I could not find relevant information in the comic.'"
    )

    return system_prompt, user_prompt


async def stream_generate_answer_async(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
):
    if not question or not question.strip():
        yield "Please provide a valid question."
        return
    if not context or not context.strip():
        yield "I could not find relevant information in the comic."
        return

    system_prompt, user_prompt = _build_streaming_prompts(
        question=question,
        context=context,
        conversation_history=conversation_history,
        current_page=current_page
    )
    token_budget = get_dynamic_max_tokens(question, current_page)

    has_yielded = False
    try:
        stream_resp = _safe_chat_stream_async(
            model=EDEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=token_budget
        )
        
        async for chunk in stream_resp:
            if chunk:
                has_yielded = True
                yield chunk
    except Exception as e:
        print(f"Streaming Error (Async): {e}")
        # [NEW FIX] Fallback response if the LLM crashes mid-stream or token limit exceeds
        if not has_yielded:
            yield "Sorry, I couldn't generate an answer due to too much context. Try asking a more specific question."
        return
        
    if not has_yielded:
        yield "Sorry, I encountered an issue while generating a response. Please try again."


def stream_generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
):
    if not question or not question.strip():
        yield "Please provide a valid question."
        return
    if not context or not context.strip():
        yield "I could not find relevant information in the comic."
        return

    system_prompt, user_prompt = _build_streaming_prompts(
        question=question,
        context=context,
        conversation_history=conversation_history,
        current_page=current_page
    )
    token_budget = get_dynamic_max_tokens(question, current_page)

    has_yielded = False
    try:
        stream_resp = _safe_chat_stream(
            model=EDEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=token_budget
        )

        for chunk in stream_resp:
            if chunk:
                has_yielded = True
                yield chunk
    except Exception as e:
        print(f"Streaming Error (Sync): {e}")
        if not has_yielded:
            yield "Sorry, I couldn't generate an answer due to too much context. Try asking a more specific question."
        return

    if not has_yielded:
        yield "Sorry, I encountered an issue while generating a response. Please try again."