"""
LLM Service

Generates grounded comic answers using Mistral AI, enforcing strict grounding rules,
pinned fallback phrase, and optional conversation memory context.
"""
import asyncio
import random
import re
import time
from mistralai.client import Mistral

from deep_translator import GoogleTranslator

from app.core.config import LLM_MODEL, MISTRAL_API_KEY, USE_LIBRARY_TRANSLATION

# -----------------------------
# Mistral Client
# -----------------------------

client = Mistral(
    api_key=MISTRAL_API_KEY
)


def _safe_chat_complete(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    max_retries: int = 5,
    **kwargs
):
    """
    Executes client.chat.complete with exponential backoff on 429 rate limit errors.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
                wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
            else:
                raise


def _safe_chat_stream(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1800,
    max_retries: int = 5,
    **kwargs
):
    """
    Executes client.chat.stream with exponential backoff on 429 rate limit errors.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.chat.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
                wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
            else:
                raise


async def _safe_chat_stream_async(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1800,
    max_retries: int = 5,
    **kwargs
):
    """
    Executes client.chat.stream_async with exponential backoff on 429 rate limit errors.
    """
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate limit" in err_str or "capacity" in err_str or "sdkerror" in err_str) and attempt < max_retries:
                wait_time = (2 ** attempt) * 2.5 + random.uniform(0.5, 1.5)
                await asyncio.sleep(wait_time)
            else:
                raise


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
    # Roman Urdu / Hindi question words & markers
    "kya", "kiya", "kaun", "kon", "kahan", "kyun", "kyu", "kab", "kese", "kaise", "kitna", "kitni", "kitne",
    "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "honge", "hona", "hua", "hui", "hue",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "ne", "tak", "mai", "me",
    "uska", "uski", "uske", "usko", "iska", "iski", "iske", "isko", "unka", "unki", "unke", "unko", "kisko",
    "mera", "meri", "mere", "apna", "apni", "apne",
    "mujhe", "tum", "aap", "humein", "hum",
    "yeh", "woh", "ye", "wo", "kuch", "sab", "batao", "bataiye", "bata", "btao", "btaao", "bolo", "karo", "karna",
    "baare", "baray", "kaisa", "kaisi", "kaise", "lag", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "dikhao", "samjhao", "chahiye", "khel", "karta", "karti", "karte",
    # Spanish / French / German / other common European markers
    "que", "qui", "quien", "quienes", "donde", "cuando", "por", "para", "como", "esta", "esto", "del", "las", "los",
    "dans", "avec", "pour", "une", "und", "der", "das", "nicht"
}


def is_english_query(query: str) -> bool:
    """
    Lightweight heuristic check to detect if a query is English.
    Returns True if the query appears to be English, False otherwise.
    """
    if not query or not query.strip():
        return True

    # 1. Non-Latin script check (Urdu, Arabic, Hindi/Devanagari, CJK, Cyrillic, etc.)
    non_latin = re.search(r"[^\x00-\x7F\u00C0-\u024F]", query)
    if non_latin:
        return False

    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]
    if not words:
        return True

    # 2. Check for explicit non-English indicator tokens
    for w in words:
        if w in NON_ENGLISH_MARKERS:
            return False

    # 3. Count English vocabulary words
    english_word_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    if english_word_count > 0:
        return True

    return True


def is_urdu_script_query(query: str) -> bool:
    """Checks if a query contains Arabic / Urdu script characters."""
    if not query:
        return False
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", query))


def extract_and_protect_proper_names(text: str) -> tuple[str, dict[str, str]]:
    """
    Extracts proper nouns / character names from English text and replaces them with
    protected placeholder tokens (__PROPER_NAME_X__) so translation tools do not alter them.
    """
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
    """Restores protected placeholder tokens back to their original exact proper names."""
    if not text or not mapping:
        return text
    restored = text
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    return restored


def transliterate_to_roman_urdu(urdu_text: str) -> str:
    """
    Lightweight, fast transliteration of Urdu script into Roman Urdu using Latin alphabet (A-Z).
    """
    if not urdu_text or not urdu_text.strip():
        return urdu_text

    prompt = (
        "You are a precise Roman Urdu transliterator. Transliterate the provided Urdu script text "
        "into natural, readable Roman Urdu using Latin alphabet letters (A-Z) only (e.g. 'Kahani ek aise shakhs ke baare mein...').\n"
        "RULES:\n"
        "1. Write ONLY in Latin letters (A-Z). NEVER use Arabic/Urdu script.\n"
        "2. Preserve all character names, proper nouns (e.g. Cynthia, Werner Von Doom, Victor, Baron), and placeholder tokens (__PROPER_NAME_*) EXACTLY intact in original Latin spelling. NEVER transliterate names phonetically.\n"
        "3. Transliterate the entire text completely from beginning to end without stopping early or adding commentary.\n"
        "4. Output ONLY the transliterated text."
    )

    response = _safe_chat_complete(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": urdu_text
            }
        ],
        temperature=0.1,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()


def translate_with_library(english_text: str, original_question: str) -> str:
    """
    Fast translation using deep-translator (GoogleTranslator) with Roman Urdu transliteration,
    strict character name protection, and automatic fallback to LLM translation if needed.
    """
    if not english_text or not english_text.strip():
        return english_text

    try:
        protected_text, name_map = extract_and_protect_proper_names(english_text)

        # Case 1: Urdu Script query -> Direct English to Urdu script translation
        if is_urdu_script_query(original_question):
            translated_ur = GoogleTranslator(source="en", target="ur").translate(protected_text)
            restored_ur = restore_proper_names(translated_ur, name_map)
            return clean_llm_response(restored_ur, question=original_question)

        # Case 2: Roman Urdu query -> English to Urdu script via library, then lightweight transliteration to Latin letters
        urdu_script = GoogleTranslator(source="en", target="ur").translate(protected_text)
        roman_urdu = transliterate_to_roman_urdu(urdu_script)
        restored_roman = restore_proper_names(roman_urdu, name_map)
        return clean_llm_response(restored_roman, question=original_question)

    except Exception as e:
        print(f"Library translation encountered issue ({e}). Falling back to LLM translation...")
        return translate_answer(english_text, original_question)


def translate_answer(english_answer: str, original_question: str) -> str:
    """
    Step 2 translation: Translates an English grounded answer into the language,
    script, and style of the user's original question using Mistral.
    """
    translation_system_prompt = (
        "You are a precise, natural translator. Translate the text faithfully "
        "matching the exact language, script/alphabet, and style of the user's question."
    )

    translation_user_prompt = f"""You are a precise translator. Translate the following English text into the exact language, script, and style of the user's original question.

CRITICAL SCRIPT & LANGUAGE RULES:
1. SCRIPT MATCHING:
   - If the USER'S ORIGINAL QUESTION is written in Latin/English letters (A-Z, e.g. Roman Urdu, Hinglish, Spanish, French, German, Indonesian, Romaji, etc.), your translation MUST be written entirely in Latin/English letters (A-Z). For Roman Urdu questions (e.g. "story kiya hai"), write in natural Roman Urdu using Latin alphabet — NEVER use Arabic/Urdu script (اردو).
   - If the USER'S ORIGINAL QUESTION is written in a non-Latin script (e.g. Arabic/Urdu script like اردو, Devanagari, Japanese Kana/Kanji, Cyrillic, etc.), write in that exact script.
2. ACCURACY: Preserve all facts, character names, events, and details EXACTLY as stated in the English text — do not invent, add, or omit any details.
3. PRESERVE PROPER NOUNS & CHARACTER NAMES: Keep all character names (e.g. 'Cynthia', 'Werner Von Doom', 'Victor Von Doom', 'Baron', 'Latveria') EXACTLY in their original English spelling. NEVER translate, alter, or phonetically transliterate character names into Urdu sound-approximations like 'Santhya', 'Santhiya', 'Varner', etc.
4. NATURAL PHRASING: Use natural, conversational grammar as written by native speakers.
5. FORMATTING: Keep proper nouns, character names, and markdown formatting intact.
6. NO COMMENTARY: Output ONLY the translated text. Do not add intros, notes, or explanations.

USER'S ORIGINAL QUESTION (for language & script reference):
{original_question}

ENGLISH TEXT TO TRANSLATE:
{english_answer}"""

    response = _safe_chat_complete(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": translation_system_prompt
            },
            {
                "role": "user",
                "content": translation_user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1500
    )

    translated = response.choices[0].message.content.strip()
    return clean_llm_response(translated, question=original_question)


def clean_llm_response(text: str, question: str = "") -> str:
    """
    Post-processes the LLM output to enforce formatting and grounding compliance:
    - Strips meta-commentary parentheticals (e.g. '(no outside knowledge used)')
    - Strips unrequested 'Missing Details' sections
    - Strips excessive bolding (max 1-2 bold terms total, unbolds section headers and bullet labels)
    """
    if not text or not text.strip():
        return text

    cleaned = text.strip()

    # 1. Strip meta-commentary parentheticals/brackets
    cleaned = re.sub(
        r"\s*\([^\)]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context|no external knowledge)[^\)]*\)",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\s*\[[^\]]*(?:no outside knowledge|outside knowledge|based only on|just comic context|comic context only|grounded in the context|no external knowledge)[^\]]*\]",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # 2. Strip 'Missing Details' / 'Missing Information' sections unless explicitly requested
    q_lower = question.lower()
    if not any(k in q_lower for k in ["missing", "unclear", "kya nahi", "kya miss", "what is missing", "what's missing"]):
        missing_section_pattern = re.compile(
            r"(?:\n+|^)(?:#{1,4}\s*)?(?:\*\*)?(?:Missing Details|Missing Information|Unclear Details|Missing / Unclear Details|Unclear Information|Gaps in Information|Missing aspects|Things not mentioned)(?:\*\*)?:?.*$",
            re.IGNORECASE | re.DOTALL
        )
        cleaned = missing_section_pattern.sub("", cleaned)

    # 3. Unbold section headers, category titles, and bullet labels (e.g. **Summary:** -> Summary:, **Key Events:** -> Key Events:)
    cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40}:)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([A-Za-z0-9\s\(\)\/_\-\,\.]{1,40})\*\*:", r"\1:", cleaned)

    # 4. Enforce the BOLD FORMATTING RULE (max 1-2 bolded terms in the entire response)
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


BROAD_SUMMARY_REGEX = re.compile(
    r"\b(summary|summarize|overview|full story|entire story|whole story|what happened|what's happening|what is happening|describe this page|explain this page|tell me about this page)\b|"
    r"\b(kya hua|kya ho raha|khulasa|poori kahani|puri kahani|sari kahani|sab batao|is page pe|ye page pe|yeh page pe)\b",
    re.IGNORECASE
)

SHORT_REQUEST_REGEX = re.compile(
    r"\b(short|shortly|brief|briefly|in short|quick|quickly|summarize quickly|one line|two lines|just a line|few words|concise|tldr)\b|"
    r"\b(short\s*(mai|mein|me)|chhota|chota|mukhtasar|kam\s+(shabdon|alfaaz|words)\s+(mein|mai|me)|chhoti|choti|short\s+karke|aik\s+line|do\s+line)\b",
    re.IGNORECASE
)

DETAIL_REQUEST_REGEX = re.compile(
    r"\b(in detail|detailed|elaborate|explain fully|full story|everything|all details|comprehensive)\b|"
    r"\b(poora\s+batao|pura\s+batao|puri\s+detail|poori\s+detail|detail\s+(mein|mai|me)|sab\s+kuch\s+batao|khol\s+kar\s+batao|tafseel)\b",
    re.IGNORECASE
)


def is_explicit_short_query(question: str) -> bool:
    """Detect if the user explicitly requested a short/brief answer."""
    return bool(question and SHORT_REQUEST_REGEX.search(question))


def is_explicit_detail_query(question: str) -> bool:
    """Detect if the user explicitly requested a detailed/elaborated answer."""
    return bool(question and DETAIL_REQUEST_REGEX.search(question))


def get_dynamic_max_tokens(question: str, current_page: int | None = None) -> int:
    """
    Returns an optimized token budget based on query intent:
    - 250 tokens for explicit short answer requests ('short mai batao', 'briefly', etc.)
    - 1800 tokens for explicit detailed breakdown requests ('in detail', 'poora batao')
    - 1500 tokens for broad summary / page overview requests
    - 700 tokens for short factual / character lookups
    """
    if is_explicit_short_query(question):
        return 250
    if is_explicit_detail_query(question):
        return 1800
    if current_page is not None:
        return 1500
    if question and BROAD_SUMMARY_REGEX.search(question):
        return 1500
    return 700


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
) -> str:
    """
    Generates an answer strictly grounded in comic context using Mistral AI.
    - Responds in clear English by default, or Roman Urdu if asked in Roman Urdu.
    - Strictly scopes response to active page when current_page is provided.
    - Enforces BOLD FORMATTING RULE (1-2 bold terms max) and complete punctuation.
    - Dynamically allocates token budget (250 for short, 700 for factual, 1500 for broad summary).
    """
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

    print("=" * 60)
    print("LLM DEBUG (GROUNDED QA)")
    print("LLM MODEL:", LLM_MODEL)
    print("QUESTION:", question)
    print("CURRENT PAGE:", current_page)
    print("DYNAMIC MAX TOKENS:", token_budget)
    print("CONTEXT LENGTH:", len(context))
    print("=" * 60)

    response = _safe_chat_complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=token_budget,
        frequency_penalty=0.3,
        presence_penalty=0.2
    )

    raw_answer = response.choices[0].message.content.strip()
    cleaned = clean_llm_response(raw_answer, question=question)

    print("=" * 60)
    print("LLM DEBUG RESULT:")
    print(cleaned)
    print("=" * 60)

    return cleaned


# -----------------------------
# Stream Generate Answer
# -----------------------------

def _build_streaming_prompts(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
) -> tuple[str, str]:
    """Helper to assemble system and user prompts for grounded streaming QA."""
    is_urdu_script = is_urdu_script_query(question)
    is_english = is_english_query(question) and not is_urdu_script
    is_roman_urdu = not is_english and not is_urdu_script

    if is_urdu_script:
        lang_reminder = "\n\n(LANGUAGE DIRECTIVE: The user asked in Urdu script (اردو). You MUST respond in fluent Urdu script.)"
    elif is_roman_urdu:
        lang_reminder = (
            f"\n\n(MANDATORY LANGUAGE DIRECTIVE: The user asked in Roman Urdu: '{question}'. "
            "You MUST write your entire response in authentic, natural Roman Urdu using the Latin alphabet (A-Z). "
            "Do NOT answer in English. Do NOT output Arabic/Urdu script (اردو).)"
        )
    else:
        lang_reminder = "\n\n(LANGUAGE DIRECTIVE: Respond in clear, fluent English.)"

    if is_explicit_short_query(question):
        formatting_reminder = (
            "\n\n(CRITICAL LENGTH CONSTRAINT: The user explicitly demanded a SHORT answer. "
            "Provide a SHORT, punchy response of EXACTLY 1-3 sentences in a single brief paragraph. "
            "Use markdown bold for AT MOST 1-2 key terms. Finish sentences completely.)"
        )
    elif is_explicit_detail_query(question):
        formatting_reminder = (
            "\n\n(LENGTH CONSTRAINT: The user requested a DETAILED breakdown. "
            "Provide a comprehensive, rich narrative covering all relevant details from the comic context. "
            "Use markdown bold for AT MOST 1-2 key terms. Finish sentences completely.)"
        )
    else:
        formatting_reminder = (
            "\n\n(FORMATTING & STYLE CONSTRAINTS: Write with genuine comic fan excitement and cinematic drama. "
            "Use markdown bold for AT MOST 1-2 key terms in your entire answer. "
            "Prefer a flowing narrative paragraph over bullet lists unless an explicit list was requested. Finish all thoughts and sentences completely.)"
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
                f"(The user is specifically asking about Page {current_page}. Keep your response strictly focused on Page {current_page} evidence.)\n"
            )
        else:
            page_hint = (
                f"\n(Note: The user is currently viewing Page {current_page}, but this question is asking about the comic's overall story, events, or characters. Synthesize the answer from all provided comic context across pages.)\n"
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
COMIC CONTEXT:
{context}

CURRENT QUESTION:
{question}{lang_reminder}{formatting_reminder}
"""
    else:
        user_prompt = f"""{page_hint}COMIC CONTEXT:
{context}

CURRENT QUESTION:
{question}{lang_reminder}{formatting_reminder}
"""

    system_prompt = (
        "You are an expert, passionate comic book reading companion for a multimodal RAG system.\n"
        "Your ONLY factual grounding source is the provided COMIC CONTEXT.\n\n"
        "==================================================\n"
        "LANGUAGE & SCRIPT RULES\n"
        "==================================================\n"
        "1. LANGUAGE MATCHING: Always respond in the EXACT language the user used to ask the question.\n"
        "   - If the user asks in Roman Urdu (e.g. 'story kiya hai isko', 'is page pe kya hua', 'short mai batao', 'kya story hai'), respond entirely in natural, expressive Roman Urdu using Latin alphabet (A-Z).\n"
        "   - If the user asks in English, respond in English.\n"
        "2. NO ARABIC/URDU SCRIPT: NEVER output responses in Arabic/Urdu script (اردو) unless the user's question was explicitly written in Arabic/Urdu script.\n"
        "3. PRESERVE CHARACTER NAMES: Keep character names (e.g., 'Victor Von Doom', 'Cynthia', 'Werner', 'Hawkeye', 'Baron') in their exact standard English spelling. Never alter them.\n\n"
        "==================================================\n"
        "AUTHENTIC COMIC READER TASTE & IMMERSION\n"
        "==================================================\n"
        "1. IMMERSIVE COMIC ENTHUSIAST VOICE:\n"
        "   - Sound like a real comic book enthusiast and connoisseur discussing an issue with a fellow fan.\n"
        "   - NO CHEESY/CANNED OPENERS: NEVER begin answers with artificial narrator clichés like 'Oh man,', 'Right from the get-go!', 'Buckle up!', or 'We're deep in the heart of...'.\n"
        "   - DIVE STRAIGHT INTO THE STORY: Open directly with the narrative drama, the character conflict, the eerie atmosphere, or the turning points shown in the panels.\n"
        "   - In Roman Urdu: Use vivid, authentic storytelling (e.g., 'Yeh comic Victor Von Doom ke dark aur tragic origin ke gird ghoomti hai...'). Make it captivating, impactful, and conversational.\n\n"
        "==================================================\n"
        "PAGE SCOPE & GROUNDING RULES\n"
        "==================================================\n"
        "1. ACTIVE PAGE SCOPE: When answering questions specifically about the active viewing page ('this page', 'here', 'scene on this page', 'who appears on this page'), keep your answer focused on that specific page. Do NOT attribute events from other pages to the active page.\n"
        "2. COMIC STORY & OVERVIEW: When the user asks about the story of the comic, overall plot, or summary ('story kya hai', 'story batao', 'what is the story of this comic', 'summary of comic', 'in short story batao', 'story kiya hai isko'), synthesize the story narrative across the entire comic from start to finish based on the provided COMIC CONTEXT.\n"
        "3. FACTUAL GROUNDING: Rely ONLY on the information explicitly provided in the COMIC CONTEXT. Never invent facts, characters, or actions not in the context. If the context does not contain enough information, reply with exactly: 'I could not find relevant information in the comic.'\n\n"
        "==================================================\n"
        "LENGTH & FORMATTING RULES\n"
        "==================================================\n"
        "1. SHORT REQUESTS: If the user asks for a short answer ('short mai', 'in short', 'briefly'), give strictly 1-3 sentences.\n"
        "2. DETAILED REQUESTS: If asked for detail ('in detail', 'poora batao'), give a comprehensive narrative.\n"
        "3. DEFAULT LENGTH: Provide 1-2 rich, flowing narrative paragraphs.\n"
        "4. BOLD FORMATTING: Use markdown bold for AT MOST 1-2 key terms across the ENTIRE response (e.g. **Victor Von Doom**).\n"
        "5. COMPLETION: Finish all sentences and thoughts completely with proper terminal punctuation."
    )

    return system_prompt, user_prompt


async def stream_generate_answer_async(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
):
    """
    Asynchronous Grounded Question Answering Streaming Generator:
    - If question is English: Streams grounded tokens from Mistral LLM directly to client.
    - If question is Non-English: Streams grounded tokens directly in the target language/script matching question style,
      enforcing strict grounding in comic context, minimal bolding, and no truncation.
    """
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

    stream_resp = await _safe_chat_stream_async(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=token_budget,
        frequency_penalty=0.3,
        presence_penalty=0.2
    )

    async for chunk in stream_resp:
        delta = chunk.data.choices[0].delta.content
        if delta:
            yield delta


def stream_generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
    current_page: int | None = None
):
    """
    Synchronous Grounded Question Answering Streaming Generator:
    - If question is English: Streams grounded tokens from Mistral LLM directly to caller.
    - If question is Non-English: Streams grounded tokens directly in the target language and script matching question style.
    """
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

    stream_resp = _safe_chat_stream(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=token_budget,
        frequency_penalty=0.3,
        presence_penalty=0.2
    )

    for chunk in stream_resp:
        delta = chunk.data.choices[0].delta.content
        if delta:
            yield delta