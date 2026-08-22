from typing import Any


def ensure_list(
    value: Any
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        return [value]

    if isinstance(value, list):

        result = []

        for item in value:

            if isinstance(item, dict):

                # Future-proofing:
                # agar OCR structure mein dict aa jaye
                # to useful text fields extract karein.

                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("description")
                    or ""
                )

                text = str(text).strip()

            else:

                text = str(item).strip()

            if text:
                result.append(text)

        return result

    value = str(value).strip()

    return [value] if value else []


def build_complete_text(
    text_data: dict
) -> str:

    """
    OCR ke available text ko intelligently combine karta hai.

    Priority:
    1. dialogue_and_narration
    2. full_text

    Duplicate text avoid karta hai.
    """

    full_text = str(
        text_data.get(
            "full_text",
            ""
        )
        or ""
    ).strip()

    dialogue = ensure_list(
        text_data.get(
            "dialogue_and_narration",
            []
        )
    )

    sound_effects = ensure_list(
        text_data.get(
            "sound_effects",
            []
        )
    )

    signs = ensure_list(
        text_data.get(
            "signs_and_labels",
            []
        )
    )

    # ---------------------------------
    # Main readable text
    # ---------------------------------

    text_parts = []

    # dialogue_and_narration usually
    # contains the most complete OCR text.
    if dialogue:

        for item in dialogue:

            if item not in text_parts:

                text_parts.append(item)

    # If dialogue is missing,
    # fall back to full_text.
    elif full_text:

        text_parts.append(
            full_text
        )

    # ---------------------------------
    # Sound Effects
    # ---------------------------------

    if sound_effects:

        text_parts.append(
            "SOUND EFFECTS:\n"
            + "\n".join(
                f"- {item}"
                for item in sound_effects
            )
        )

    # ---------------------------------
    # Signs / Labels
    # ---------------------------------

    if signs:

        text_parts.append(
            "SIGNS AND LABELS:\n"
            + "\n".join(
                f"- {item}"
                for item in signs
            )
        )

    return "\n\n".join(
        text_parts
    )


def build_page_content(
    page: dict
) -> str:

    analysis = page.get(
        "analysis"
    ) or {}

    page_number = page.get(
        "page_number"
    )

    # ---------------------------------
    # Page Summary
    # ---------------------------------

    page_summary = str(
        analysis.get(
            "page_summary",
            ""
        )
        or ""
    ).strip()

    # ---------------------------------
    # Text
    # ---------------------------------

    text_data = analysis.get(
        "text"
    ) or {}

    if not isinstance(
        text_data,
        dict
    ):
        text_data = {
            "full_text": str(
                text_data
            )
        }

    complete_text = build_complete_text(
        text_data
    )

    # ---------------------------------
    # Visual Description
    # ---------------------------------

    visual = analysis.get(
        "visual_description"
    ) or {}

    if not isinstance(
        visual,
        dict
    ):
        visual = {}

    characters = ensure_list(
        visual.get(
            "characters"
        )
    )

    actions = ensure_list(
        visual.get(
            "actions"
        )
    )

    environment = str(
        visual.get(
            "environment",
            ""
        )
        or ""
    ).strip()

    objects = ensure_list(
        visual.get(
            "objects"
        )
    )

    background = str(
        visual.get(
            "background",
            ""
        )
        or ""
    ).strip()

    other_details = ensure_list(
        visual.get(
            "other_details"
        )
    )

    # ---------------------------------
    # Build Complete Page Document
    # ---------------------------------

    sections = [
        f"PAGE {page_number}"
    ]

    if page_summary:

        sections.append(
            "PAGE SUMMARY:\n"
            + page_summary
        )

    if complete_text:

        sections.append(
            "TEXT:\n"
            + complete_text
        )

    if characters:

        sections.append(
            "CHARACTERS / FIGURES:\n"
            + "\n".join(
                f"- {item}"
                for item in characters
            )
        )

    if actions:

        sections.append(
            "ACTIONS / POSES:\n"
            + "\n".join(
                f"- {item}"
                for item in actions
            )
        )

    if environment:

        sections.append(
            "ENVIRONMENT:\n"
            + environment
        )

    if objects:

        sections.append(
            "OBJECTS:\n"
            + "\n".join(
                f"- {item}"
                for item in objects
            )
        )

    if background:

        sections.append(
            "BACKGROUND:\n"
            + background
        )

    if other_details:

        sections.append(
            "OTHER DETAILS:\n"
            + "\n".join(
                f"- {item}"
                for item in other_details
            )
        )

    return "\n\n".join(
        sections
    )


def comic_to_rag_documents(
    comic_data: dict
) -> list[dict]:

    comic = comic_data.get(
        "comic"
    ) or {}

    comic_id = comic.get(
        "id"
    )

    comic_name = comic.get(
        "name"
    )

    source_format = comic.get(
        "source_format"
    )

    pages = comic_data.get(
        "pages",
        []
    )

    documents = []

    # ---------------------------------
    # Preserve Comic Page Order
    # ---------------------------------

    ordered_pages = sorted(
        pages,
        key=lambda page: page.get(
            "page_number",
            0
        )
    )

    # ---------------------------------
    # Create One Logical Document
    # Per Comic Page
    # ---------------------------------

    for page in ordered_pages:

        if page.get(
            "status"
        ) != "success":

            continue

        content = build_page_content(
            page
        )

        if not content.strip():

            continue

        page_number = page.get(
            "page_number"
        )

        documents.append(
            {
                "content": content,

                "metadata": {
                    "comic_id": comic_id,
                    "comic_name": comic_name,
                    "source_format": source_format,
                    "page_number": page_number,
                    "filename": page.get(
                        "filename"
                    ),
                    "image_path": page.get(
                        "image_path"
                    )
                }
            }
        )

    return documents