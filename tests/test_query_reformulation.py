"""
Unit Tests for Smart Query Reformulation (Coreference Resolution)
"""
import uuid
import pytest
from unittest import mock

from app.services.conversation import get_sliding_window_history
from app.services.llm import (
    QUERY_OPTIMIZER_SYSTEM_PROMPT,
    reformulate_query,
    reformulate_query_async,
)
from app.services.rag_qa import answer_question


def test_query_optimizer_prompt_verbatim():
    """Ensure the system prompt matches the exact prompt required by user specifications."""
    expected_prompt = (
        "You are a Query Reformulator, NOT an answering assistant. DO NOT answer the user's question. "
        "Your ONLY job is to resolve pronouns using the chat history and rewrite the question. "
        "If no pronouns exist or no rewrite is needed, you MUST output the exact original question. "
        "NEVER apologize, NEVER say 'I cannot answer', just output the query."
    )
    assert QUERY_OPTIMIZER_SYSTEM_PROMPT == expected_prompt


def test_reformulate_query_bad_phrases_fallback():
    """Verify that if the LLM produces bad phrases like 'I could not find', it falls back to original query."""
    fake_history = "User: Who is Doctor Doom?\nAssistant: He is Victor."
    user_q = "What did he build?"

    bad_phrases = [
        "I could not find relevant information in the comic.",
        "I cannot answer that question based on context.",
        "I don't know what he built.",
        "I am sorry, but I can only rewrite queries.",
        "The context does not contain information on this."
    ]

    for bad in bad_phrases:
        with mock.patch("app.services.llm._safe_chat_complete", return_value=bad):
            res = reformulate_query(query=user_q, chat_history=fake_history)
            assert res == user_q


def test_reformulate_query_safe_parsing_none():
    """Verify that if LLM returns None, empty string, or dict, it safely falls back to original query without crashing."""
    fake_history = "User: Who is Doctor Doom?\nAssistant: He is Victor."
    user_q = "Where does he live?"

    for bad_ret in [None, "", {}, {"text": None}]:
        with mock.patch("app.services.llm._safe_chat_complete", return_value=bad_ret):
            res = reformulate_query(query=user_q, chat_history=fake_history)
            assert res == user_q


def test_reformulate_query_empty_history():
    """When chat history is empty, reformulate_query should immediately return the original query untouched."""
    q = "What did he build?"
    result = reformulate_query(query=q, chat_history="")
    assert result == q

    result_whitespace = reformulate_query(query=q, chat_history="   \n  ")
    assert result_whitespace == q


def test_reformulate_query_calls_llm():
    """Verify that reformulate_query correctly invokes the LLM with system and user prompts."""
    fake_history = "User: Who is Victor Von Doom?\nAssistant: Victor Von Doom is a monarch of Latveria."
    user_q = "What did he build?"

    with mock.patch("app.services.llm._safe_chat_complete") as mock_complete:
        mock_complete.return_value = "What did Victor Von Doom build?"
        result = reformulate_query(query=user_q, chat_history=fake_history)

        assert result == "What did Victor Von Doom build?"
        mock_complete.assert_called_once()
        args, kwargs = mock_complete.call_args
        messages = kwargs.get("messages", [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == QUERY_OPTIMIZER_SYSTEM_PROMPT
        assert "Chat History:" in messages[1]["content"]
        assert "Victor Von Doom" in messages[1]["content"]
        assert "User's Latest Question:" in messages[1]["content"]
        assert user_q in messages[1]["content"]


@pytest.mark.anyio
async def test_reformulate_query_async():
    """Verify asynchronous query reformulation works properly in async event loops."""
    fake_history = "User: Who is Doctor Doom?\nAssistant: He is Victor Von Doom."
    user_q = "Where does he live?"

    with mock.patch("app.services.llm._safe_chat_complete") as mock_complete:
        mock_complete.return_value = "Where does Victor Von Doom live?"
        result = await reformulate_query_async(query=user_q, chat_history=fake_history)
        assert result == "Where does Victor Von Doom live?"


def test_get_sliding_window_history_formatting():
    """Verify sliding window queries messages and formats them into User: ... \n Assistant: ..."""
    mock_db = mock.MagicMock()

    class FakeMsg:
        def __init__(self, role, content):
            self.role = role
            self.content = content
            self.created_at = None
            self.sources_json = None

    # Return 6 messages in DB, order_by desc
    fake_db_msgs = [
        FakeMsg("assistant", "He built the Doombot."),
        FakeMsg("user", "What did he build?"),
        FakeMsg("assistant", "Victor Von Doom is a scientist."),
        FakeMsg("user", "Who is Doom?"),
        FakeMsg("assistant", "Hello! How can I help?"),
        FakeMsg("user", "Hi")
    ]

    mock_db.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = fake_db_msgs[:5]

    test_conv_id = str(uuid.uuid4())
    test_comic_id = "test-comic-123"

    raw_msgs, history_str = get_sliding_window_history(
        conversation_id=test_conv_id,
        comic_id=test_comic_id,
        limit=5,
        db=mock_db
    )

    # Reversed order should be chronological
    assert len(raw_msgs) == 5
    assert "User: Who is Doom?" in history_str
    assert "Assistant: Victor Von Doom is a scientist." in history_str
    assert "User: What did he build?" in history_str
    assert "Assistant: He built the Doombot." in history_str
    assert history_str.startswith("User: Who is Doom?") or history_str.startswith("Assistant: Hello! How can I help?")


def test_answer_question_with_standalone_query():
    """Verify answer_question utilizes standalone_query for vector retrieval and generation."""
    with mock.patch("app.services.rag_qa.retrieve_chunks") as mock_retrieve, \
         mock.patch("app.services.rag_qa.generate_answer") as mock_gen:

        mock_retrieve.return_value = [{
            "chunk_id": "c1",
            "content": "Victor Von Doom created the time platform.",
            "metadata": {"comic_id": "comic-1", "page_number": 1, "chunk_index": 1},
            "distance": 0.2
        }]
        mock_gen.return_value = "Victor Von Doom created the time platform."

        res = answer_question(
            question="What did he create?",
            comic_id="comic-1",
            standalone_query="What did Victor Von Doom create?"
        )

        assert res["question"] == "What did he create?"
        assert res["standalone_query"] == "What did Victor Von Doom create?"
        assert res["answer"] == "Victor Von Doom created the time platform."

        # Verify retrieve_chunks was called with normalized standalone query
        mock_retrieve.assert_called_once()
        retrieval_query_arg = mock_retrieve.call_args[1]["query"]
        assert "victor" in retrieval_query_arg.lower()
        assert "doom" in retrieval_query_arg.lower()

        # Verify generate_answer received standalone_query
        mock_gen.assert_called_once()
        assert mock_gen.call_args[1]["question"] == "What did Victor Von Doom create?"
