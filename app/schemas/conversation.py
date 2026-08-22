"""
Conversation Schemas

Pydantic models for conversation creation, history details, and chat questions.
"""
from pydantic import BaseModel, Field

from app.schemas.comic import SourceItem


class ConversationCreateRequest(BaseModel):
    comic_id: str = Field(description="Comic identifier to tie this conversation to")


class ConversationResponse(BaseModel):
    conversation_id: str = Field(description="Unique conversation identifier (UUID)")
    comic_id: str = Field(description="Comic identifier associated with the conversation")
    created_at: str = Field(description="ISO-8601 creation timestamp")
    updated_at: str = Field(description="ISO-8601 last update timestamp")


class ConversationMessage(BaseModel):
    role: str = Field(description="Role of the speaker: 'user' or 'assistant'")
    content: str = Field(description="Text message content")
    timestamp: str = Field(description="ISO-8601 message timestamp")


class ConversationDetailResponse(BaseModel):
    conversation_id: str = Field(description="Unique conversation identifier (UUID)")
    comic_id: str = Field(description="Comic identifier associated with the conversation")
    created_at: str = Field(description="ISO-8601 creation timestamp")
    updated_at: str = Field(description="ISO-8601 last update timestamp")
    messages: list[ConversationMessage] = Field(default_factory=list, description="Chronological conversation message history")


class ConversationQuestionRequest(BaseModel):
    question: str = Field(default="", description="User question or follow-up question")
    current_page: int | None = Field(default=None, description="Current page number viewed by user")


class ConversationQuestionResponse(BaseModel):
    conversation_id: str = Field(description="Conversation identifier")
    comic_id: str = Field(description="Comic identifier")
    question: str = Field(description="Original user question")
    answer: str = Field(description="Contextually grounded answer from comic evidence")
    sources: list[SourceItem] = Field(default_factory=list, description="Retrieved comic chunk sources")
