from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    comic_id: str = Field(description="Comic identifier")
    page_number: int = Field(description="Page number where evidence is found")
    chunk_id: str = Field(description="Unique chunk identifier")
    chunk_index: int = Field(description="Sequential chunk index on page")
    distance: float = Field(description="Cosine distance score")


class QuestionRequest(BaseModel):
    question: str = Field(default="", description="User question about the comic")
    comic_id: str | None = Field(default=None, description="Comic identifier (required for generic /ask)")


class QuestionResponse(BaseModel):
    comic_id: str = Field(description="Comic identifier")
    question: str = Field(description="Original user question")
    answer: str = Field(description="Grounded answer based on retrieved comic context")
    sources: list[SourceItem] = Field(default_factory=list, description="List of evidence chunks used")


class ComicUploadResponse(BaseModel):
    message: str = Field(description="Status message")
    comic_id: str = Field(description="Generated unique comic identifier")
    filename: str = Field(description="Original uploaded filename")
    format: str = Field(description="File format extension")
    total_pages: int = Field(description="Total pages extracted")
    status: str = Field(default="processing", description="Processing status: 'processing', 'completed', 'failed'")
    analyzed_pages: int = Field(default=0, description="Pages analyzed so far")
    successful_pages: int = Field(default=0, description="Successfully analyzed pages")
    failed_pages: int = Field(default=0, description="Failed pages during analysis")
    json_path: str = Field(description="Relative path to stored comic JSON")
    rag_ingested: bool = Field(default=False, description="Whether RAG vector ingestion succeeded")
    rag_chunks_stored: int = Field(default=0, description="Number of chunks stored in vector database")
    rag_error: str | None = Field(default=None, description="Error message if RAG ingestion failed")


class ComicStatusResponse(BaseModel):
    comic_id: str = Field(description="Comic identifier")
    title: str = Field(default="", description="Comic title")
    status: str = Field(description="Ingestion status: 'processing', 'completed', 'failed'")
    total_pages: int = Field(default=0, description="Total pages extracted")
    analyzed_pages: int = Field(default=0, description="Number of pages analyzed so far")
    successful_pages: int = Field(default=0, description="Number of successfully analyzed pages")
    failed_pages: int = Field(default=0, description="Number of failed pages")
    rag_ingested: bool = Field(default=False, description="Whether RAG vector ingestion is complete")


class ComicListItem(BaseModel):
    comic_id: str = Field(description="Comic identifier")
    title: str = Field(description="Comic title or series name")
    total_pages: int = Field(default=0, description="Total page count")
    status: str = Field(default="completed", description="Status: 'processing', 'completed', 'failed'")
    analyzed_pages: int = Field(default=0, description="Number of analyzed pages")
    source_format: str = Field(default="cbr", description="Source comic file format")
    uploaded_at: str = Field(default="", description="ISO timestamp of comic upload")
    last_opened_at: str | None = Field(default=None, description="ISO timestamp when last opened")
    cover_thumbnail_url: str | None = Field(default=None, description="Signed URL for cover thumbnail")


class ComicDeleteResponse(BaseModel):
    message: str = Field(description="Deletion status message")
    comic_id: str = Field(description="Deleted comic identifier")

