"""
Comic & ComicPage SQLAlchemy Models
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


def get_utc_now():
    return datetime.now(timezone.utc)


class Comic(Base):
    __tablename__ = "comics"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    source_format = Column(String(20), nullable=False, default="cbr")
    status = Column(String(50), nullable=False, default="processing", index=True)
    total_pages = Column(Integer, nullable=False, default=0)
    analyzed_pages = Column(Integer, nullable=False, default=0)
    successful_pages = Column(Integer, nullable=False, default=0)
    failed_pages = Column(Integer, nullable=False, default=0)
    original_storage_path = Column(String(512), nullable=True)
    json_storage_path = Column(String(512), nullable=True)
    full_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=get_utc_now, onupdate=get_utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="comics")
    pages = relationship("ComicPage", back_populates="comic", cascade="all, delete-orphan", order_by="ComicPage.page_number")
    conversations = relationship("Conversation", back_populates="comic", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Comic id={self.id} title={self.title} status={self.status} user_id={self.user_id}>"


class ComicPage(Base):
    __tablename__ = "comic_pages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    comic_id = Column(String(36), ForeignKey("comics.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="processing")
    image_storage_path = Column(String(512), nullable=True)
    thumbnail_storage_path = Column(String(512), nullable=True)
    analysis_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    # Relationships
    comic = relationship("Comic", back_populates="pages")

    def __repr__(self):
        return f"<ComicPage comic_id={self.comic_id} page={self.page_number} status={self.status}>"
