"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CRSModel(str, Enum):
    """Available CRS model types."""

    RAG = "rag"
    AGENT = "agent"


class MessageRole(str, Enum):
    """Conversation message roles."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single conversation message."""

    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's current message / question.",
        examples=["I loved Inception and Interstellar. What should I watch next?"],
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Previous conversation messages for context.",
    )
    model: CRSModel = Field(
        default=CRSModel.RAG,
        description="Which CRS approach to use: 'rag' or 'agent'.",
    )


class ModelInfo(BaseModel):
    """Information about an available CRS model."""

    id: str
    name: str
    description: str


class ModelsResponse(BaseModel):
    """Response for the /models endpoint."""

    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = "healthy"
    version: str = "1.0.0"
    models_available: list[str]
    vector_store_ready: bool
    movie_count: int = 0


class MovieMetadata(BaseModel):
    """Metadata for a movie in the knowledge base."""

    title: str
    genres: list[str] = Field(default_factory=list)
    year: Optional[str] = None
    description: str = ""

    @property
    def display_text(self) -> str:
        """Formatted display string for the movie."""
        genre_str = ", ".join(self.genres) if self.genres else "Unknown Genre"
        year_str = f" ({self.year})" if self.year else ""
        return f"{self.title}{year_str} [{genre_str}]"
