"""Abstract base class for CRS implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.models import Message


class BaseCRS(ABC):
    """Abstract base class for Conversational Recommender Systems.

    All CRS implementations must inherit from this class and implement
    the `recommend` method for streaming responses.
    """

    @abstractmethod
    async def recommend(
        self,
        message: str,
        history: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming movie recommendations.

        Args:
            message: The user's current message.
            history: Previous conversation messages.

        Yields:
            Response text chunks for SSE streaming.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this CRS model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name for this CRS model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model_description(self) -> str:
        """Description of this CRS model's approach."""
        ...  # pragma: no cover
