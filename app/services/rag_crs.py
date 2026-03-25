"""RAG-based Conversational Recommender System.

Implements Approach #1: Retrieval-Augmented Generation.
Retrieves relevant movies from ChromaDB via semantic similarity,
then uses an LLM with few-shot examples and CoT to generate
streaming recommendations.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    format_few_shot_examples,
    format_history,
    format_retrieved_movies,
)
from app.core.vector_store import search_movies
from app.models import Message
from app.services.base_crs import BaseCRS

logger = logging.getLogger(__name__)


class RagCRS(BaseCRS):
    """RAG-based Conversational Recommender System.

    Flow:
    1. Embed user's message → search ChromaDB for top-K similar movies
    2. Optionally inject dynamic few-shot examples from the dataset
    3. Build prompt with persona, CoT instructions, and constraints
    4. Stream LLM response token-by-token
    """

    def __init__(self, few_shot_examples: list[dict[str, str]] | None = None) -> None:
        """Initialize the RAG CRS.

        Args:
            few_shot_examples: Pre-loaded exemplar conversations for
                dynamic few-shot prompting.
        """
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._few_shot_examples = few_shot_examples or []

    @property
    def model_id(self) -> str:
        return "rag"

    @property
    def model_name(self) -> str:
        return "RAG Recommender"

    @property
    def model_description(self) -> str:
        return (
            "Retrieval-Augmented Generation approach. Searches a movie "
            "knowledge base for semantically similar movies, then uses "
            "few-shot Chain-of-Thought prompting to generate personalized "
            "recommendations."
        )

    async def recommend(
        self,
        message: str,
        history: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming movie recommendations using RAG.

        Args:
            message: The user's current message.
            history: Previous conversation messages.

        Yields:
            Response text chunks for SSE streaming.
        """
        # Step 1: Retrieve relevant movies from vector store
        logger.info("RAG: Searching for movies matching: %s", message[:100])
        retrieved_movies = search_movies(
            query=message,
            n_results=self._settings.rag_top_k,
        )
        logger.info("RAG: Retrieved %d candidate movies", len(retrieved_movies))

        # Step 2: Format prompt components
        history_dicts = [{"role": m.role.value, "content": m.content} for m in history]
        few_shot_str = format_few_shot_examples(
            self._few_shot_examples,
            k=self._settings.rag_few_shot_k,
        )
        movies_str = format_retrieved_movies(retrieved_movies)
        history_str = format_history(history_dicts)

        # Step 3: Build the user message with RAG context
        user_prompt = RAG_USER_TEMPLATE.format(
            few_shot_examples=few_shot_str,
            retrieved_movies=movies_str,
            history=history_str,
            user_message=message,
        )

        # Step 4: Stream the response from OpenAI
        logger.info("RAG: Streaming response from %s", self._settings.openai_model)
        stream = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
