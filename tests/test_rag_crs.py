"""Tests for the RAG CRS engine."""

from __future__ import annotations

import pytest

from app.core.prompts import (
    format_few_shot_examples,
    format_history,
    format_retrieved_movies,
)


class TestPromptFormatting:
    """Tests for prompt formatting utilities."""

    def test_format_empty_history(self) -> None:
        """Empty history should return placeholder text."""
        result = format_history([])
        assert result == "(Start of conversation)"

    def test_format_history_with_messages(self) -> None:
        """History should format messages with role labels."""
        history = [
            {"role": "user", "content": "I like sci-fi"},
            {"role": "assistant", "content": "Great choice!"},
        ]
        result = format_history(history)
        assert "User" in result
        assert "Assistant" in result
        assert "I like sci-fi" in result
        assert "Great choice!" in result

    def test_format_history_truncates_to_10(self) -> None:
        """History should be truncated to the last 10 messages."""
        history = [{"role": "user", "content": f"Message {i}"} for i in range(20)]
        result = format_history(history)
        assert "Message 10" in result
        assert "Message 19" in result
        # First messages should be excluded
        assert "Message 0" not in result

    def test_format_empty_few_shot(self) -> None:
        """Empty examples should return placeholder."""
        result = format_few_shot_examples([])
        assert "(No examples available)" in result

    def test_format_few_shot_examples(self) -> None:
        """Few-shot examples should be formatted with numbering."""
        examples = [
            {
                "user_message": "I want a thriller",
                "assistant_response": "Try The Silence of the Lambs!",
            },
            {
                "user_message": "Something funny",
                "assistant_response": "The Big Lebowski is hilarious!",
            },
        ]
        result = format_few_shot_examples(examples, k=2)
        assert "Example 1" in result
        assert "Example 2" in result
        assert "thriller" in result
        assert "Silence of the Lambs" in result

    def test_format_few_shot_limits_k(self) -> None:
        """Only k examples should be included."""
        examples = [
            {"user_message": f"Q{i}", "assistant_response": f"A{i}"}
            for i in range(10)
        ]
        result = format_few_shot_examples(examples, k=3)
        assert "Example 3" in result
        # Should not have a 4th example
        assert "Q3" not in result

    def test_format_empty_movies(self) -> None:
        """Empty movie list should return fallback message."""
        result = format_retrieved_movies([])
        assert "No movies retrieved" in result

    def test_format_retrieved_movies(self) -> None:
        """Movies should be formatted with numbering and metadata."""
        movies = [
            {
                "title": "Inception",
                "year": "2010",
                "genres": ["Sci-Fi", "Thriller"],
                "description": "A mind-bending heist movie",
                "similarity_score": 0.95,
            },
            {
                "title": "Interstellar",
                "year": "2014",
                "genres": ["Sci-Fi", "Drama"],
                "description": "Space exploration epic",
                "similarity_score": 0.88,
            },
        ]
        result = format_retrieved_movies(movies)
        assert "Inception" in result
        assert "2010" in result
        assert "Sci-Fi" in result
        assert "Interstellar" in result
        assert "0.95" in result


class TestRagCRSProperties:
    """Tests for RAG CRS model identity."""

    def test_model_id(self) -> None:
        """Model ID should be 'rag'."""
        from app.services.rag_crs import RagCRS

        # Cannot fully init without OpenAI key, but test property
        import os
        os.environ.setdefault("OPENAI_API_KEY", "test-key")

        crs = RagCRS(few_shot_examples=[])
        assert crs.model_id == "rag"
        assert crs.model_name == "RAG Recommender"
        assert "Retrieval-Augmented" in crs.model_description
