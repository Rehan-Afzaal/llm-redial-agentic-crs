"""Tests for the Multi-Agent CRS engine."""

from __future__ import annotations

import pytest


class TestAgentCRSProperties:
    """Tests for Agent CRS model identity."""

    def test_model_id(self) -> None:
        """Model ID should be 'agent'."""
        import os
        os.environ.setdefault("OPENAI_API_KEY", "test-key")

        from app.services.agent_crs import AgentCRS

        crs = AgentCRS()
        assert crs.model_id == "agent"
        assert crs.model_name == "Multi-Agent Recommender"
        assert "LangGraph" in crs.model_description

    def test_agent_state_structure(self) -> None:
        """AgentState should have the expected keys."""
        from app.services.agent_crs import AgentState

        # TypedDict should accept these keys
        state: AgentState = {
            "user_message": "test",
            "history": [],
            "preferences": {},
            "retrieved_movies": [],
            "recommendations": [],
            "final_response": "",
            "error": None,
        }
        assert state["user_message"] == "test"
        assert state["error"] is None
