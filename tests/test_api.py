"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.main import app
from app.routers.chat import register_crs
from app.services.rag_crs import RagCRS
from app.services.agent_crs import AgentCRS


@pytest.fixture(autouse=True)
def _register_engines() -> None:
    """Ensure CRS engines are registered before tests run.

    The TestClient may not trigger lifespan in all pytest versions,
    so we manually register the engines.
    """
    register_crs("rag", RagCRS(few_shot_examples=[]))
    register_crs("agent", AgentCRS())


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        """Health response should contain expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_available" in data
        assert "vector_store_ready" in data
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    def test_models_returns_200(self, client: TestClient) -> None:
        """Models endpoint should return 200."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_contains_rag_and_agent(self, client: TestClient) -> None:
        """Both RAG and Agent models should be listed."""
        response = client.get("/models")
        data = response.json()
        model_ids = [m["id"] for m in data["models"]]
        assert "rag" in model_ids
        assert "agent" in model_ids


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_requires_message(self, client: TestClient) -> None:
        """Chat endpoint should reject empty requests."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_accepts_valid_request(self, client: TestClient) -> None:
        """Chat endpoint should accept valid request format and return SSE stream."""
        response = client.post(
            "/chat",
            json={
                "message": "I like action movies",
                "history": [],
                "model": "rag",
            },
        )
        # SSE stream starts (may contain error events due to test API key)
        assert response.status_code == 200
