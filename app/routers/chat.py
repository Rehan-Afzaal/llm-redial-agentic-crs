"""API router for chat, models, and health endpoints.

Provides:
- POST /chat  → SSE streaming movie recommendations
- GET  /models → Available CRS model info
- GET  /health → System health check
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.vector_store import get_movie_count, is_ready
from app.models import (
    ChatRequest,
    CRSModel,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from app.services.base_crs import BaseCRS

logger = logging.getLogger(__name__)

router = APIRouter()

# CRS instances are registered at app startup (see main.py)
_crs_engines: dict[str, BaseCRS] = {}


def register_crs(model_id: str, engine: BaseCRS) -> None:
    """Register a CRS engine for use by the chat endpoint.

    Args:
        model_id: Identifier matching CRSModel enum values.
        engine: The CRS engine instance.
    """
    _crs_engines[model_id] = engine
    logger.info("Registered CRS engine: %s (%s)", model_id, engine.model_name)


def get_crs(model: CRSModel) -> BaseCRS:
    """Get the CRS engine for a given model type.

    Args:
        model: The requested CRS model.

    Returns:
        The corresponding CRS engine instance.

    Raises:
        HTTPException: If the model is not available.
    """
    engine = _crs_engines.get(model.value)
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model.value}' is not available. "
            f"Available: {list(_crs_engines.keys())}",
        )
    return engine


@router.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream movie recommendations via Server-Sent Events (SSE).

    Accepts a user message with optional conversation history and
    streams the response token-by-token using the selected CRS model.

    Args:
        request: ChatRequest with message, history, and model selection.

    Returns:
        SSE stream with 'token' events and a final 'done' event.
    """
    logger.info(
        "Chat request: model=%s, message=%s, history_len=%d",
        request.model.value,
        request.message[:80],
        len(request.history),
    )

    crs = get_crs(request.model)

    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events from the CRS response stream."""
        try:
            async for token in crs.recommend(
                message=request.message,
                history=request.history,
            ):
                yield {
                    "event": "token",
                    "data": token,
                }
        except Exception as e:
            logger.error("Streaming error: %s", str(e), exc_info=True)
            yield {
                "event": "error",
                "data": f"Error generating recommendation: {str(e)}",
            }
        finally:
            yield {
                "event": "done",
                "data": "[DONE]",
            }

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List all available CRS models.

    Returns:
        ModelsResponse with details about each available model.
    """
    models = [
        ModelInfo(
            id=engine.model_id,
            name=engine.model_name,
            description=engine.model_description,
        )
        for engine in _crs_engines.values()
    ]
    return ModelsResponse(models=models)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check.

    Returns:
        HealthResponse with system status, model availability,
        and vector store readiness.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_available=list(_crs_engines.keys()),
        vector_store_ready=is_ready(),
        movie_count=get_movie_count(),
    )
