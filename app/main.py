"""FastAPI application entry point.

Configures the app with CORS, lifespan handling,
and CRS engine initialization.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.data.loader import extract_few_shot_conversations
from app.routers.chat import register_crs, router as chat_router
from app.services.agent_crs import AgentCRS
from app.services.rag_crs import RagCRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize CRS engines on startup.

    Loads few-shot examples from Conversation.txt (if available) and
    registers both CRS engines for use by the API.
    """
    logger.info("=" * 60)
    logger.info("Starting Conversational Recommender System (CRS)")
    logger.info("=" * 60)

    settings = get_settings()

    # Load few-shot examples from Conversation.txt (gracefully handles missing data)
    few_shot_examples: list[dict[str, str]] = []
    try:
        few_shot_examples = extract_few_shot_conversations(max_examples=50)
        if few_shot_examples:
            logger.info("Loaded %d few-shot examples from dataset", len(few_shot_examples))
        else:
            logger.warning(
                "No conversations found at %s — CRS will work without few-shot examples. "
                "Run `python -m scripts.ingest` to populate the vector store.",
                settings.data_path,
            )
    except Exception as e:
        logger.warning("Failed to load dataset: %s", str(e))

    # Register CRS engines
    rag_crs = RagCRS(few_shot_examples=few_shot_examples)
    register_crs("rag", rag_crs)
    logger.info("✓ RAG CRS engine registered")

    agent_crs = AgentCRS()
    register_crs("agent", agent_crs)
    logger.info("✓ Multi-Agent CRS engine registered")

    logger.info("=" * 60)
    logger.info("CRS ready — serving at http://localhost:8000")
    logger.info("=" * 60)

    yield

    # Cleanup
    logger.info("Shutting down CRS...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Movie Recommender CRS",
        description=(
            "A Conversational Recommender System (CRS) for movies using "
            "LLM-based approaches: RAG and Multi-Agent (LangGraph). "
            "Built on the LLM-Redial dataset."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_router, tags=["CRS"])

    # Serve demo UI
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/static/demo.html")

    return app


# Application instance
app = create_app()
