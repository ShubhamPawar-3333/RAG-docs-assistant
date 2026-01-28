"""
FastAPI Application - Main Entry Point

DocuMind AI - RAG-Powered Documentation Assistant API
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import query, ingest, health
from src.api.middleware.errors import (
    ErrorHandlerMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
)
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting DocuMind AI API...")
    logger.info(f"Environment: {settings.environment}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DocuMind AI API...")


# Create FastAPI application
app = FastAPI(
    title="DocuMind AI",
    description="RAG-Powered Documentation Assistant API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - first added = last executed)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DocuMind AI",
        "version": "1.0.0",
        "description": "RAG-Powered Documentation Assistant",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
