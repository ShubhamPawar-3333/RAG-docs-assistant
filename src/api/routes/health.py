"""
Health Check Routes

Provides endpoints for monitoring API health and readiness.
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.models import HealthResponse, DetailedHealthResponse, ComponentHealth
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns the API status, version, and environment.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with component status.
    
    Checks the health of each component:
    - Embeddings model
    - Vector store
    - LLM connection
    """
    components = []
    overall_status = "healthy"
    
    # Check embeddings
    try:
        from src.rag.embeddings import get_embeddings
        embeddings = get_embeddings()
        # Quick test
        _ = embeddings.embed_query("test")
        components.append(ComponentHealth(
            name="embeddings",
            status="healthy",
            details="Model loaded successfully"
        ))
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}")
        components.append(ComponentHealth(
            name="embeddings",
            status="unhealthy",
            details=str(e)
        ))
        overall_status = "degraded"
    
    # Check vector store
    try:
        from src.rag.vectorstore import create_vector_store
        from src.rag.embeddings import get_embeddings
        embeddings = get_embeddings()
        store = create_vector_store(embeddings)
        stats = store.get_collection_stats()
        components.append(ComponentHealth(
            name="vector_store",
            status="healthy",
            details=f"Collection: {stats['collection_name']}, Docs: {stats['document_count']}"
        ))
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        components.append(ComponentHealth(
            name="vector_store",
            status="unhealthy",
            details=str(e)
        ))
        overall_status = "degraded"
    
    # Check LLM
    try:
        from src.rag.llm import LLMManager
        manager = LLMManager()
        info = manager.get_model_info()
        if info.get("api_key_configured"):
            components.append(ComponentHealth(
                name="llm",
                status="healthy",
                details=f"Model: {info['model_name']}"
            ))
        else:
            components.append(ComponentHealth(
                name="llm",
                status="degraded",
                details="API key not configured"
            ))
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        components.append(ComponentHealth(
            name="llm",
            status="unhealthy",
            details=str(e)
        ))
        overall_status = "degraded"
    
    return DetailedHealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    """
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}
