"""
Query Routes

Handles document querying with RAG pipeline.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from src.api.models import QueryRequest, QueryResponse, SourceDocument, ErrorResponse
from src.rag.pipeline import create_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache pipeline instances per collection
_pipeline_cache = {}


def _query_http_error(e: Exception) -> HTTPException:
    """Map a pipeline/LLM exception to an appropriate HTTP response.

    Provider SDK errors (rate limits, bad keys) should not all collapse into a
    generic 500 - the client can act differently on a 429 or a 401.
    """
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    msg = str(e)
    low = msg.lower()

    if status == 429 or "rate_limit" in low or "too many requests" in low:
        return HTTPException(
            status_code=429,
            detail={"error": "RateLimited", "message": (
                "The LLM provider rate-limited this request. Retry shortly."
            )},
        )
    if isinstance(e, ValueError) and "api key is required" in low:
        return HTTPException(
            status_code=400,
            detail={"error": "MissingAPIKey", "message": msg},
        )
    if status in (401, 403) or "api key not valid" in low or "invalid api key" in low:
        return HTTPException(
            status_code=401,
            detail={"error": "InvalidAPIKey", "message": (
                "The provided LLM API key was rejected by the provider."
            )},
        )
    if status == 404 or "model_not_found" in low or "does not exist or you do not have access" in low:
        return HTTPException(
            status_code=502,
            detail={"error": "ModelUnavailable", "message": (
                "The configured model is not available from the provider. "
                "Set GEMINI_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL / GROQ_MODEL "
                f"to a supported model. Provider said: {msg}"
            )},
        )
    return HTTPException(
        status_code=500,
        detail={"error": "QueryError", "message": msg},
    )


def get_pipeline(collection_name: str, top_k: int):
    """Get or create a RAG pipeline for the collection."""
    cache_key = f"{collection_name}_{top_k}"
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = create_rag_pipeline(
            collection_name=collection_name,
            top_k=top_k,
        )
    return _pipeline_cache[cache_key]


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG pipeline.
    
    Retrieves relevant documents and generates an answer
    using the configured LLM.
    
    - **question**: The question to ask
    - **collection_name**: Document collection to search
    - **top_k**: Number of documents to retrieve
    - **include_sources**: Include source documents in response
    - **api_key**: User-provided API key (BYOK)
    - **provider**: LLM provider (gemini, openai, anthropic, groq)
    """
    try:
        logger.info(f"Query received: {request.question[:50]}...")
        
        # Get pipeline
        pipeline = get_pipeline(request.collection_name, request.top_k)

        # Execute query with user API key and provider.
        # Retrieval + LLM call are synchronous and blocking, so run them in a
        # worker thread to keep the event loop free for other requests.
        result = await run_in_threadpool(
            pipeline.query,
            question=request.question,
            include_sources=request.include_sources,
            api_key=request.api_key,
            provider=request.provider,
        )
        
        # Build response
        sources = None
        if request.include_sources and result.get("sources"):
            sources = [
                SourceDocument(
                    content=s["content"],
                    metadata=s["metadata"],
                    score=s.get("score"),
                )
                for s in result["sources"]
            ]
        
        return QueryResponse(
            answer=result["answer"],
            question=result["question"],
            sources=sources,
            num_sources=result.get("num_sources"),
            cached=result.get("cached", False),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise _query_http_error(e)


@router.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream query response using Server-Sent Events (SSE).
    
    Returns a streaming response with chunks of the answer
    as they are generated. Format is SSE-compatible.
    """
    try:
        logger.info(f"Streaming query: {request.question[:50]}...")
        
        pipeline = get_pipeline(request.collection_name, request.top_k)
        
        async def generate():
            """Generate SSE-formatted response chunks."""
            try:
                # pipeline.stream() is a synchronous generator whose every step
                # (retrieval, then each LLM token batch) blocks. Drive it from a
                # worker thread so the event loop stays responsive.
                sync_chunks = pipeline.stream(
                    request.question,
                    api_key=request.api_key,
                    provider=request.provider,
                )
                async for chunk in iterate_in_threadpool(sync_chunks):
                    # SSE format: data: <content>\n\n
                    yield f"data: {chunk}\n\n"
                # Signal completion
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Stream query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "StreamError", "message": str(e)}
        )

