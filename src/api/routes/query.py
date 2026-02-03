"""
Query Routes

Handles document querying with RAG pipeline.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import QueryRequest, QueryResponse, SourceDocument, ErrorResponse
from src.rag.pipeline import create_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache pipeline instances per collection
_pipeline_cache = {}


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
    - **api_key**: Optional user-provided Gemini API key (BYOK)
    """
    try:
        logger.info(f"Query received: {request.question[:50]}...")
        
        # Get pipeline
        pipeline = get_pipeline(request.collection_name, request.top_k)
        
        # Execute query with optional user API key
        result = pipeline.query(
            question=request.question,
            include_sources=request.include_sources,
            api_key=request.api_key,  # Pass user's API key if provided
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
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "QueryError", "message": str(e)}
        )


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
                for chunk in pipeline.stream(request.question):
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

