"""
Pydantic Models for API Request/Response

Defines the data structures for API communication.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============== Query Models ==============

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(
        ...,
        description="The question to ask about the documents",
        min_length=1,
        max_length=2000,
        examples=["What is the refund policy?"]
    )
    collection_name: str = Field(
        default="documents",
        description="Name of the document collection to search"
    )
    top_k: int = Field(
        default=5,
        description="Number of documents to retrieve",
        ge=1,
        le=20
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source documents in response"
    )


class SourceDocument(BaseModel):
    """A source document used in the answer."""
    content: str = Field(..., description="Document content snippet")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    score: Optional[float] = Field(
        None,
        description="Relevance score"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    sources: Optional[List[SourceDocument]] = Field(
        None,
        description="Source documents used"
    )
    num_sources: Optional[int] = Field(
        None,
        description="Number of sources used"
    )


# ============== Ingest Models ==============

class IngestRequest(BaseModel):
    """Request model for file ingestion."""
    collection_name: str = Field(
        default="documents",
        description="Name of the collection to ingest into"
    )
    chunk_size: int = Field(
        default=1000,
        description="Size of document chunks",
        ge=100,
        le=4000
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks",
        ge=0,
        le=500
    )


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    success: bool = Field(..., description="Whether ingestion succeeded")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed"
    )
    chunks_created: int = Field(
        default=0,
        description="Number of chunks created"
    )
    collection_name: str = Field(
        ...,
        description="Collection documents were added to"
    )


class IngestTextRequest(BaseModel):
    """Request model for text ingestion."""
    text: str = Field(
        ...,
        description="Text content to ingest",
        min_length=1,
        max_length=100000
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for the document"
    )
    collection_name: str = Field(
        default="documents",
        description="Name of the collection"
    )


# ============== Health Models ==============

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")


class ComponentHealth(BaseModel):
    """Health status of a component."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    details: Optional[str] = Field(None, description="Additional details")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    components: List[ComponentHealth] = Field(
        ...,
        description="Component health statuses"
    )


# ============== Error Models ==============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
