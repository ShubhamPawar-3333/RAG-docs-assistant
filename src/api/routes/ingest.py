"""
Ingest Routes

Handles document ingestion into the vector store.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from langchain_core.documents import Document

from src.api.models import IngestResponse, IngestTextRequest, ErrorResponse
from src.rag.loaders import MultiFormatDocumentLoader
from src.rag.chunking import DocumentChunker
from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import create_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

# Addition
TMP_DIR = Path(
    os.getenv("DOCUMIND_TMP_DIR", tempfile.gettempdir())
)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Ingestion failed"}
    }
)
async def ingest_files(
    files: List[UploadFile] = File(..., description="Files to ingest"),
    collection_name: str = Form(default="documents"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
):
    """
    Ingest files into the vector store.
    
    Accepts PDF, Markdown, and text files. Files are chunked
    and stored with embeddings for retrieval.
    
    - **files**: Files to upload and process
    - **collection_name**: Target collection name
    - **chunk_size**: Size of document chunks
    - **chunk_overlap**: Overlap between chunks
    """
    try:
        logger.info("INGEST ENDPOINT HIT")
        logger.info(f"Ingesting {len(files)} file(s) to collection '{collection_name}'")
        
        all_documents = []
        loader = MultiFormatDocumentLoader()
        
        for file in files:
            # Validate file type
            ext = Path(file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "InvalidFileType",
                        "message": f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
                    }
                )
            
            # Save to temp file and load
            # with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=TMP_DIR) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                docs = loader.load_file(tmp_path)
                # Update metadata with original filename
                for doc in docs:
                    doc.metadata["file_name"] = file.filename
                    doc.metadata["original_file"] = file.filename
                all_documents.extend(docs)
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        
        if not all_documents:
            return IngestResponse(
                success=True,
                message="No documents to process",
                documents_processed=0,
                chunks_created=0,
                collection_name=collection_name,
            )
        
        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk_documents(all_documents)
        
        # Store in vector store
        embeddings = get_embeddings()
        store = create_vector_store(
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        store.add_documents(chunks)
        
        logger.info(
            f"Ingested {len(all_documents)} documents, "
            f"created {len(chunks)} chunks"
        )
        
        return IngestResponse(
            success=True,
            message="Files ingested successfully",
            documents_processed=len(all_documents),
            chunks_created=len(chunks),
            collection_name=collection_name,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "IngestionError", "message": str(e)}
        )


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Ingestion failed"}
    }
)
async def ingest_text(request: IngestTextRequest):
    """
    Ingest raw text into the vector store.
    
    Useful for adding content programmatically without file upload.
    
    - **text**: The text content to ingest
    - **metadata**: Optional metadata for the document
    - **collection_name**: Target collection name
    """
    try:
        logger.info(f"Ingesting text ({len(request.text)} chars) to '{request.collection_name}'")
        
        # Create document from text
        doc = Document(
            page_content=request.text,
            metadata=request.metadata,
        )
        
        # Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents([doc])
        
        # Store
        embeddings = get_embeddings()
        store = create_vector_store(
            embedding_function=embeddings,
            collection_name=request.collection_name,
        )
        store.add_documents(chunks)
        
        return IngestResponse(
            success=True,
            message="Text ingested successfully",
            documents_processed=1,
            chunks_created=len(chunks),
            collection_name=request.collection_name,
        )
        
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "IngestionError", "message": str(e)}
        )


@router.delete("/ingest/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection.
    
    Removes all documents and the collection from the vector store.
    """
    try:
        logger.info(f"Deleting collection: {collection_name}")
        
        embeddings = get_embeddings()
        store = create_vector_store(
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        store.delete_collection()
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted",
        }
        
    except Exception as e:
        logger.error(f"Delete collection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "DeleteError", "message": str(e)}
        )
