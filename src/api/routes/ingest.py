"""
Ingest Routes

Handles document ingestion into the vector store.

Loading, chunking, embedding and writing to ChromaDB are all CPU/IO-bound and
fully synchronous. They are dispatched to a worker thread with
``run_in_threadpool`` so a single ingest request cannot block the event loop
(and therefore every other request, including ``/health``) for its whole
duration.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from config.settings import settings
from src.api.models import IngestResponse, IngestTextRequest, ErrorResponse
from src.rag.loaders import MultiFormatDocumentLoader
from src.rag.chunking import DocumentChunker
from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import create_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

TMP_DIR = Path(os.getenv("DOCUMIND_TMP_DIR", tempfile.gettempdir()))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}

_MB = 1024 * 1024


def _process_documents(
    saved_files: List[Tuple[str, str]],
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResponse:
    """
    Synchronous ingestion work: load -> chunk -> embed -> store.

    Runs in a worker thread. ``saved_files`` is a list of
    ``(temp_path, original_filename)`` tuples.
    """
    loader = MultiFormatDocumentLoader()
    all_documents: List[Document] = []

    for tmp_path, original_name in saved_files:
        docs = loader.load_file(tmp_path)
        for doc in docs:
            doc.metadata["file_name"] = original_name
            doc.metadata["original_file"] = original_name
        all_documents.extend(docs)

    if not all_documents:
        return IngestResponse(
            success=True,
            message="No documents to process",
            documents_processed=0,
            chunks_created=0,
            collection_name=collection_name,
        )

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(all_documents)

    store = create_vector_store(
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )
    store.add_documents(chunks)

    logger.info(
        f"Ingested {len(all_documents)} documents, created {len(chunks)} chunks "
        f"into '{collection_name}'"
    )
    return IngestResponse(
        success=True,
        message="Files ingested successfully",
        documents_processed=len(all_documents),
        chunks_created=len(chunks),
        collection_name=collection_name,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        500: {"model": ErrorResponse, "description": "Ingestion failed"},
    },
)
async def ingest_files(
    files: List[UploadFile] = File(..., description="Files to ingest"),
    collection_name: str = Form(default="documents"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
):
    """
    Ingest files into the vector store.

    Accepts PDF, Markdown, and text files. Files are chunked and stored with
    embeddings for retrieval.

    - **files**: Files to upload and process
    - **collection_name**: Target collection name
    - **chunk_size**: Size of document chunks
    - **chunk_overlap**: Overlap between chunks
    """
    logger.info(
        f"Ingesting {len(files)} file(s) to collection '{collection_name}'"
    )

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "TooManyFiles",
                "message": (
                    f"{len(files)} files exceeds the limit of "
                    f"{settings.max_upload_files} per request."
                ),
            },
        )

    max_file_bytes = settings.max_file_size_mb * _MB
    max_total_bytes = settings.max_total_upload_mb * _MB

    saved_files: List[Tuple[str, str]] = []
    total_bytes = 0
    try:
        for file in files:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "InvalidFileType",
                        "message": (
                            f"Unsupported file type: {ext or '(none)'}. "
                            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                        ),
                    },
                )

            content = await file.read()
            size = len(content)
            total_bytes += size

            if size > max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "FileTooLarge",
                        "message": (
                            f"'{file.filename}' is {size / _MB:.1f} MB; the limit "
                            f"is {settings.max_file_size_mb} MB per file."
                        ),
                    },
                )
            if total_bytes > max_total_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "UploadTooLarge",
                        "message": (
                            f"Total upload exceeds {settings.max_total_upload_mb} MB."
                        ),
                    },
                )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext, dir=TMP_DIR
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            saved_files.append((tmp_path, file.filename))

        if not saved_files:
            return IngestResponse(
                success=True,
                message="No documents to process",
                documents_processed=0,
                chunks_created=0,
                collection_name=collection_name,
            )

        return await run_in_threadpool(
            _process_documents,
            saved_files,
            collection_name,
            chunk_size,
            chunk_overlap,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "IngestionError", "message": str(e)},
        )
    finally:
        for tmp_path, _ in saved_files:
            Path(tmp_path).unlink(missing_ok=True)


def _process_text(
    text: str, metadata: dict, collection_name: str
) -> IngestResponse:
    """Synchronous text ingestion work (chunk -> embed -> store)."""
    doc = Document(page_content=text, metadata=metadata)
    chunks = DocumentChunker().chunk_documents([doc])

    store = create_vector_store(
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )
    store.add_documents(chunks)

    return IngestResponse(
        success=True,
        message="Text ingested successfully",
        documents_processed=1,
        chunks_created=len(chunks),
        collection_name=collection_name,
    )


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    responses={500: {"model": ErrorResponse, "description": "Ingestion failed"}},
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
        logger.info(
            f"Ingesting text ({len(request.text)} chars) to "
            f"'{request.collection_name}'"
        )
        return await run_in_threadpool(
            _process_text,
            request.text,
            request.metadata,
            request.collection_name,
        )
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "IngestionError", "message": str(e)},
        )


@router.delete("/ingest/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection.

    Removes all documents and the collection from the vector store.
    """
    try:
        logger.info(f"Deleting collection: {collection_name}")

        def _delete() -> None:
            store = create_vector_store(
                embedding_function=get_embeddings(),
                collection_name=collection_name,
            )
            store.delete_collection()

        await run_in_threadpool(_delete)

        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted",
        }

    except Exception as e:
        logger.error(f"Delete collection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "DeleteError", "message": str(e)},
        )
