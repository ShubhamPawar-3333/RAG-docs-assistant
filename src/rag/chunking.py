"""
Text Chunking Module

Handles splitting documents into smaller chunks for embedding and retrieval.
Supports multiple chunking strategies:
- Recursive character splitting
- Semantic splitting (sentence-aware)
- Token-based splitting
"""

import logging
from enum import Enum
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"      # Character-based with smart separators
    SEMANTIC = "semantic"        # Sentence-aware splitting
    TOKEN = "token"              # Token-count based


class DocumentChunker:
    """
    Splits documents into smaller chunks for embedding and retrieval.
    
    The chunker uses overlapping windows to preserve context across
    chunk boundaries, which is critical for RAG quality.
    
    Example:
        >>> chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        >>> chunks = chunker.chunk_documents(documents)
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (default from settings).
            chunk_overlap: Overlap between consecutive chunks.
            strategy: Chunking strategy to use.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy
        
        # Initialize the appropriate splitter
        self._splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create the text splitter based on strategy."""
        
        if self.strategy == ChunkingStrategy.RECURSIVE:
            # Recursive splitter with smart separators
            # Tries to split on paragraphs first, then sentences, then words
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",      # Paragraphs
                    "\n",        # Lines
                    ". ",        # Sentences
                    "! ",        # Exclamations
                    "? ",        # Questions
                    "; ",        # Semicolons
                    ", ",        # Commas
                    " ",         # Words
                    "",          # Characters (last resort)
                ],
                is_separator_regex=False,
            )
        
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            # Sentence-aware splitting using transformer tokenizer
            # Best for preserving semantic meaning
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",      # Paragraphs first
                    "\n",        # Then lines
                    "(?<=[.!?]) ",  # Sentence boundaries (regex)
                    " ",         # Words
                    "",          # Characters
                ],
                is_separator_regex=True,
            )
        
        elif self.strategy == ChunkingStrategy.TOKEN:
            # Token-based splitting (useful for staying within model limits)
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=min(self.chunk_overlap, 50),  # Token overlap
                tokens_per_chunk=self.chunk_size // 4,  # Approximate conversion
            )
        
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a single document into chunks.
        
        Args:
            document: Document to split.
            
        Returns:
            List of Document chunks with preserved metadata.
        """
        # Split the document
        chunks = self._splitter.split_documents([document])
        
        # Enrich metadata with chunk information
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_strategy"] = self.strategy.value
        
        logger.debug(
            f"Split document '{document.metadata.get('file_name', 'unknown')}' "
            f"into {len(chunks)} chunks"
        )
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of all chunks from all documents.
        """
        all_chunks: List[Document] = []
        
        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    f"Error chunking document "
                    f"'{document.metadata.get('file_name', 'unknown')}': {e}"
                )
                continue
        
        logger.info(
            f"Chunked {len(documents)} document(s) into {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Calculate statistics about the chunks.
        
        Args:
            chunks: List of document chunks.
            
        Returns:
            Dictionary with chunk statistics.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0,
            }
        
        sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
        }


# Convenience function
def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: str = "recursive"
) -> List[Document]:
    """
    Convenience function to chunk documents.
    
    Args:
        documents: List of documents to chunk.
        chunk_size: Maximum chunk size (default from settings).
        chunk_overlap: Overlap between chunks (default from settings).
        strategy: "recursive", "semantic", or "token".
        
    Returns:
        List of document chunks.
    """
    # Convert string to enum
    strategy_enum = ChunkingStrategy(strategy)
    
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy_enum,
    )
    
    return chunker.chunk_documents(documents)
