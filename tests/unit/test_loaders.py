"""
Unit Tests for Document Processing

Tests for document loaders and chunking functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestDocumentLoaders:
    """Tests for document loader functionality."""
    
    def test_text_loader_loads_file(self, sample_documents):
        """Test that TextLoader can load a .txt file."""
        from src.rag.loaders import TextLoader
        
        loader = TextLoader()
        txt_file = sample_documents / "sample.txt"
        
        documents = loader.load(str(txt_file))
        
        assert len(documents) == 1
        assert "RAG" in documents[0].page_content
        assert documents[0].metadata["source"] == str(txt_file)
    
    def test_markdown_loader_loads_file(self, sample_documents):
        """Test that MarkdownLoader can load a .md file."""
        from src.rag.loaders import MarkdownLoader
        
        loader = MarkdownLoader()
        md_file = sample_documents / "guide.md"
        
        documents = loader.load(str(md_file))
        
        assert len(documents) >= 1
        assert "User Guide" in documents[0].page_content
    
    def test_loader_handles_missing_file(self):
        """Test that loader raises error for missing file."""
        from src.rag.loaders import TextLoader
        
        loader = TextLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")
    
    def test_document_loader_factory(self, sample_documents):
        """Test the DocumentLoaderFactory selects correct loader."""
        from src.rag.loaders import DocumentLoaderFactory
        
        txt_file = sample_documents / "sample.txt"
        md_file = sample_documents / "guide.md"
        
        # Should select TextLoader for .txt
        txt_loader = DocumentLoaderFactory.get_loader(str(txt_file))
        assert txt_loader is not None
        
        # Should select MarkdownLoader for .md
        md_loader = DocumentLoaderFactory.get_loader(str(md_file))
        assert md_loader is not None


class TestChunking:
    """Tests for document chunking functionality."""
    
    def test_chunker_splits_document(self):
        """Test that chunker splits large documents."""
        from src.rag.chunking import DocumentChunker
        from langchain_core.documents import Document
        
        # Create a long document
        long_text = "This is a sentence. " * 100
        doc = Document(page_content=long_text, metadata={"source": "test"})
        
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents([doc])
        
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 250 for c in chunks)  # Allow some margin
    
    def test_chunker_preserves_metadata(self):
        """Test that chunker preserves document metadata."""
        from src.rag.chunking import DocumentChunker
        from langchain_core.documents import Document
        
        doc = Document(
            page_content="Short document content.",
            metadata={"source": "test.txt", "custom": "value"}
        )
        
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk_documents([doc])
        
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["custom"] == "value"
    
    def test_chunker_handles_empty_document(self):
        """Test that chunker handles empty documents gracefully."""
        from src.rag.chunking import DocumentChunker
        from langchain_core.documents import Document
        
        doc = Document(page_content="", metadata={})
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents([doc])
        
        # Empty docs should produce no chunks or one empty chunk
        assert len(chunks) <= 1
