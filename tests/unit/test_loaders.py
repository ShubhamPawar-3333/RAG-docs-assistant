"""
Unit Tests for Document Processing

Tests for document loaders and chunking functionality.
"""
import pytest

class TestDocumentLoaders:
    """Tests for document loader functionality."""
    
    def test_multiformat_loader_loads_txt_file(self, sample_documents):
        """Test that MultiFormatDocumentLoader can load a .txt file."""
        from src.rag.loaders import MultiFormatDocumentLoader
        
        loader = MultiFormatDocumentLoader()
        txt_file = sample_documents / "sample.txt"
        
        documents = loader.load_file(txt_file)
        
        assert len(documents) == 1
        assert "RAG" in documents[0].page_content
        assert documents[0].metadata["source"] == str(txt_file)
    
    def test_multiformat_loader_loads_md_file(self, sample_documents):
        """Test that MultiFormatDocumentLoader can load a .md file."""
        from src.rag.loaders import MultiFormatDocumentLoader
        
        loader = MultiFormatDocumentLoader()
        md_file = sample_documents / "guide.md"
        
        documents = loader.load_file(md_file)
        
        assert len(documents) >= 1
        assert "User Guide" in documents[0].page_content
    
    def test_loader_handles_missing_file(self):
        """Test that loader raises error for missing file."""
        from src.rag.loaders import MultiFormatDocumentLoader, DocumentLoaderError
        
        loader = MultiFormatDocumentLoader()
        
        with pytest.raises(DocumentLoaderError):
            loader.load_file("/nonexistent/file.txt")
    
    def test_loader_get_supported_extensions(self):
        """Test getting supported file extensions."""
        from src.rag.loaders import MultiFormatDocumentLoader
        
        extensions = MultiFormatDocumentLoader.get_supported_extensions()
        
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".pdf" in extensions
    
    def test_load_documents_convenience_function(self, sample_documents):
        """Test the load_documents convenience function."""
        from src.rag.loaders import load_documents
        
        txt_file = sample_documents / "sample.txt"
        
        documents = load_documents(txt_file)
        
        assert len(documents) >= 1
        assert "RAG" in documents[0].page_content


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
