"""
Unit Tests for RAG Pipeline

Tests for the core RAG pipeline functionality.
"""

from unittest.mock import patch, MagicMock


class TestRAGPipeline:
    """Tests for RAG pipeline."""
    
    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized."""
        from src.rag.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(
            collection_name="test_collection",
            top_k=3,
        )
        
        assert pipeline.collection_name == "test_collection"
        assert pipeline.top_k == 3
    
    def test_pipeline_builder_pattern(self):
        """Test the builder pattern for pipeline creation."""
        from src.rag.pipeline import RAGPipelineBuilder
        
        pipeline = (
            RAGPipelineBuilder()
            .with_collection("my_docs")
            .with_model("gemini-2.5-flash")
            .with_temperature(0.5)
            .with_top_k(5)
            .build()
        )
        
        assert pipeline.collection_name == "my_docs"
        assert pipeline.llm_model == "gemini-2.5-flash"
        assert pipeline.temperature == 0.5
        assert pipeline.top_k == 5
    
    def test_format_docs(self):
        """Test document formatting for context."""
        from src.rag.pipeline import RAGPipeline
        from langchain_core.documents import Document
        
        pipeline = RAGPipeline()
        
        docs = [
            Document(page_content="First document content."),
            Document(page_content="Second document content."),
        ]
        
        formatted = pipeline._format_docs(docs)
        
        assert "First document" in formatted
        assert "Second document" in formatted
        assert "---" in formatted  # Separator
    
    def test_get_pipeline_info(self):
        """Test getting pipeline configuration info."""
        from src.rag.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(
            collection_name="test",
            top_k=7,
        )
        
        info = pipeline.get_pipeline_info()
        
        assert info["collection_name"] == "test"
        assert info["top_k"] == 7
        assert "llm_model" in info
        assert "embedding_model" in info
    
    @patch("src.rag.pipeline.RAGPipeline.retriever")
    @patch("src.rag.pipeline.RAGPipeline._build_chain_with_key")
    def test_query_with_mocked_components(self, mock_build_chain, mock_retriever):
        """Test query with mocked retriever and chain."""
        from src.rag.pipeline import RAGPipeline
        from src.rag.retrieval import RetrievalResult
        from langchain_core.documents import Document
        
        # Setup mocks
        mock_result = RetrievalResult(
            documents=[Document(page_content="Test content")],
            scores=[0.9],
            query="test query",
        )
        mock_retriever.retrieve.return_value = mock_result
        
        # Mock the chain built with user's API key
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is the answer."
        mock_build_chain.return_value = mock_chain
        
        pipeline = RAGPipeline()
        
        # Pass api_key and provider (required in BYOK mode)
        result = pipeline.query("What is test?", include_sources=True, api_key="test-api-key", provider="gemini")
        
        assert "answer" in result
        assert result["answer"] == "This is the answer."
    
    def test_create_rag_pipeline_convenience(self):
        """Test the convenience function for creating pipelines."""
        from src.rag.pipeline import create_rag_pipeline
        
        pipeline = create_rag_pipeline(
            collection_name="quick_test",
            top_k=3,
        )
        
        assert pipeline.collection_name == "quick_test"
        assert pipeline.top_k == 3


class TestRetriever:
    """Tests for retrieval functionality."""
    
    def test_retrieval_result_structure(self):
        """Test RetrievalResult data structure."""
        from src.rag.retrieval import RetrievalResult
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]
        
        result = RetrievalResult(
            documents=docs,
            scores=[0.9, 0.8],
            query="test",
        )
        
        assert result.num_results == 2
        assert result.num_results > 0  # Has results
        
        context = result.get_context(separator="\n---\n")
        assert "Doc 1" in context
        assert "Doc 2" in context
    
    def test_empty_retrieval_result(self):
        """Test empty retrieval result."""
        from src.rag.retrieval import RetrievalResult
        
        result = RetrievalResult(
            documents=[],
            scores=[],
            query="test",
        )
        
        assert result.num_results == 0
        assert result.num_results == 0  # No results
        assert result.get_context() == ""

