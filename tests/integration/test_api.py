"""
Integration Tests for FastAPI Endpoints

Tests for API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client for the API."""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_detailed(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code in [200, 503]
    
    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200


class TestQueryEndpoints:
    """Tests for query endpoints."""
    
    @patch("src.api.routes.query.RAGPipeline")
    def test_query_endpoint(self, mock_pipeline_class, client):
        """Test the /api/query endpoint."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "This is a test answer.",
            "question": "Test question?",
            "sources": [],
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        response = client.post(
            "/api/query",
            json={
                "question": "Test question?",
                "collection_name": "test_collection",
                "top_k": 5,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_validation_error(self, client):
        """Test query with invalid request body."""
        response = client.post(
            "/api/query",
            json={}  # Missing required 'question' field
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_empty_question(self, client):
        """Test query with empty question."""
        response = client.post(
            "/api/query",
            json={"question": ""}
        )
        
        # Should return 422 or handle gracefully
        assert response.status_code in [200, 422]


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""
    
    @patch("src.api.routes.ingest.DocumentIngester")
    def test_ingest_text(self, mock_ingester_class, client):
        """Test the /api/ingest/text endpoint."""
        mock_ingester = MagicMock()
        mock_ingester.ingest_text.return_value = {
            "success": True,
            "chunks_created": 3,
        }
        mock_ingester_class.return_value = mock_ingester
        
        response = client.post(
            "/api/ingest/text",
            json={
                "text": "This is test content to ingest.",
                "collection_name": "test_collection",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_ingest_files_no_files(self, client):
        """Test file ingestion with no files."""
        response = client.post("/api/ingest")
        
        # Should fail without files
        assert response.status_code == 422


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.get("/api/query")  # Should be POST
        
        assert response.status_code == 405
    
    @patch("src.api.routes.query.RAGPipeline")
    def test_internal_error_handling(self, mock_pipeline_class, client):
        """Test that internal errors are handled gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = Exception("Internal error")
        mock_pipeline_class.return_value = mock_pipeline
        
        response = client.post(
            "/api/query",
            json={"question": "Test?"}
        )
        
        # Should return 500 with error details
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data
