"""
Pytest Configuration

Shared fixtures and configuration for all tests.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    docs_dir = temp_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    # Create a sample text file
    txt_file = docs_dir / "sample.txt"
    txt_file.write_text(
        "This is a sample document about RAG systems.\n"
        "RAG stands for Retrieval-Augmented Generation.\n"
        "It combines retrieval with language model generation."
    )
    
    # Create a sample markdown file
    md_file = docs_dir / "guide.md"
    md_file.write_text(
        "# User Guide\n\n"
        "## Getting Started\n\n"
        "This guide explains how to use the DocuMind AI system.\n\n"
        "## Features\n\n"
        "- Document upload\n"
        "- Question answering\n"
        "- Source citations\n"
    )
    
    return docs_dir


@pytest.fixture
def mock_settings(monkeypatch, temp_dir):
    """Mock settings for testing."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(temp_dir / "chroma"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    

@pytest.fixture
def test_collection_name():
    """Return a unique test collection name."""
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"
