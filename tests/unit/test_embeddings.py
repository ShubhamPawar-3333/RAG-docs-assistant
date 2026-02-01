"""
Unit Tests for Embeddings

Tests for embedding model functionality.
"""

class TestEmbeddings:
    """Tests for embedding functionality."""
    
    def test_embeddings_initialization(self):
        """Test that embeddings can be initialized."""
        from src.rag.embeddings import get_embeddings
        
        embeddings = get_embeddings()
        
        assert embeddings is not None
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        from src.rag.embeddings import get_embeddings
        
        embeddings = get_embeddings()
        
        text = "This is a test sentence."
        vector = embeddings.embed_query(text)
        
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        from src.rag.embeddings import get_embeddings
        
        embeddings = get_embeddings()
        
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vectors = embeddings.embed_documents(texts)
        
        assert len(vectors) == 3
        assert all(len(v) == len(vectors[0]) for v in vectors)
    
    def test_similar_texts_have_similar_embeddings(self):
        """Test that semantically similar texts have similar embeddings."""
        from src.rag.embeddings import get_embeddings
        import numpy as np
        
        embeddings = get_embeddings()
        
        # Similar sentences
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        # Different sentence
        text3 = "The stock market crashed today."
        
        vec1 = embeddings.embed_query(text1)
        vec2 = embeddings.embed_query(text2)
        vec3 = embeddings.embed_query(text3)
        
        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_12 = cosine_sim(vec1, vec2)
        sim_13 = cosine_sim(vec1, vec3)
        
        # Similar sentences should have higher similarity
        assert sim_12 > sim_13
    
    def test_empty_text_embedding(self):
        """Test embedding empty text."""
        from src.rag.embeddings import get_embeddings
        
        embeddings = get_embeddings()
        
        # Should handle empty text gracefully
        vector = embeddings.embed_query("")
        
        assert isinstance(vector, list)
