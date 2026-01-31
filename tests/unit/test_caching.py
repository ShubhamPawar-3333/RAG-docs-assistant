"""
Unit tests for the caching module.
"""

import time
import pytest
from src.rag.caching import (
    CacheEntry,
    InMemoryCache,
    QueryCache,
    create_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            sources=[{"source": "doc1.pdf", "page": 1}],
            collection_name="test_collection",
            created_at=time.time(),
            ttl=3600,
        )
        
        assert entry.query == "What is RAG?"
        assert entry.answer == "RAG stands for Retrieval-Augmented Generation."
        assert len(entry.sources) == 1
        assert entry.hit_count == 0
    
    def test_cache_entry_not_expired(self):
        """Test that a fresh entry is not expired."""
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time(),
            ttl=3600,
        )
        
        assert not entry.is_expired
    
    def test_cache_entry_expired(self):
        """Test that an old entry is expired."""
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time() - 7200,  # 2 hours ago
            ttl=3600,  # 1 hour TTL
        )
        
        assert entry.is_expired
    
    def test_cache_entry_serialization(self):
        """Test converting to and from dict."""
        entry = CacheEntry(
            query="What is RAG?",
            answer="RAG is...",
            sources=[{"source": "doc.pdf"}],
            collection_name="test",
            created_at=1234567890.0,
            ttl=3600,
            hit_count=5,
        )
        
        data = entry.to_dict()
        restored = CacheEntry.from_dict(data)
        
        assert restored.query == entry.query
        assert restored.answer == entry.answer
        assert restored.hit_count == entry.hit_count


class TestInMemoryCache:
    """Tests for InMemoryCache backend."""
    
    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = InMemoryCache()
        
        entry = CacheEntry(
            query="test query",
            answer="test answer",
            sources=[],
            collection_name="test",
            created_at=time.time(),
            ttl=3600,
        )
        
        cache.set("key1", entry)
        result = cache.get("key1")
        
        assert result is not None
        assert result.query == "test query"
        assert result.answer == "test answer"
    
    def test_get_missing_key(self):
        """Test getting a non-existent key."""
        cache = InMemoryCache()
        
        result = cache.get("nonexistent")
        
        assert result is None
    
    def test_get_expired_entry(self):
        """Test that expired entries return None."""
        cache = InMemoryCache()
        
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time() - 7200,
            ttl=3600,
        )
        
        cache.set("expired_key", entry)
        result = cache.get("expired_key")
        
        assert result is None
    
    def test_delete(self):
        """Test deleting a cache entry."""
        cache = InMemoryCache()
        
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time(),
            ttl=3600,
        )
        
        cache.set("key1", entry)
        assert cache.get("key1") is not None
        
        deleted = cache.delete("key1")
        assert deleted is True
        assert cache.get("key1") is None
    
    def test_clear(self):
        """Test clearing all cache entries."""
        cache = InMemoryCache()
        
        for i in range(5):
            entry = CacheEntry(
                query=f"query{i}",
                answer=f"answer{i}",
                sources=[],
                collection_name="test",
                created_at=time.time(),
                ttl=3600,
            )
            cache.set(f"key{i}", entry)
        
        cache.clear()
        
        for i in range(5):
            assert cache.get(f"key{i}") is None
    
    def test_max_size_eviction(self):
        """Test that oldest entries are evicted at max size."""
        cache = InMemoryCache(max_size=3)
        
        for i in range(5):
            entry = CacheEntry(
                query=f"query{i}",
                answer=f"answer{i}",
                sources=[],
                collection_name="test",
                created_at=time.time() + i,  # Increasing timestamps
                ttl=3600,
            )
            cache.set(f"key{i}", entry)
        
        # First two entries should be evicted
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        # Last three should remain
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
    
    def test_hit_count_increment(self):
        """Test that hit count increments on get."""
        cache = InMemoryCache()
        
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time(),
            ttl=3600,
        )
        
        cache.set("key1", entry)
        
        cache.get("key1")
        cache.get("key1")
        result = cache.get("key1")
        
        assert result.hit_count == 3
    
    def test_stats(self):
        """Test getting cache statistics."""
        cache = InMemoryCache(max_size=100)
        
        entry = CacheEntry(
            query="test",
            answer="answer",
            sources=[],
            collection_name="test",
            created_at=time.time(),
            ttl=3600,
        )
        
        cache.set("key1", entry)
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("missing")  # miss
        
        stats = cache.stats()
        
        assert stats["backend"] == "in-memory"
        assert stats["entries"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 2
        assert stats["misses"] == 1


class TestQueryCache:
    """Tests for QueryCache main class."""
    
    def test_get_and_set(self):
        """Test basic get and set with query parameters."""
        cache = QueryCache()
        
        cache.set(
            query="What is RAG?",
            collection_name="docs",
            answer="RAG is...",
            sources=[{"source": "doc.pdf"}],
        )
        
        result = cache.get("What is RAG?", "docs")
        
        assert result is not None
        assert result["answer"] == "RAG is..."
        assert result["cached"] is True
    
    def test_get_missing(self):
        """Test getting a non-cached query."""
        cache = QueryCache()
        
        result = cache.get("Unknown query", "docs")
        
        assert result is None
    
    def test_query_normalization(self):
        """Test that queries are normalized for matching."""
        cache = QueryCache()
        
        cache.set(
            query="What is RAG?",
            collection_name="docs",
            answer="RAG is...",
            sources=[],
        )
        
        # Different case and whitespace
        result = cache.get("  WHAT IS RAG?  ", "docs")
        
        assert result is not None
    
    def test_different_collections(self):
        """Test that different collections have separate caches."""
        cache = QueryCache()
        
        cache.set(
            query="What is X?",
            collection_name="docs1",
            answer="Answer from docs1",
            sources=[],
        )
        cache.set(
            query="What is X?",
            collection_name="docs2",
            answer="Answer from docs2",
            sources=[],
        )
        
        result1 = cache.get("What is X?", "docs1")
        result2 = cache.get("What is X?", "docs2")
        
        assert result1["answer"] == "Answer from docs1"
        assert result2["answer"] == "Answer from docs2"
    
    def test_get_or_compute_cache_hit(self):
        """Test get_or_compute returns cached result."""
        cache = QueryCache()
        compute_count = 0
        
        def compute_fn():
            nonlocal compute_count
            compute_count += 1
            return {"answer": "Computed answer", "sources": []}
        
        # First call: computes
        result1 = cache.get_or_compute(
            query="What is RAG?",
            collection_name="docs",
            compute_fn=compute_fn,
        )
        
        # Second call: returns cached
        result2 = cache.get_or_compute(
            query="What is RAG?",
            collection_name="docs",
            compute_fn=compute_fn,
        )
        
        assert compute_count == 1  # Only computed once
        assert result1["cached"] is False
        assert result2["cached"] is True
    
    def test_disabled_cache(self):
        """Test that disabled cache always misses."""
        cache = QueryCache(enabled=False)
        
        cache.set(
            query="test",
            collection_name="docs",
            answer="answer",
            sources=[],
        )
        
        result = cache.get("test", "docs")
        
        assert result is None
    
    def test_custom_ttl(self):
        """Test setting custom TTL."""
        cache = QueryCache(default_ttl=60)
        
        cache.set(
            query="test",
            collection_name="docs",
            answer="answer",
            sources=[],
            ttl=1,  # 1 second
        )
        
        # Should be available immediately
        assert cache.get("test", "docs") is not None
        
        # Wait for expiration
        time.sleep(1.5)
        assert cache.get("test", "docs") is None
    
    def test_invalidate(self):
        """Test invalidating a specific cache entry."""
        cache = QueryCache()
        
        cache.set(
            query="test",
            collection_name="docs",
            answer="answer",
            sources=[],
        )
        
        assert cache.get("test", "docs") is not None
        
        cache.invalidate("test", "docs")
        
        assert cache.get("test", "docs") is None
    
    def test_invalidate_collection(self):
        """Test invalidating all entries for a collection."""
        cache = QueryCache()
        
        for i in range(3):
            cache.set(
                query=f"query{i}",
                collection_name="target",
                answer=f"answer{i}",
                sources=[],
            )
        
        cache.set(
            query="other",
            collection_name="keep",
            answer="kept",
            sources=[],
        )
        
        cache.invalidate_collection("target")
        
        # Target collection entries gone
        for i in range(3):
            assert cache.get(f"query{i}", "target") is None
        
        # Other collection preserved
        assert cache.get("other", "keep") is not None
    
    def test_stats(self):
        """Test getting cache statistics."""
        cache = QueryCache(default_ttl=7200)
        
        stats = cache.stats()
        
        assert stats["enabled"] is True
        assert stats["default_ttl"] == 7200


class TestCreateCache:
    """Tests for create_cache factory function."""
    
    def test_create_memory_cache(self):
        """Test creating in-memory cache."""
        cache = create_cache("memory", max_size=500, ttl=1800)
        
        assert isinstance(cache, QueryCache)
        assert isinstance(cache.backend, InMemoryCache)
        assert cache.default_ttl == 1800
    
    def test_create_cache_defaults(self):
        """Test default cache creation."""
        cache = create_cache()
        
        assert isinstance(cache.backend, InMemoryCache)
        assert cache.enabled is True
    
    def test_create_cache_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_cache("invalid_backend")
