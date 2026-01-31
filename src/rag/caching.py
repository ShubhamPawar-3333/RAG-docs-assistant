"""
Query Caching Module

Provides caching for RAG queries to improve response times
and reduce API costs for repeated queries.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached query result."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    collection_name: str
    created_at: float
    ttl: int  # Time to live in seconds
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > (self.created_at + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache using a dictionary.
    
    Best for single-instance deployments and development.
    Data is lost on restart.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the in-memory cache.
        
        Args:
            max_size: Maximum number of entries to store.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry, returning None if not found or expired."""
        entry = self._cache.get(key)
        
        if entry is None:
            self._misses += 1
            return None
        
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        
        entry.hit_count += 1
        self._hits += 1
        return entry
    
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry, evicting oldest if at capacity."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key[:20]}...")
        
        self._cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("In-memory cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "backend": "in-memory",
            "entries": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


class RedisCache(CacheBackend):
    """
    Redis-based cache for distributed deployments.
    
    Requires redis-py: pip install redis
    Supports persistence and multi-instance deployments.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:cache:",
        url: Optional[str] = None,
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password.
            prefix: Key prefix for cache entries.
            url: Redis URL (overrides host/port/db/password).
        """
        self._prefix = prefix
        self._client = None
        self._connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "url": url,
        }
    
    @property
    def client(self):
        """Lazy load Redis client."""
        if self._client is None:
            try:
                import redis
                
                if self._connection_params["url"]:
                    self._client = redis.from_url(
                        self._connection_params["url"],
                        decode_responses=True,
                    )
                else:
                    self._client = redis.Redis(
                        host=self._connection_params["host"],
                        port=self._connection_params["port"],
                        db=self._connection_params["db"],
                        password=self._connection_params["password"],
                        decode_responses=True,
                    )
                # Test connection
                self._client.ping()
                logger.info("Connected to Redis cache")
            except ImportError:
                raise ImportError("redis-py not installed: pip install redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry from Redis."""
        try:
            data = self.client.get(self._make_key(key))
            if data is None:
                return None
            
            entry = CacheEntry.from_dict(json.loads(data))
            
            if entry.is_expired:
                self.delete(key)
                return None
            
            # Increment hit count
            entry.hit_count += 1
            self.set(key, entry)
            
            return entry
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry in Redis with TTL."""
        try:
            self.client.setex(
                self._make_key(key),
                entry.ttl,
                json.dumps(entry.to_dict()),
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry from Redis."""
        try:
            return self.client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries with our prefix."""
        try:
            keys = self.client.keys(f"{self._prefix}*")
            if keys:
                self.client.delete(*keys)
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            keys = self.client.keys(f"{self._prefix}*")
            info = self.client.info("stats")
            
            return {
                "backend": "redis",
                "entries": len(keys),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"backend": "redis", "error": str(e)}


class QueryCache:
    """
    Main query caching class.
    
    Wraps a cache backend and provides query-specific functionality.
    
    Example:
        >>> cache = QueryCache(backend=InMemoryCache())
        >>> result = cache.get_or_compute(
        ...     query="What is RAG?",
        ...     collection="docs",
        ...     compute_fn=lambda: pipeline.query("What is RAG?")
        ... )
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: int = 3600,  # 1 hour
        enabled: bool = True,
    ):
        """
        Initialize the query cache.
        
        Args:
            backend: Cache backend to use (defaults to InMemoryCache).
            default_ttl: Default time-to-live in seconds.
            enabled: Whether caching is enabled.
        """
        self.backend = backend or InMemoryCache()
        self.default_ttl = default_ttl
        self.enabled = enabled
    
    def _generate_key(
        self,
        query: str,
        collection_name: str,
        **kwargs,
    ) -> str:
        """Generate a unique cache key for a query."""
        key_data = {
            "query": query.lower().strip(),
            "collection": collection_name,
            **kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        collection_name: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached result.
        
        Args:
            query: The query string.
            collection_name: The collection being queried.
            **kwargs: Additional key parameters.
        
        Returns:
            Cached result dict or None if not found.
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(query, collection_name, **kwargs)
        entry = self.backend.get(key)
        
        if entry:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return {
                "answer": entry.answer,
                "sources": entry.sources,
                "cached": True,
                "cache_hit_count": entry.hit_count,
            }
        
        return None
    
    def set(
        self,
        query: str,
        collection_name: str,
        answer: str,
        sources: List[Dict[str, Any]],
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Cache a query result.
        
        Args:
            query: The query string.
            collection_name: The collection queried.
            answer: The generated answer.
            sources: Source documents.
            ttl: Time-to-live (uses default if None).
            **kwargs: Additional key parameters.
        """
        if not self.enabled:
            return
        
        key = self._generate_key(query, collection_name, **kwargs)
        
        entry = CacheEntry(
            query=query,
            answer=answer,
            sources=sources,
            collection_name=collection_name,
            created_at=time.time(),
            ttl=ttl or self.default_ttl,
        )
        
        self.backend.set(key, entry)
        logger.debug(f"Cached query: {query[:50]}...")
    
    def get_or_compute(
        self,
        query: str,
        collection_name: str,
        compute_fn: callable,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get cached result or compute and cache.
        
        This is the main method for cache-through behavior.
        
        Args:
            query: The query string.
            collection_name: The collection to query.
            compute_fn: Function to compute result if not cached.
            ttl: Time-to-live for cached result.
            **kwargs: Additional key parameters.
        
        Returns:
            Result dict with answer and sources.
        """
        # Try cache first
        cached = self.get(query, collection_name, **kwargs)
        if cached:
            return cached
        
        # Compute result
        result = compute_fn()
        
        # Cache the result
        self.set(
            query=query,
            collection_name=collection_name,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            ttl=ttl,
            **kwargs,
        )
        
        result["cached"] = False
        return result
    
    def invalidate(
        self,
        query: str,
        collection_name: str,
        **kwargs,
    ) -> bool:
        """Invalidate a specific cache entry."""
        key = self._generate_key(query, collection_name, **kwargs)
        return self.backend.delete(key)
    
    def invalidate_collection(self, collection_name: str) -> None:
        """
        Invalidate all cached entries for a collection.
        
        Note: Only fully supported by Redis backend.
        For InMemoryCache, clears entire cache.
        """
        if isinstance(self.backend, InMemoryCache):
            # Filter and remove matching entries
            keys_to_delete = [
                k for k, v in self.backend._cache.items()
                if v.collection_name == collection_name
            ]
            for key in keys_to_delete:
                self.backend.delete(key)
            logger.info(f"Invalidated {len(keys_to_delete)} entries for {collection_name}")
        else:
            # For Redis, we'd need to scan, which is expensive
            logger.warning("Collection invalidation not fully supported for this backend")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "default_ttl": self.default_ttl,
            **self.backend.stats(),
        }


def create_cache(
    backend_type: str = "memory",
    **kwargs,
) -> QueryCache:
    """
    Factory function to create a query cache.
    
    Args:
        backend_type: "memory" or "redis"
        **kwargs: Backend-specific configuration.
    
    Returns:
        Configured QueryCache instance.
    
    Example:
        >>> cache = create_cache("memory", max_size=500)
        >>> cache = create_cache("redis", url="redis://localhost:6379")
    """
    if backend_type == "memory":
        backend = InMemoryCache(
            max_size=kwargs.get("max_size", 1000)
        )
    elif backend_type == "redis":
        backend = RedisCache(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6379),
            db=kwargs.get("db", 0),
            password=kwargs.get("password"),
            url=kwargs.get("url"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return QueryCache(
        backend=backend,
        default_ttl=kwargs.get("ttl", 3600),
        enabled=kwargs.get("enabled", True),
    )
