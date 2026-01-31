# 🚀 Phase 5: Query Caching

## 🧠 What is Query Caching?

Query caching stores the results of RAG queries so that **identical questions don't require re-computation**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY CACHING FLOW                            │
│                                                                  │
│  Query: "What is the refund policy?"                            │
│       ↓                                                          │
│  [Check Cache]                                                   │
│       ↓                                                          │
│  Cache Hit? ──Yes──→ Return cached answer (fast!)               │
│       │                                                          │
│      No                                                          │
│       ↓                                                          │
│  [Run RAG Pipeline]                                              │
│       ↓                                                          │
│  [Store in Cache]                                                │
│       ↓                                                          │
│  Return answer                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Why Cache RAG Queries?

| Benefit | Explanation |
|---------|-------------|
| **Speed** | Cached responses return in <10ms vs 2-5 seconds |
| **Cost Reduction** | Avoids repeated LLM API calls |
| **Consistency** | Same query always returns same answer |
| **Reduced Load** | Less stress on vector store and LLM |

---

## 📁 File: `src/rag/caching.py`

### Classes Overview

| Class | Purpose | Use Case |
|-------|---------|----------|
| `CacheEntry` | Dataclass for cached data | Internal storage |
| `CacheBackend` | Abstract base class | Interface for backends |
| `InMemoryCache` | Dictionary-based cache | Development, single instance |
| `RedisCache` | Redis-based cache | Production, multi-instance |
| `QueryCache` | Main caching class | User-facing API |

---

## 🔧 Class: `CacheEntry`

```python
@dataclass
class CacheEntry:
    query: str                    # Original query
    answer: str                   # Generated answer
    sources: List[Dict[str, Any]] # Source documents
    collection_name: str          # Collection queried
    created_at: float             # Timestamp
    ttl: int                      # Time to live (seconds)
    hit_count: int = 0            # Number of cache hits
```

### Property: `is_expired`
```python
@property
def is_expired(self) -> bool:
    return time.time() > (self.created_at + self.ttl)
```
Automatically checks if the entry has exceeded its TTL.

---

## 🔧 Class: `CacheBackend` (Abstract)

```python
class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]: ...
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None: ...
    
    @abstractmethod
    def delete(self, key: str) -> bool: ...
    
    @abstractmethod
    def clear(self) -> None: ...
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...
```

**Purpose:** Defines the interface that all cache backends must implement.

---

## 🔧 Class: `InMemoryCache`

### Constructor
```python
def __init__(self, max_size: int = 1000):
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | 1000 | Maximum entries before eviction |

### How It Works

```python
class InMemoryCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}  # Storage
        self._max_size = max_size
        self._hits = 0    # Track hits
        self._misses = 0  # Track misses
```

### Method: `get()`
```python
def get(self, key: str) -> Optional[CacheEntry]:
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
```

### Method: `set()` - With LRU Eviction
```python
def set(self, key: str, entry: CacheEntry) -> None:
    if len(self._cache) >= self._max_size:
        # Evict oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_key]
    
    self._cache[key] = entry
```

---

## 🔧 Class: `RedisCache`

### Constructor
```python
def __init__(
    self,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    prefix: str = "rag:cache:",
    url: Optional[str] = None,  # Redis URL overrides host/port
):
```

### Key Features
- **Lazy connection:** Connects to Redis only when first used
- **TTL support:** Uses Redis's native SETEX for expiration
- **Key prefix:** All keys start with `rag:cache:` for isolation

### Method: `set()` - With Native TTL
```python
def set(self, key: str, entry: CacheEntry) -> None:
    self.client.setex(
        self._make_key(key),
        entry.ttl,  # Redis handles expiration
        json.dumps(entry.to_dict()),
    )
```

---

## 🔧 Class: `QueryCache`

The main user-facing class that wraps a backend.

### Constructor
```python
def __init__(
    self,
    backend: Optional[CacheBackend] = None,  # Defaults to InMemoryCache
    default_ttl: int = 3600,  # 1 hour
    enabled: bool = True,
):
```

### Method: `_generate_key()` - Hash-Based Keys
```python
def _generate_key(
    self,
    query: str,
    collection_name: str,
    **kwargs,
) -> str:
    key_data = {
        "query": query.lower().strip(),
        "collection": collection_name,
        **kwargs,
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

**Why hash?** 
- Normalizes query text (lowercase, trimmed)
- Creates fixed-length keys
- Handles special characters

### Method: `get_or_compute()` - Main API
```python
def get_or_compute(
    self,
    query: str,
    collection_name: str,
    compute_fn: callable,
    ttl: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    # Try cache first
    cached = self.get(query, collection_name, **kwargs)
    if cached:
        return cached  # Fast path!
    
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
```

**Usage:**
```python
cache = QueryCache()

result = cache.get_or_compute(
    query="What is RAG?",
    collection_name="docs",
    compute_fn=lambda: pipeline.query("What is RAG?"),
    ttl=3600,  # 1 hour
)

print(result["cached"])  # True on second call
```

---

## 🔧 Factory Function: `create_cache()`

```python
def create_cache(
    backend_type: str = "memory",  # "memory" or "redis"
    **kwargs,
) -> QueryCache:
```

**Examples:**
```python
# In-memory (development)
cache = create_cache("memory", max_size=500)

# Redis (production)
cache = create_cache(
    "redis",
    url="redis://localhost:6379",
    ttl=7200,  # 2 hours
)
```

---

## 📊 Cache Statistics

```python
stats = cache.stats()
# {
#     "enabled": True,
#     "default_ttl": 3600,
#     "backend": "in-memory",
#     "entries": 42,
#     "max_size": 1000,
#     "hits": 156,
#     "misses": 23,
#     "hit_rate": "87.2%"
# }
```

---

## 🔗 Integration Example

```python
from src.rag import create_rag_pipeline
from src.rag.caching import create_cache

# Create cache
cache = create_cache("memory", ttl=3600)

# Create pipeline
pipeline = create_rag_pipeline(collection_name="docs")

# Query with caching
def query_with_cache(question: str) -> dict:
    return cache.get_or_compute(
        query=question,
        collection_name="docs",
        compute_fn=lambda: pipeline.query(question),
    )

# First call: ~2s (pipeline runs)
result1 = query_with_cache("What is RAG?")

# Second call: <10ms (from cache)
result2 = query_with_cache("What is RAG?")
```

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `CacheEntry` | Stores cached query + answer |
| `CacheBackend` | Abstract interface |
| `InMemoryCache` | Fast, single-instance |
| `RedisCache` | Distributed, persistent |
| `QueryCache` | Main API, wraps backend |
| `get_or_compute()` | Cache-through pattern |
| `create_cache()` | Factory for easy setup |
