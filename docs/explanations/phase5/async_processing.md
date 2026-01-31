# 🚀 Phase 5: Async & Batch Processing

## 🧠 Why Async/Batch Processing?

Processing documents and embeddings one at a time is slow. Async/batch processing enables:

| Benefit | Explanation |
|---------|-------------|
| **Parallelism** | Process multiple items simultaneously |
| **Efficiency** | Batch API calls to reduce overhead |
| **Resilience** | Retry failed operations automatically |
| **Control** | Rate limiting prevents API throttling |

---

## 📁 File: `src/rag/async_utils.py`

### Classes Overview

| Class | Purpose |
|-------|---------|
| `BatchResult` | Dataclass for batch operation results |
| `AsyncBatchProcessor` | Generic batch processor with concurrency |
| `AsyncDocumentProcessor` | Document pipeline processor |
| `RateLimiter` | Token bucket rate limiter |
| `AsyncRetry` | Retry decorator with backoff |

---

## 🔧 Class: `BatchResult`

```python
@dataclass
class BatchResult(Generic[T]):
    results: List[T]                    # All results (may include None)
    errors: List[Optional[Exception]]   # Errors for each item
    success_count: int
    error_count: int
    
    @property
    def all_succeeded(self) -> bool: ...
    
    def get_successful(self) -> List[T]: ...
```

---

## 🔧 Class: `AsyncBatchProcessor`

### Constructor
```python
def __init__(
    self,
    process_fn: Callable[[T], R],  # Sync or async function
    batch_size: int = 32,
    max_concurrency: int = 4,
    retry_count: int = 3,
    retry_delay: float = 1.0,
):
```

### Method: `process()`
```python
async def process(
    self,
    items: List[T],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[R]:
```

**Usage:**
```python
processor = AsyncBatchProcessor(
    process_fn=embed_text,
    batch_size=32,
    max_concurrency=4,
)

# Async usage
results = await processor.process(texts)

# Sync usage
results = processor.process_sync(texts)

print(f"Processed {results.success_count} items")
```

---

## 🔧 Class: `RateLimiter`

Token bucket algorithm for smooth rate limiting.

```python
limiter = RateLimiter(
    requests_per_second=10.0,
    burst_size=20,
)

async with limiter:
    await make_api_call()
```

---

## 🔧 Class: `AsyncRetry`

Decorator for automatic retries with exponential backoff.

```python
@AsyncRetry(max_retries=3, base_delay=1.0)
async def fetch_data():
    response = await api.get("/data")
    return response
```

---

## 🔧 Helper Functions

### `gather_with_concurrency()`
```python
results = await gather_with_concurrency(
    coros=[fetch(url) for url in urls],
    max_concurrency=10,
)
```

### `run_async()`
```python
# Run async code from sync context
result = run_async(some_coroutine())
```

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `BatchResult` | Structured batch output |
| `AsyncBatchProcessor` | Concurrent item processing |
| `AsyncDocumentProcessor` | Full document pipeline |
| `RateLimiter` | API rate control |
| `AsyncRetry` | Automatic retry with backoff |
| `gather_with_concurrency` | Limited parallel execution |
| `run_async` | Sync-async bridge |
