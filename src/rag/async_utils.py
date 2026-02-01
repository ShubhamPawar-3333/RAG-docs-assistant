"""
Async Utilities Module

Provides async/batch processing utilities for improved
performance in embedding generation, document processing,
and API requests.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch operation."""
    results: List[T]
    errors: List[Optional[Exception]]
    success_count: int
    error_count: int
    
    @property
    def all_succeeded(self) -> bool:
        """Check if all items succeeded."""
        return self.error_count == 0
    
    def get_successful(self) -> List[T]:
        """Get only successful results."""
        return [r for r, e in zip(self.results, self.errors) if e is None]


class AsyncBatchProcessor(Generic[T, R]):
    """
    Process items in batches with concurrency control.
    
    Supports both sync and async processing functions,
    with configurable batch size and max concurrency.
    
    Example:
        >>> processor = AsyncBatchProcessor(
        ...     process_fn=embed_text,
        ...     batch_size=32,
        ...     max_concurrency=4
        ... )
        >>> results = await processor.process(texts)
    """
    
    def __init__(
        self,
        process_fn: Callable[[T], R],
        batch_size: int = 32,
        max_concurrency: int = 4,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the batch processor.
        
        Args:
            process_fn: Function to process each item (sync or async).
            batch_size: Number of items per batch.
            max_concurrency: Maximum concurrent batches.
            retry_count: Number of retries on failure.
            retry_delay: Delay between retries (seconds).
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Split items into batches."""
        return [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
    
    async def _process_with_retry(
        self,
        item: T,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> tuple[Optional[R], Optional[Exception]]:
        """Process a single item with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                if asyncio.iscoroutinefunction(self.process_fn):
                    result = await self.process_fn(item)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        executor,
                        self.process_fn,
                        item,
                    )
                return result, None
            except Exception as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    logger.warning(f"Retry {attempt + 1} for item: {e}")
        
        return None, last_error
    
    async def _process_batch(
        self,
        batch: List[T],
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> List[tuple[Optional[R], Optional[Exception]]]:
        """Process a single batch with concurrency control."""
        async with self._semaphore:
            tasks = [
                self._process_with_retry(item, executor)
                for item in batch
            ]
            return await asyncio.gather(*tasks)
    
    async def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """
        Process all items in batches.
        
        Args:
            items: Items to process.
            progress_callback: Optional callback(completed, total).
        
        Returns:
            BatchResult with all results and errors.
        """
        if not items:
            return BatchResult(
                results=[],
                errors=[],
                success_count=0,
                error_count=0,
            )
        
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        batches = self._create_batches(items)
        
        all_results: List[Optional[R]] = []
        all_errors: List[Optional[Exception]] = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for batch in batches:
                batch_results = await self._process_batch(batch, executor)
                
                for result, error in batch_results:
                    all_results.append(result)
                    all_errors.append(error)
                
                completed += len(batch)
                if progress_callback:
                    progress_callback(completed, len(items))
        
        success_count = sum(1 for e in all_errors if e is None)
        error_count = len(all_errors) - success_count
        
        logger.info(
            f"Batch processing complete: {success_count} succeeded, "
            f"{error_count} failed"
        )
        
        return BatchResult(
            results=all_results,
            errors=all_errors,
            success_count=success_count,
            error_count=error_count,
        )
    
    def process_sync(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Synchronous wrapper for process()."""
        return asyncio.run(self.process(items, progress_callback))


class AsyncDocumentProcessor:
    """
    Async document processing pipeline.
    
    Handles concurrent document loading, chunking, and embedding.
    """
    
    def __init__(
        self,
        loader_fn: Callable,
        chunker_fn: Callable,
        embedder_fn: Callable,
        batch_size: int = 32,
        max_concurrency: int = 4,
    ):
        self.loader_fn = loader_fn
        self.chunker_fn = chunker_fn
        self.embedder_fn = embedder_fn
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def process_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple files concurrently.
        
        Args:
            file_paths: Paths to files to process.
            progress_callback: Optional callback(stage, completed, total).
        
        Returns:
            Dict with processed documents, chunks, and statistics.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def load_file(path: str) -> tuple[str, Any, Optional[Exception]]:
            async with semaphore:
                try:
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(
                        None, self.loader_fn, path
                    )
                    return path, docs, None
                except Exception as e:
                    return path, None, e
        
        # Load all files concurrently
        if progress_callback:
            progress_callback("loading", 0, len(file_paths))
        
        load_tasks = [load_file(path) for path in file_paths]
        load_results = await asyncio.gather(*load_tasks)
        
        all_documents = []
        errors = []
        
        for path, docs, error in load_results:
            if error:
                errors.append({"file": path, "error": str(error)})
                logger.error(f"Failed to load {path}: {error}")
            elif docs:
                all_documents.extend(docs)
        
        if progress_callback:
            progress_callback("loading", len(file_paths), len(file_paths))
        
        # Chunk documents
        if progress_callback:
            progress_callback("chunking", 0, len(all_documents))
        
        loop = asyncio.get_event_loop()
        all_chunks = await loop.run_in_executor(
            None, self.chunker_fn, all_documents
        )
        
        if progress_callback:
            progress_callback("chunking", len(all_documents), len(all_documents))
        
        # Embed chunks in batches
        if progress_callback:
            progress_callback("embedding", 0, len(all_chunks))
        
        chunk_texts = [chunk.page_content for chunk in all_chunks]
        
        # Process embeddings in batches
        processor = AsyncBatchProcessor(
            process_fn=self.embedder_fn,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
        )
        
        embed_result = await processor.process(
            chunk_texts,
            lambda c, t: progress_callback("embedding", c, t) if progress_callback else None,
        )
        
        return {
            "documents": all_documents,
            "chunks": all_chunks,
            "embeddings": embed_result.get_successful(),
            "errors": errors,
            "stats": {
                "files_processed": len(file_paths) - len(errors),
                "files_failed": len(errors),
                "documents": len(all_documents),
                "chunks": len(all_chunks),
                "embeddings": embed_result.success_count,
            },
        }


async def gather_with_concurrency(
    coros: List[Coroutine],
    max_concurrency: int = 10,
) -> List[Any]:
    """
    Run coroutines with limited concurrency.
    
    Args:
        coros: List of coroutines to run.
        max_concurrency: Maximum concurrent executions.
    
    Returns:
        List of results in order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bounded(c) for c in coros])


def run_async(coro: Coroutine) -> Any:
    """
    Run an async coroutine from sync code.
    
    Handles event loop creation/reuse properly.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, can't use asyncio.run
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)


class RateLimiter:
    """
    Async rate limiter for API calls.
    
    Uses token bucket algorithm for smooth rate limiting.
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum sustained rate.
            burst_size: Maximum burst size.
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            
            if self.tokens < 1:
                # Wait for a token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, *args):
        pass


class AsyncRetry:
    """
    Async retry decorator with exponential backoff.
    
    Usage:
        @AsyncRetry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions
    
    def __call__(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_error = e
                    if attempt < self.max_retries:
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt),
                            self.max_delay
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{self.max_retries} "
                            f"for {func.__name__}: {e}. "
                            f"Waiting {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
            
            raise last_error
        
        return wrapper
