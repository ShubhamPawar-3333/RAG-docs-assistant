"""
Unit tests for the async_utils module.
"""

import asyncio
import pytest
import importlib.util
import os

# Load module directly to avoid triggering full package init
spec = importlib.util.spec_from_file_location(
    "async_utils",
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "rag", "async_utils.py")
)
async_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(async_utils)

BatchResult = async_utils.BatchResult
AsyncBatchProcessor = async_utils.AsyncBatchProcessor
RateLimiter = async_utils.RateLimiter
AsyncRetry = async_utils.AsyncRetry
gather_with_concurrency = async_utils.gather_with_concurrency


class TestBatchResult:
    """Tests for BatchResult dataclass."""
    
    def test_batch_result_creation(self):
        """Test creating a batch result."""
        result = BatchResult(
            results=[1, 2, 3],
            errors=[None, None, None],
            success_count=3,
            error_count=0,
        )
        
        assert len(result.results) == 3
        assert result.all_succeeded
    
    def test_batch_result_with_errors(self):
        """Test batch result with errors."""
        result = BatchResult(
            results=[1, None, 3],
            errors=[None, ValueError("failed"), None],
            success_count=2,
            error_count=1,
        )
        
        assert not result.all_succeeded
        assert result.success_count == 2
        assert result.error_count == 1
    
    def test_get_successful(self):
        """Test getting only successful results."""
        result = BatchResult(
            results=["a", None, "c"],
            errors=[None, ValueError("failed"), None],
            success_count=2,
            error_count=1,
        )
        
        successful = result.get_successful()
        assert successful == ["a", "c"]


class TestAsyncBatchProcessor:
    """Tests for AsyncBatchProcessor."""
    
    @pytest.mark.asyncio
    async def test_process_empty_items(self):
        """Test processing empty list."""
        processor = AsyncBatchProcessor(
            process_fn=lambda x: x * 2,
            batch_size=10,
        )
        
        result = await processor.process([])
        
        assert result.results == []
        assert result.success_count == 0
    
    @pytest.mark.asyncio
    async def test_process_sync_function(self):
        """Test processing with sync function."""
        def double(x):
            return x * 2
        
        processor = AsyncBatchProcessor(
            process_fn=double,
            batch_size=3,
            max_concurrency=2,
        )
        
        result = await processor.process([1, 2, 3, 4, 5])
        
        assert result.success_count == 5
        assert result.results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_process_async_function(self):
        """Test processing with async function."""
        async def async_double(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        processor = AsyncBatchProcessor(
            process_fn=async_double,
            batch_size=3,
        )
        
        result = await processor.process([1, 2, 3])
        
        assert result.success_count == 3
        assert result.results == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_process_with_errors(self):
        """Test processing with some errors."""
        def maybe_fail(x):
            if x == 3:
                raise ValueError("Failed on 3")
            return x * 2
        
        processor = AsyncBatchProcessor(
            process_fn=maybe_fail,
            batch_size=10,
            retry_count=1,
        )
        
        result = await processor.process([1, 2, 3, 4])
        
        assert result.success_count == 3
        assert result.error_count == 1
        assert result.results[0] == 2
        assert result.results[2] is None  # Failed item
    
    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback is called."""
        progress_calls = []
        
        def track_progress(completed, total):
            progress_calls.append((completed, total))
        
        processor = AsyncBatchProcessor(
            process_fn=lambda x: x,
            batch_size=2,
        )
        
        await processor.process([1, 2, 3, 4], progress_callback=track_progress)
        
        assert len(progress_calls) > 0
        assert progress_calls[-1] == (4, 4)
    
    def test_process_sync_wrapper(self):
        """Test synchronous wrapper."""
        processor = AsyncBatchProcessor(
            process_fn=lambda x: x * 2,
            batch_size=2,
        )
        
        result = processor.process_sync([1, 2, 3])
        
        assert result.success_count == 3
        assert result.results == [2, 4, 6]


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Test that burst size is initially available."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        
        # Should allow 5 requests immediately
        for _ in range(5):
            await limiter.acquire()
    
    @pytest.mark.asyncio
    async def test_rate_limiter_context_manager(self):
        """Test async context manager usage."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        
        async with limiter:
            pass  # Request allowed
        
        # Should work multiple times
        async with limiter:
            pass


class TestAsyncRetry:
    """Tests for AsyncRetry decorator."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Test function that succeeds first time."""
        call_count = 0
        
        @AsyncRetry(max_retries=3)
        async def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await always_succeeds()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test function that succeeds after retry."""
        call_count = 0
        
        @AsyncRetry(max_retries=3, base_delay=0.01)
        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = await fails_twice()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test function that always fails."""
        @AsyncRetry(max_retries=2, base_delay=0.01)
        async def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            await always_fails()


class TestGatherWithConcurrency:
    """Tests for gather_with_concurrency."""
    
    @pytest.mark.asyncio
    async def test_gather_results_in_order(self):
        """Test that results maintain order."""
        async def delayed_return(x, delay):
            await asyncio.sleep(delay)
            return x
        
        coros = [
            delayed_return(1, 0.02),
            delayed_return(2, 0.01),
            delayed_return(3, 0.03),
        ]
        
        results = await gather_with_concurrency(coros, max_concurrency=3)
        
        assert results == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_gather_respects_concurrency(self):
        """Test that concurrency limit is respected."""
        max_concurrent = 0
        current_concurrent = 0
        
        async def track_concurrency(x):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return x
        
        coros = [track_concurrency(i) for i in range(10)]
        
        await gather_with_concurrency(coros, max_concurrency=3)
        
        assert max_concurrent <= 3
