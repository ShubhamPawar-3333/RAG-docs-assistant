"""
Langfuse Observability Module

Provides LLM tracing and observability using Langfuse.
Tracks token usage, latency, costs, and full request/response chains.
"""

import logging
from typing import Optional, Any, Dict
from functools import wraps
import time

from config.settings import settings

logger = logging.getLogger(__name__)

# Langfuse client singleton
_langfuse_client = None
_langfuse_enabled = False


def get_langfuse():
    """
    Get or initialize the Langfuse client.
    
    Returns None if Langfuse is not configured.
    """
    global _langfuse_client, _langfuse_enabled
    
    if _langfuse_client is not None:
        return _langfuse_client
    
    # Check if Langfuse is configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("Langfuse not configured - tracing disabled")
        _langfuse_enabled = False
        return None
    
    try:
        from langfuse import Langfuse
        
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _langfuse_enabled = True
        logger.info(f"Langfuse initialized: {settings.langfuse_host}")
        return _langfuse_client
    except ImportError:
        logger.warning("Langfuse package not installed")
        _langfuse_enabled = False
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        _langfuse_enabled = False
        return None


def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    global _langfuse_enabled
    get_langfuse()  # Ensure initialization attempted
    return _langfuse_enabled


class LangfuseTracer:
    """
    Langfuse tracer for RAG pipeline.
    
    Provides context managers and decorators for tracing
    different stages of the RAG pipeline.
    """
    
    def __init__(self, trace_name: str = "rag-query"):
        """
        Initialize tracer with a trace name.
        
        Args:
            trace_name: Name for the trace
        """
        self.trace_name = trace_name
        self.client = get_langfuse()
        self.trace = None
    
    def start_trace(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        """
        Start a new trace.
        
        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata to attach
            tags: Tags for filtering
        """
        if not self.client:
            return self
        
        try:
            self.trace = self.client.trace(
                name=self.trace_name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or [],
            )
        except Exception as e:
            logger.error(f"Failed to start trace: {e}")
        
        return self
    
    def span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span within the current trace.
        
        Args:
            name: Span name (e.g., "retrieval", "generation")
            input_data: Input data for the span
            metadata: Additional metadata
        
        Returns:
            Span object or None if tracing disabled
        """
        if not self.trace:
            return DummySpan()
        
        try:
            return self.trace.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return DummySpan()
    
    def generation(
        self,
        name: str,
        model: str,
        input_data: Any,
        output_data: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an LLM generation.
        
        Args:
            name: Generation name
            model: Model name
            input_data: Prompt/input
            output_data: Response/output
            usage: Token usage dict
            metadata: Additional metadata
        """
        if not self.trace:
            return DummyGeneration()
        
        try:
            return self.trace.generation(
                name=name,
                model=model,
                input=input_data,
                output=output_data,
                usage=usage,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")
            return DummyGeneration()
    
    def end_trace(
        self,
        output: Optional[Any] = None,
        status: str = "success",
    ):
        """
        End the current trace.
        
        Args:
            output: Final output of the trace
            status: Trace status
        """
        if not self.trace:
            return
        
        try:
            self.trace.update(
                output=output,
                status_message=status,
            )
        except Exception as e:
            logger.error(f"Failed to end trace: {e}")
    
    def flush(self):
        """Flush pending traces to Langfuse."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


class DummySpan:
    """Dummy span for when Langfuse is disabled."""
    
    def update(self, **kwargs):
        pass
    
    def end(self, **kwargs):
        pass


class DummyGeneration:
    """Dummy generation for when Langfuse is disabled."""
    
    def update(self, **kwargs):
        pass
    
    def end(self, **kwargs):
        pass


def trace_rag_query(func):
    """
    Decorator to trace RAG query functions.
    
    Automatically creates a trace, measures timing,
    and logs the query/response.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = LangfuseTracer(trace_name="rag-query")
        tracer.start_trace(
            metadata={"function": func.__name__},
            tags=["rag", "query"],
        )
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs, _tracer=tracer)
            
            duration = time.time() - start_time
            tracer.end_trace(
                output={"duration_seconds": duration},
                status="success",
            )
            
            return result
        except Exception as e:
            tracer.end_trace(
                output={"error": str(e)},
                status="error",
            )
            raise
        finally:
            tracer.flush()
    
    return wrapper


# ============== LangChain Integration ==============

def get_langfuse_callback_handler():
    """
    Get a LangChain callback handler for Langfuse.
    
    Returns None if Langfuse is not configured.
    """
    if not is_langfuse_enabled():
        return None
    
    try:
        from langfuse.callback import CallbackHandler
        
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError:
        logger.warning("Langfuse callback handler not available")
        return None
    except Exception as e:
        logger.error(f"Failed to create callback handler: {e}")
        return None


def create_traced_chain(chain, trace_name: str = "rag-chain"):
    """
    Wrap a LangChain chain with Langfuse tracing.
    
    Args:
        chain: LangChain chain to wrap
        trace_name: Name for traces
    
    Returns:
        Wrapped chain with tracing
    """
    handler = get_langfuse_callback_handler()
    
    if not handler:
        return chain
    
    # Return chain with callback configured
    return chain.with_config(callbacks=[handler])
