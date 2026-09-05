"""
RAG Pipeline Module

Implements the complete Retrieval-Augmented Generation pipeline
using LangChain Expression Language (LCEL).
"""

import logging
from typing import Optional, Dict, Any, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag.retrieval import Retriever, create_retriever
from src.rag.llm import LLMManager
from src.rag.caching import QueryCache, create_cache
from config.settings import settings

logger = logging.getLogger(__name__)


# Default RAG prompt template
DEFAULT_RAG_PROMPT = """You are an expert documentation assistant. Your task is to provide detailed, accurate answers based on the provided context.

Instructions:
- Answer the question using the information from the context below.
- Synthesize information from multiple context sections if needed.
- If the context contains partial information, provide what you can and note what's missing.
- Use specific details, quotes, and references from the context.
- Structure your answer clearly with bullet points or paragraphs as appropriate.

Context:
{context}

Question: {question}

Detailed Answer:"""


class RAGPipeline:
    """
    Complete RAG pipeline with retrieval and generation.
    
    Uses LangChain Expression Language (LCEL) for composable,
    streamable, and traceable pipelines.
    
    Example:
        >>> pipeline = RAGPipeline()
        >>> answer = pipeline.query("What is the refund policy?")
        >>> # Or stream the response
        >>> for chunk in pipeline.stream("What is RAG?"):
        ...     print(chunk, end="")
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = None,
        temperature: float = None,
        top_k: int = None,
        prompt_template: str = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: ChromaDB collection name.
            embedding_model: HuggingFace embedding model.
            llm_model: LLM model to use (default from settings).
            temperature: LLM temperature (default from settings).
            top_k: Number of documents to retrieve.
            prompt_template: Custom prompt template.
            cache_enabled: Whether to enable query caching.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.top_k = top_k or settings.top_k_results
        self.prompt_template = prompt_template or DEFAULT_RAG_PROMPT
        
        # Lazy initialization
        self._retriever: Optional[Retriever] = None
        self._llm_manager: Optional[LLMManager] = None
        self._chain = None
        
        # Initialize query cache
        self._cache = self._init_cache(enabled=cache_enabled, ttl=cache_ttl)
        
        logger.info(
            f"Initialized RAGPipeline: collection={collection_name}, "
            f"model={self.llm_model}, k={self.top_k}, "
            f"cache={self._cache.backend.__class__.__name__}"
        )
    
    def _init_cache(self, enabled: bool = True, ttl: int = 3600) -> QueryCache:
        """
        Initialize the query cache.
        
        Uses Redis if UPSTASH_REDIS_URL is configured, 
        otherwise falls back to in-memory cache.
        """
        redis_url = settings.upstash_redis_url or ""
        redis_configured = redis_url.startswith(("redis://", "rediss://", "unix://"))
        try:
            if redis_configured:
                logger.info("Initializing Redis cache (Upstash)")
                return create_cache(
                    backend_type="redis",
                    url=redis_url,
                    ttl=ttl,
                    enabled=enabled,
                )
            else:
                if redis_url:
                    logger.warning(
                        "UPSTASH_REDIS_URL is set but is not a redis:// URL "
                        "(got %r); using in-memory cache.", redis_url[:20] + "…"
                    )
                logger.info("Initializing in-memory cache")
                return create_cache(
                    backend_type="memory",
                    max_size=1000,
                    ttl=ttl,
                    enabled=enabled,
                )
        except Exception as e:
            logger.warning(f"Cache initialization failed, using in-memory fallback: {e}")
            return create_cache(backend_type="memory", ttl=ttl, enabled=enabled)
    
    @property
    def retriever(self) -> Retriever:
        """Get retriever instance (lazy loaded)."""
        if self._retriever is None:
            self._retriever = create_retriever(
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                default_k=self.top_k,
            )
        return self._retriever
    
    @property
    def llm_manager(self) -> LLMManager:
        """Get LLM manager instance (lazy loaded)."""
        if self._llm_manager is None:
            self._llm_manager = LLMManager(
                model_name=self.llm_model,
                temperature=self.temperature,
            )
        return self._llm_manager
    
    @property
    def chain(self):
        """
        Get the LCEL chain (lazy built).
        
        Chain structure:
        {context, question} -> prompt -> LLM -> output_parser
        """
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain
    
    def _build_chain(self):
        """Build the LCEL chain with optional tracing."""
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        # Get LLM
        llm = self.llm_manager.get_llm()
        
        # Output parser
        output_parser = StrOutputParser()
        
        # Build the chain
        chain = prompt | llm | output_parser
        
        # Add Langfuse tracing if available
        try:
            from src.llmops.langfuse_tracer import create_traced_chain, is_langfuse_enabled
            if is_langfuse_enabled():
                chain = create_traced_chain(chain, trace_name="rag-pipeline")
                logger.info("Built LCEL RAG chain with Langfuse tracing")
            else:
                logger.info("Built LCEL RAG chain (Langfuse not configured)")
        except ImportError:
            logger.info("Built LCEL RAG chain (Langfuse not available)")
        
        return chain
    
    def _format_docs(self, docs: list) -> str:
        """Format retrieved documents into context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    def query(
        self,
        question: str,
        include_sources: bool = False,
        api_key: Optional[str] = None,
        provider: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with caching and dynamic retrieval.
        
        Flow: Cache Check → Broad Retrieval → Reranking → 
              Relevance Filter → Cap → Generate → Cache Store
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # API key is required (BYOK-only mode)
        if not api_key:
            raise ValueError("API key is required. Please provide your API key.")
        
        # Step 0: Cache check — return immediately if cached
        cached_result = self._cache.get(
            query=question,
            collection_name=self.collection_name,
        )
        if cached_result:
            logger.info(f"Cache hit for query: {question[:50]}...")
            cached_result["question"] = question
            return cached_result
        
        # Step 1: Broad retrieval — fetch 20 candidates
        FETCH_K = 20
        MAX_CHUNKS = 15
        DISTANCE_THRESHOLD = 1.5  # ChromaDB distance: lower = more similar
        
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=FETCH_K,
            include_scores=True,
        )
        
        # Step 2: Dynamic filtering — keep chunks below distance threshold
        if retrieval_result.scores:
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
                if score <= DISTANCE_THRESHOLD:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            
            # Cap at MAX_CHUNKS to avoid exceeding LLM context
            filtered_docs = filtered_docs[:MAX_CHUNKS]
            filtered_scores = filtered_scores[:MAX_CHUNKS]
            
            logger.info(
                f"Dynamic retrieval: {len(retrieval_result.documents)} candidates → "
                f"{len(filtered_docs)} relevant (threshold={DISTANCE_THRESHOLD})"
            )
            
            # Update retrieval result with filtered data
            from src.rag.retrieval import RetrievalResult
            retrieval_result = RetrievalResult(
                documents=filtered_docs,
                scores=filtered_scores,
                query=question,
                metadata={"dynamic_retrieval": True, "threshold": DISTANCE_THRESHOLD},
            )
        
        # Step 3: Format context
        context = retrieval_result.get_context(separator="\n\n---\n\n")
        
        # Step 4: Generate answer using user's API key and provider
        logger.info(f"Using provider: {provider}")
        chain = self._build_chain_with_key(api_key, provider)
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        logger.info(f"Generated answer: {len(answer)} characters")
        
        # Build sources list
        sources = []
        if include_sources:
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in zip(
                    retrieval_result.documents,
                    retrieval_result.scores or []
                )
            ]
        
        # Step 5: Cache store — save result for future identical queries
        self._cache.set(
            query=question,
            collection_name=self.collection_name,
            answer=answer,
            sources=sources,
        )
        
        # Build response
        response = {
            "answer": answer,
            "question": question,
            "cached": False,
        }
        
        if include_sources:
            response["sources"] = sources
            response["num_sources"] = retrieval_result.num_results
        
        return response
    
    def _make_llm(self, api_key: str, provider: str = "gemini"):
        """Instantiate a chat model for the given provider using a BYOK key.

        Model IDs come from settings (env-overridable) because vendors
        deprecate and rename models frequently.
        """
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=self.temperature,
                google_api_key=api_key,
                convert_system_message_to_human=True,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=self.temperature,
                api_key=api_key,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=settings.anthropic_model,
                temperature=self.temperature,
                api_key=api_key,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=settings.groq_model,
                temperature=self.temperature,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _build_chain_with_key(self, api_key: str, provider: str = "gemini"):
        """Build a one-time chain with user-provided API key and provider."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = self._make_llm(api_key, provider)
        return prompt | llm | StrOutputParser()

    def stream(
        self,
        question: str,
        api_key: Optional[str] = None,
        provider: str = "gemini",
    ) -> Iterator[str]:
        """
        Stream the RAG pipeline response.

        Args:
            question: User's question.
            api_key: User-provided API key (BYOK). Required.
            provider: LLM provider (gemini, openai, anthropic, groq).

        Yields:
            Response chunks as they are generated.
        """
        logger.info(f"Streaming query: {question[:50]}...")

        if not api_key:
            raise ValueError("API key is required. Please provide your API key.")

        # Retrieve context
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=self.top_k,
            include_scores=False,
        )
        context = retrieval_result.get_context(separator="\n\n---\n\n")

        # Stream response using the caller's key/provider (not the server key)
        chain = self._build_chain_with_key(api_key, provider)
        for chunk in chain.stream({
            "context": context,
            "question": question,
        }):
            yield chunk
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "cache": self._cache.stats(),
        }
    
    def invalidate_cache(self, collection_name: Optional[str] = None) -> None:
        """Invalidate cache entries, optionally for a specific collection."""
        if collection_name:
            self._cache.invalidate_collection(collection_name)
        else:
            self._cache.backend.clear()


class RAGPipelineBuilder:
    """
    Builder pattern for constructing RAG pipelines.
    
    Example:
        >>> pipeline = (
        ...     RAGPipelineBuilder()
        ...     .with_collection("my_docs")
        ...     .with_model("gemini-2.5-pro")
        ...     .with_temperature(0.5)
        ...     .build()
        ... )
    """
    
    def __init__(self):
        """Initialize with defaults."""
        self._collection_name = "documents"
        self._embedding_model = "all-MiniLM-L6-v2"
        self._llm_model = None
        self._temperature = None
        self._top_k = None
        self._prompt_template = None
    
    def with_collection(self, name: str) -> "RAGPipelineBuilder":
        """Set the collection name."""
        self._collection_name = name
        return self
    
    def with_embedding_model(self, model: str) -> "RAGPipelineBuilder":
        """Set the embedding model."""
        self._embedding_model = model
        return self
    
    def with_model(self, model: str) -> "RAGPipelineBuilder":
        """Set the LLM model."""
        self._llm_model = model
        return self
    
    def with_temperature(self, temp: float) -> "RAGPipelineBuilder":
        """Set the temperature."""
        self._temperature = temp
        return self
    
    def with_top_k(self, k: int) -> "RAGPipelineBuilder":
        """Set the number of documents to retrieve."""
        self._top_k = k
        return self
    
    def with_prompt(self, template: str) -> "RAGPipelineBuilder":
        """Set a custom prompt template."""
        self._prompt_template = template
        return self
    
    def build(self) -> RAGPipeline:
        """Build the RAG pipeline."""
        return RAGPipeline(
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            llm_model=self._llm_model,
            temperature=self._temperature,
            top_k=self._top_k,
            prompt_template=self._prompt_template,
        )


# Convenience function
def create_rag_pipeline(
    collection_name: str = "documents",
    llm_model: str = None,
    top_k: int = None,
) -> RAGPipeline:
    """
    Create a RAG pipeline with default configuration.
    
    Args:
        collection_name: ChromaDB collection name.
        llm_model: LLM model to use.
        top_k: Number of documents to retrieve.
        
    Returns:
        Configured RAGPipeline instance.
    """
    return RAGPipeline(
        collection_name=collection_name,
        llm_model=llm_model,
        top_k=top_k,
    )
