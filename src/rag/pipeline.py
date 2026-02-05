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
from config.settings import settings

logger = logging.getLogger(__name__)


# Default RAG prompt template
DEFAULT_RAG_PROMPT = """You are a helpful documentation assistant.
Answer the question based ONLY on the following context.
If the context doesn't contain enough information to answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


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
        
        logger.info(
            f"Initialized RAGPipeline: collection={collection_name}, "
            f"model={self.llm_model}, k={self.top_k}"
        )
    
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
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: User's question.
            include_sources: Whether to include source documents.
            api_key: Optional user-provided API key (BYOK).
            
        Returns:
            Dictionary with answer and optional sources.
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # API key is required (BYOK-only mode)
        if not api_key:
            raise ValueError("API key is required. Please provide your Gemini API key.")
        
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=self.top_k,
            include_scores=True,
        )
        
        # Step 2: Format context
        context = retrieval_result.get_context(separator="\n\n---\n\n")
        
        # Step 3: Generate answer using user's API key
        logger.info("Using user-provided API key (BYOK)")
        chain = self._build_chain_with_key(api_key)
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        logger.info(f"Generated answer: {len(answer)} characters")
        
        # Build response
        response = {
            "answer": answer,
            "question": question,
        }
        
        if include_sources:
            response["sources"] = [
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
            response["num_sources"] = retrieval_result.num_results
        
        return response
    
    def _build_chain_with_key(self, api_key: str):
        """Build a one-time chain with user-provided API key."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=self.temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
        return prompt | llm | StrOutputParser()
    
    def stream(
        self,
        question: str,
    ) -> Iterator[str]:
        """
        Stream the RAG pipeline response.
        
        Args:
            question: User's question.
            
        Yields:
            Response chunks as they are generated.
        """
        logger.info(f"Streaming query: {question[:50]}...")
        
        # Retrieve context
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=self.top_k,
            include_scores=False,
        )
        context = retrieval_result.get_context(separator="\n\n---\n\n")
        
        # Stream response
        for chunk in self.chain.stream({
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
        }


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
