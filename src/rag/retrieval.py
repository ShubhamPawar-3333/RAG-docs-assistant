"""
Retrieval Module

Handles document retrieval from the vector store with various
search strategies and result processing.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document

from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import VectorStore, create_vector_store
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Structured result from a retrieval query.
    
    Attributes:
        documents: List of retrieved documents.
        scores: Optional relevance scores for each document.
        query: The original query string.
        metadata: Additional retrieval metadata.
    """
    documents: List[Document]
    scores: Optional[List[float]] = None
    query: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def num_results(self) -> int:
        """Number of retrieved documents."""
        return len(self.documents)
    
    def get_context(self, separator: str = "\n\n") -> str:
        """
        Combine all documents into a single context string.
        
        Args:
            separator: String to separate documents.
            
        Returns:
            Combined context string.
        """
        return separator.join(doc.page_content for doc in self.documents)
    
    def filter_by_score(self, min_score: float) -> "RetrievalResult":
        """
        Filter results by minimum relevance score.
        
        Args:
            min_score: Minimum score threshold.
            
        Returns:
            New RetrievalResult with filtered documents.
        """
        if self.scores is None:
            return self
        
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(self.documents, self.scores):
            if score >= min_score:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return RetrievalResult(
            documents=filtered_docs,
            scores=filtered_scores,
            query=self.query,
            metadata=self.metadata,
        )


class Retriever:
    """
    Document retriever with configurable search strategies.
    
    Provides a high-level interface for retrieving relevant documents
    from the vector store based on semantic similarity.
    
    Example:
        >>> retriever = Retriever(vector_store)
        >>> results = retriever.retrieve("What is the refund policy?")
        >>> context = results.get_context()
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        default_k: int = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance to search.
            default_k: Default number of results to return.
            score_threshold: Optional minimum score for results.
        """
        self.vector_store = vector_store
        self.default_k = default_k or settings.top_k_results
        self.score_threshold = score_threshold
        
        logger.info(
            f"Initialized Retriever with k={self.default_k}, "
            f"threshold={self.score_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text.
            k: Number of results (default from settings).
            filter: Optional metadata filter.
            include_scores: Whether to include relevance scores.
            
        Returns:
            RetrievalResult with documents and metadata.
        """
        k = k or self.default_k
        
        logger.debug(f"Retrieving for query: '{query[:50]}...' k={k}")
        
        if include_scores:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )
            
            documents = [doc for doc, _ in results_with_scores]
            scores = [score for _, score in results_with_scores]
        else:
            documents = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
            scores = None
        
        result = RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            metadata={
                "k": k,
                "filter": filter,
                "include_scores": include_scores,
            },
        )
        
        # Apply score threshold if configured
        if self.score_threshold and scores:
            result = result.filter_by_score(self.score_threshold)
        
        logger.info(f"Retrieved {result.num_results} documents for query")
        return result
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Retrieve and return combined context string.
        
        Convenience method for RAG pipelines that just need
        the context string.
        
        Args:
            query: Search query text.
            k: Number of results.
            separator: Document separator.
            
        Returns:
            Combined context string from retrieved documents.
        """
        result = self.retrieve(query, k=k, include_scores=False)
        return result.get_context(separator=separator)
    
    def get_langchain_retriever(self, **kwargs):
        """
        Get a LangChain-compatible retriever.
        
        Args:
            **kwargs: Arguments passed to as_retriever().
            
        Returns:
            LangChain VectorStoreRetriever.
        """
        if not kwargs:
            kwargs = {
                "search_type": "similarity",
                "search_kwargs": {"k": self.default_k},
            }
        return self.vector_store.as_retriever(**kwargs)


class RetrieverFactory:
    """
    Factory for creating configured retrievers.
    
    Handles the setup of embeddings and vector store.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the factory.
        
        Args:
            collection_name: ChromaDB collection name.
            embedding_model: HuggingFace embedding model.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._vector_store: Optional[VectorStore] = None
    
    def get_vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            embeddings = get_embeddings(model_name=self.embedding_model)
            self._vector_store = create_vector_store(
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )
        return self._vector_store
    
    def create_retriever(
        self,
        default_k: int = None,
        score_threshold: float = None,
    ) -> Retriever:
        """
        Create a configured retriever.
        
        Args:
            default_k: Default number of results.
            score_threshold: Minimum score threshold.
            
        Returns:
            Configured Retriever instance.
        """
        vector_store = self.get_vector_store()
        return Retriever(
            vector_store=vector_store,
            default_k=default_k,
            score_threshold=score_threshold,
        )


# Convenience function
def create_retriever(
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    default_k: int = None,
    score_threshold: float = None,
) -> Retriever:
    """
    Create a retriever with default configuration.
    
    Args:
        collection_name: ChromaDB collection name.
        embedding_model: HuggingFace embedding model.
        default_k: Default number of results.
        score_threshold: Minimum score threshold.
        
    Returns:
        Configured Retriever instance.
    """
    factory = RetrieverFactory(
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return factory.create_retriever(
        default_k=default_k,
        score_threshold=score_threshold,
    )
