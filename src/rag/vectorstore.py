"""
Vector Store Module

Manages the ChromaDB vector database for document storage and retrieval.
Provides a unified interface for:
- Storing document embeddings
- Similarity search
- Collection management
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


class VectorStore:
    """
    ChromaDB-based vector store for document embeddings.
    
    Provides persistent storage and similarity search capabilities
    for RAG applications.
    
    Example:
        >>> from src.rag.embeddings import get_embeddings
        >>> embeddings = get_embeddings()
        >>> store = VectorStore(embeddings, collection_name="my_docs")
        >>> store.add_documents(chunks)
        >>> results = store.similarity_search("What is RAG?", k=5)
    """
    
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_function: LangChain embeddings instance.
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
        """
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        
        # Set persist directory from settings or parameter
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = settings.chroma_dir
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self._store = self._create_store()
        
        logger.info(
            f"Initialized VectorStore: collection='{collection_name}', "
            f"persist_dir='{self.persist_directory}'"
        )
    
    def _create_store(self) -> Chroma:
        """Create or load the ChromaDB store."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_directory),
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            batch_size: Number of documents to add per batch.
            
        Returns:
            List of document IDs.
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        all_ids: List[str] = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                ids = self._store.add_documents(batch)
                all_ids.extend(ids)
                logger.debug(f"Added batch {i // batch_size + 1}: {len(batch)} documents")
            except Exception as e:
                logger.error(f"Error adding batch {i // batch_size + 1}: {e}")
                raise VectorStoreError(f"Failed to add documents: {e}") from e
        
        logger.info(f"Added {len(all_ids)} documents to collection '{self.collection_name}'")
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text.
            k: Number of results to return (default from settings).
            filter: Optional metadata filter.
            
        Returns:
            List of similar documents.
        """
        k = k or settings.top_k_results
        
        try:
            results = self._store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.debug(f"Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            
        Returns:
            List of (Document, score) tuples.
        """
        k = k or settings.top_k_results
        
        try:
            results = self._store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.debug(
                f"Found {len(results)} results with scores for query: '{query[:50]}...'"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self._store.delete_collection()
            logger.info(f"Deleted collection '{self.collection_name}'")
            
            # Recreate empty store
            self._store = self._create_store()
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        try:
            # Access the underlying ChromaDB collection
            collection = self._store._collection
            
            return {
                "collection_name": self.collection_name,
                "document_count": collection.count(),
                "persist_directory": str(self.persist_directory),
            }
            
        except Exception as e:
            logger.warning(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": "unknown",
                "persist_directory": str(self.persist_directory),
            }
    
    @property
    def retriever(self):
        """
        Get a LangChain retriever from this store.
        
        Returns:
            VectorStoreRetriever instance.
        """
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.top_k_results}
        )
    
    def as_retriever(self, **kwargs):
        """
        Get a customized retriever.
        
        Args:
            **kwargs: Arguments passed to as_retriever().
            
        Returns:
            VectorStoreRetriever instance.
        """
        return self._store.as_retriever(**kwargs)


# Factory function for creating vector stores
def create_vector_store(
    embedding_function: Embeddings,
    collection_name: str = "documents",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        embedding_function: LangChain embeddings instance.
        collection_name: Name of the collection.
        persist_directory: Directory to persist data.
        
    Returns:
        Configured VectorStore instance.
    """
    return VectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
