"""
Embeddings Module

Provides text embedding functionality using HuggingFace models.
Embeddings convert text into dense vector representations for
semantic similarity search.
"""

import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Default embedding model - excellent balance of quality and speed
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model configurations with their properties
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_seq_length": 256,
        "description": "Fast, good quality, small size (80MB)",
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "max_seq_length": 384,
        "description": "Higher quality, larger size (420MB)",
    },
    "bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "max_seq_length": 512,
        "description": "State-of-the-art for retrieval tasks",
    },
    "e5-small-v2": {
        "model_name": "intfloat/e5-small-v2",
        "dimensions": 384,
        "max_seq_length": 512,
        "description": "Microsoft's efficient embedding model",
    },
}


class EmbeddingsManager:
    """
    Manages text embedding models for the RAG pipeline.
    
    Provides a unified interface for creating embeddings with
    configurable models and caching options.
    
    Example:
        >>> manager = EmbeddingsManager()
        >>> embeddings = manager.get_embeddings()
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: HuggingFace model name or alias.
            device: Device to run model on ("cpu" or "cuda").
            normalize_embeddings: Whether to L2 normalize vectors.
            cache_folder: Folder to cache downloaded models.
        """
        # Resolve alias to full model name
        if model_name in EMBEDDING_MODELS:
            self.model_config = EMBEDDING_MODELS[model_name]
            self.model_name = self.model_config["model_name"]
        else:
            self.model_name = model_name
            self.model_config = None
        
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder
        
        # Lazy initialization
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        
        logger.info(f"Initialized EmbeddingsManager with model: {self.model_name}")
    
    def get_embeddings(self) -> Embeddings:
        """
        Get the embeddings instance (lazy loaded).
        
        Returns:
            HuggingFace embeddings instance.
        """
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """Create the HuggingFace embeddings instance."""
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": self.normalize_embeddings}
        
        kwargs = {
            "model_name": self.model_name,
            "model_kwargs": model_kwargs,
            "encode_kwargs": encode_kwargs,
        }
        
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        embeddings = HuggingFaceEmbeddings(**kwargs)
        
        logger.info("Embedding model loaded successfully")
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model details.
        """
        if self.model_config:
            return {
                "model_name": self.model_name,
                "dimensions": self.model_config["dimensions"],
                "max_seq_length": self.model_config["max_seq_length"],
                "description": self.model_config["description"],
                "device": self.device,
            }
        else:
            return {
                "model_name": self.model_name,
                "device": self.device,
            }


# Singleton instance for convenience
_default_manager: Optional[EmbeddingsManager] = None


def get_embeddings(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    force_new: bool = False
) -> Embeddings:
    """
    Get embeddings instance (convenience function).
    
    Uses a singleton pattern for efficiency - the model is only
    loaded once per process.
    
    Args:
        model_name: Model to use (alias or full HuggingFace name).
        device: Device to run on ("cpu" or "cuda").
        force_new: Force creation of new instance.
        
    Returns:
        Embeddings instance ready for use.
    """
    global _default_manager
    
    if force_new or _default_manager is None:
        _default_manager = EmbeddingsManager(
            model_name=model_name,
            device=device
        )
    
    return _default_manager.get_embeddings()


def list_available_models() -> dict:
    """
    List all pre-configured embedding models.
    
    Returns:
        Dictionary of model aliases and their configurations.
    """
    return EMBEDDING_MODELS.copy()
