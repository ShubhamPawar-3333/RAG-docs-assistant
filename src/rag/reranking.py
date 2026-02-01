"""
Cross-Encoder Reranking Module

Improves retrieval quality by reranking initial results
using a more powerful cross-encoder model.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    documents: List[Document]
    scores: List[float]
    original_scores: List[float]


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders for similarity
    scoring because they process query and document together.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 32,
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            top_k: Number of documents to return after reranking
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed for reranking")
                raise ImportError(
                    "Please install sentence-transformers: "
                    "pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None,
    ) -> RerankResult:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            original_scores: Original retrieval scores (optional)
        
        Returns:
            RerankResult with reranked documents and scores
        """
        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                original_scores=[],
            )
        
        if original_scores is None:
            original_scores = [0.0] * len(documents)
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fall back to original order
            return RerankResult(
                documents=documents[:self.top_k],
                scores=original_scores[:self.top_k],
                original_scores=original_scores[:self.top_k],
            )
        
        # Sort by score (descending)
        doc_score_pairs = list(zip(documents, scores, original_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_docs = doc_score_pairs[:self.top_k]
        
        return RerankResult(
            documents=[d[0] for d in top_docs],
            scores=[d[1] for d in top_docs],
            original_scores=[d[2] for d in top_docs],
        )


class LLMReranker:
    """
    Reranks documents using an LLM for more nuanced scoring.
    
    Uses prompt-based relevance scoring with Gemini.
    """
    
    def __init__(
        self,
        top_k: int = 5,
        model: str = "gemini-2.0-flash",
    ):
        self.top_k = top_k
        self.model = model
        self._llm = None
    
    @property
    def llm(self):
        """Lazy load the LLM."""
        if self._llm is None:
            from src.llmops.llm_manager import LLMManager
            manager = LLMManager(model=self.model)
            self._llm = manager.get_llm()
        return self._llm
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None,
    ) -> RerankResult:
        """
        Rerank using LLM-based relevance scoring.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            original_scores: Original retrieval scores
        
        Returns:
            RerankResult with reranked documents
        """
        if not documents:
            return RerankResult(documents=[], scores=[], original_scores=[])
        
        if original_scores is None:
            original_scores = [0.0] * len(documents)
        
        scores = []
        
        for doc in documents:
            prompt = f"""Rate the relevance of this document to the query.
            
Query: {query}

Document: {doc.page_content[:500]}

Rate from 0-10 where 10 is highly relevant. Reply with just the number."""
            
            try:
                response = self.llm.invoke(prompt)
                score = float(response.content.strip())
                scores.append(min(max(score, 0), 10))  # Clamp to 0-10
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")
                scores.append(5.0)  # Default mid-score
        
        # Sort by score
        doc_score_pairs = list(zip(documents, scores, original_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = doc_score_pairs[:self.top_k]
        
        return RerankResult(
            documents=[d[0] for d in top_docs],
            scores=[d[1] for d in top_docs],
            original_scores=[d[2] for d in top_docs],
        )


class HybridReranker:
    """
    Combines cross-encoder and LLM reranking.
    
    Uses cross-encoder for speed, with optional LLM refinement
    for top candidates.
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_top_k: int = 10,
        final_top_k: int = 5,
        use_llm_refinement: bool = False,
    ):
        self.cross_encoder = CrossEncoderReranker(
            model_name=cross_encoder_model,
            top_k=cross_encoder_top_k,
        )
        self.final_top_k = final_top_k
        self.use_llm_refinement = use_llm_refinement
        
        if use_llm_refinement:
            self.llm_reranker = LLMReranker(top_k=final_top_k)
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        original_scores: Optional[List[float]] = None,
    ) -> RerankResult:
        """Hybrid reranking with cross-encoder and optional LLM."""
        
        # First pass: cross-encoder
        ce_result = self.cross_encoder.rerank(query, documents, original_scores)
        
        if not self.use_llm_refinement:
            # Just return cross-encoder results
            return RerankResult(
                documents=ce_result.documents[:self.final_top_k],
                scores=ce_result.scores[:self.final_top_k],
                original_scores=ce_result.original_scores[:self.final_top_k],
            )
        
        # Second pass: LLM refinement on top candidates
        llm_result = self.llm_reranker.rerank(
            query,
            ce_result.documents,
            ce_result.scores,
        )
        
        return llm_result


def create_reranker(
    reranker_type: str = "cross-encoder",
    **kwargs,
) -> CrossEncoderReranker | LLMReranker | HybridReranker:
    """
    Factory function to create a reranker.
    
    Args:
        reranker_type: One of "cross-encoder", "llm", "hybrid"
        **kwargs: Additional arguments passed to reranker
    
    Returns:
        Configured reranker instance
    """
    rerankers = {
        "cross-encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "hybrid": HybridReranker,
    }
    
    if reranker_type not in rerankers:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
    
    return rerankers[reranker_type](**kwargs)
