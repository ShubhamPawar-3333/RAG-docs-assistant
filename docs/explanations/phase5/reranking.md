# 🚀 Phase 5: Reranking Optimization

## 🧠 What is Reranking?

Reranking is a **two-stage retrieval** strategy that improves result quality by:
1. **Stage 1 (Fast):** Retrieve many candidates using bi-encoder embeddings
2. **Stage 2 (Accurate):** Score candidates using a cross-encoder or LLM

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE RETRIEVAL                          │
│                                                                  │
│  Query: "What is the refund policy?"                            │
│       ↓                                                          │
│  [Stage 1: Bi-Encoder (Fast)]                                   │
│       → Retrieve top 20 candidates from vector store            │
│       ↓                                                          │
│  [Stage 2: Cross-Encoder (Accurate)]                            │
│       → Score each (query, document) pair                       │
│       → Sort by score                                            │
│       ↓                                                          │
│  Return top 5 reranked results                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Why Two Stages?

| Model Type | How It Works | Speed | Accuracy |
|------------|--------------|-------|----------|
| **Bi-Encoder** | Encodes query and documents separately, compares vectors | Very Fast | Good |
| **Cross-Encoder** | Encodes query + document together | Slow | Excellent |

**Trade-off:** Cross-encoders are too slow to run on millions of documents,
so we use bi-encoders for initial filtering and cross-encoders for refinement.

---

## 📁 File: `src/rag/reranking.py`

### Classes Overview

| Class | Purpose | Best For |
|-------|---------|----------|
| `RerankResult` | Dataclass for reranking output | - |
| `CrossEncoderReranker` | Fast, accurate reranking | Default choice |
| `LLMReranker` | Highest accuracy, uses Gemini | Quality-critical |
| `HybridReranker` | Combines both approaches | Best balance |

---

## 🔧 Class: `RerankResult`

```python
@dataclass
class RerankResult:
    documents: List[Document]      # Reranked documents
    scores: List[float]            # Cross-encoder scores
    original_scores: List[float]   # Original bi-encoder scores
```

**Purpose:** Structured output from reranking, preserves both original and new scores.

---

## 🔧 Class: `CrossEncoderReranker`

### Constructor
```python
def __init__(
    self,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
    batch_size: int = 32,
):
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | ms-marco-MiniLM-L-6-v2 | HuggingFace cross-encoder model |
| `top_k` | 5 | Number of documents to return |
| `batch_size` | 32 | Batch size for scoring |

### Method: `rerank()`
```python
def rerank(
    self,
    query: str,
    documents: List[Document],
    original_scores: Optional[List[float]] = None,
) -> RerankResult:
```

**How it works:**
1. Creates `(query, document)` pairs for each document
2. Runs cross-encoder model to score each pair
3. Sorts by score (descending)
4. Returns top-k documents

**Example:**
```python
reranker = CrossEncoderReranker(top_k=5)
result = reranker.rerank(
    query="refund policy",
    documents=retrieved_docs,
    original_scores=[0.9, 0.85, 0.7, 0.65, 0.5]
)
# result.documents = [doc2, doc1, doc4, doc3, doc5] (reordered)
# result.scores = [0.95, 0.92, 0.88, 0.72, 0.65] (cross-encoder scores)
```

---

## 🔧 Class: `LLMReranker`

### Constructor
```python
def __init__(
    self,
    top_k: int = 5,
    model: str = "gemini-2.0-flash",
):
```

### Method: `rerank()`

**How it works:**
1. For each document, asks LLM to rate relevance 0-10
2. Sorts by LLM score
3. Returns top-k

**Prompt used:**
```
Rate the relevance of this document to the query.
Query: {query}
Document: {doc.page_content[:500]}
Rate from 0-10 where 10 is highly relevant. Reply with just the number.
```

**Trade-off:** Slower and costs API calls, but highest accuracy.

---

## 🔧 Class: `HybridReranker`

```python
def __init__(
    self,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder_top_k: int = 10,
    final_top_k: int = 5,
    use_llm_refinement: bool = False,
):
```

**How it works:**
1. Cross-encoder narrows to top 10
2. (Optional) LLM refines to top 5
3. Returns highest quality results

---

## 🔧 Factory Function: `create_reranker()`

```python
def create_reranker(
    reranker_type: str = "cross-encoder",  # "cross-encoder", "llm", "hybrid"
    **kwargs,
) -> CrossEncoderReranker | LLMReranker | HybridReranker:
```

**Examples:**
```python
# Default cross-encoder
reranker = create_reranker("cross-encoder", top_k=5)

# LLM reranker (highest quality)
reranker = create_reranker("llm", top_k=3)

# Hybrid with LLM refinement
reranker = create_reranker("hybrid", use_llm_refinement=True)
```

---

## 🔗 Integration with Retriever

The `Retriever` class was updated to support reranking:

### Updated Constructor
```python
def __init__(
    self,
    vector_store: VectorStore,
    default_k: int = None,
    score_threshold: Optional[float] = None,
    reranker: Optional[Any] = None,      # NEW
    rerank_top_n: int = 20,               # NEW
):
```

### Updated `retrieve()` Flow
```
1. fetch_k = rerank_top_n if reranker else k
2. Retrieve fetch_k documents from vector store
3. If reranker:
   - rerank_result = reranker.rerank(query, documents, scores)
   - documents = rerank_result.documents[:k]
4. Return top k documents
```

---

## 📊 Performance Comparison

| Approach | Latency | Accuracy | Cost |
|----------|---------|----------|------|
| Bi-encoder only | ~50ms | Good | Free |
| + Cross-encoder | ~150ms | Excellent | Free |
| + LLM reranking | ~500ms | Best | API cost |

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `RerankResult` | Structured reranking output |
| `CrossEncoderReranker` | Fast, accurate (default) |
| `LLMReranker` | Highest quality, slower |
| `HybridReranker` | Best of both |
| `create_reranker()` | Factory for easy setup |
| `Retriever.reranker` | Integration point |
| `Retriever.rerank_top_n` | Candidates before reranking |
