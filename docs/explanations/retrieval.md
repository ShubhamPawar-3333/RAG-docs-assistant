# 🔍 Deep Dive: Retrieval Pipeline Architecture

## 🧠 What is Retrieval in RAG?

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG RETRIEVAL FLOW                            │
│                                                                  │
│  User Query: "What is the refund policy?"                       │
│        ↓                                                         │
│  Retriever.retrieve(query)                                      │
│        ↓                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │ 1. Embed query                          │                    │
│  │ 2. Search vector store                  │                    │
│  │ 3. Return top-k relevant documents      │                    │
│  │ 4. Package into RetrievalResult         │                    │
│  └─────────────────────────────────────────┘                    │
│        ↓                                                         │
│  Context: "Refunds processed within 7 days..."                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL MODULE                              │
│                                                                  │
│  ┌────────────────┐    ┌─────────────────┐                      │
│  │ RetrievalResult│    │   Retriever     │                      │
│  │ (dataclass)    │◄───│   (main class)  │                      │
│  └────────────────┘    └────────┬────────┘                      │
│         ↑                       │                                │
│         │              ┌────────┴────────┐                      │
│         │              │  VectorStore    │                      │
│         │              │  (from Task 6)  │                      │
│         │              └─────────────────┘                      │
│         │                                                        │
│  ┌──────┴─────────┐    ┌─────────────────┐                      │
│  │ RetrieverFactory│    │ create_retriever│                     │
│  │ (factory class)│    │ (convenience fn)│                      │
│  └────────────────┘    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 RetrievalResult Dataclass

```python
@dataclass
class RetrievalResult:
    documents: List[Document]        # Retrieved documents
    scores: Optional[List[float]]    # Relevance scores
    query: str                       # Original query
    metadata: Optional[Dict]         # Additional info
```

### Why a Dataclass?

| Benefit | Explanation |
|---------|-------------|
| **Structured output** | All retrieval data in one object |
| **Type hints** | IDE autocomplete, error catching |
| **Immutable-ish** | Clean, predictable usage |
| **Extensible** | Easy to add fields |

### Key Methods:

#### `num_results` (Property)
```python
@property
def num_results(self) -> int:
    return len(self.documents)
```
Quick way to check how many documents were retrieved.

#### `get_context()` — Extract Combined Text
```python
def get_context(self, separator: str = "\n\n") -> str:
    return separator.join(doc.page_content for doc in self.documents)
```

**Use Case:**
```python
result = retriever.retrieve("refund policy")
context = result.get_context()
# "Refunds within 7 days...\n\nReturn shipping is free..."
# → Pass to LLM as context
```

#### `filter_by_score()` — Quality Filtering
```python
def filter_by_score(self, min_score: float) -> "RetrievalResult":
    # Keep only documents above threshold
```

**Why Score Filtering?**
```
Without filtering:
Doc 1: "Refund policy..."     (score: 0.95) ← Relevant
Doc 2: "Return shipping..."   (score: 0.82) ← Relevant
Doc 3: "Contact support..."   (score: 0.31) ← NOT relevant!

With filtering (min_score=0.5):
Only Doc 1 and Doc 2 are kept
→ Better context quality for LLM
```

---

## 🏗️ Retriever Class

### Constructor
```python
def __init__(
    self,
    vector_store: VectorStore,      # Where to search
    default_k: int = None,          # Default results count
    score_threshold: float = None,  # Auto-filter threshold
):
```

**Dependency Injection:** Takes vector store as parameter, not creating it internally.

### `retrieve()` — Main Search Method
```python
def retrieve(
    self,
    query: str,
    k: int = None,
    filter: Optional[Dict] = None,
    include_scores: bool = True,
) -> RetrievalResult:
```

**Flow:**
```
┌──────────────────────────────────────────────────────────────────┐
│  retrieve("What is the refund policy?")                          │
│       ↓                                                          │
│  if include_scores:                                              │
│       similarity_search_with_score()                             │
│  else:                                                           │
│       similarity_search()                                        │
│       ↓                                                          │
│  Package into RetrievalResult                                    │
│       ↓                                                          │
│  if score_threshold:                                             │
│       result.filter_by_score(threshold)                          │
│       ↓                                                          │
│  Return RetrievalResult                                          │
└──────────────────────────────────────────────────────────────────┘
```

### `retrieve_with_context()` — Convenience Method
```python
def retrieve_with_context(self, query: str, k: int = None) -> str:
    result = self.retrieve(query, k=k, include_scores=False)
    return result.get_context(separator="\n\n---\n\n")
```

**For RAG pipelines:** Get context string directly without handling RetrievalResult.

### `get_langchain_retriever()` — LangChain Integration
```python
def get_langchain_retriever(self, **kwargs):
    return self.vector_store.as_retriever(**kwargs)
```

**Why?** LangChain LCEL chains need `Retriever` interface:
```python
chain = (
    {"context": retriever.get_langchain_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

---

## 🏭 RetrieverFactory Class

```python
class RetrieverFactory:
    def __init__(self, collection_name: str, embedding_model: str):
        # Store configuration
    
    def get_vector_store(self) -> VectorStore:
        # Create embeddings + vector store (lazy)
    
    def create_retriever(self, default_k, score_threshold) -> Retriever:
        # Create configured Retriever
```

**Factory Pattern Benefits:**
- Encapsulates complex setup (embeddings → vector store → retriever)
- Reusable configuration
- Testable (mock the factory)

---

## 🔄 Convenience Function

```python
def create_retriever(
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    default_k: int = None,
    score_threshold: float = None,
) -> Retriever:
```

**One-liner setup:**
```python
# Simple
retriever = create_retriever()

# Customized
retriever = create_retriever(
    collection_name="my_docs",
    default_k=10,
    score_threshold=0.5
)
```

---

## 📋 Summary

| Component | Pattern | Purpose |
|-----------|---------|---------|
| `RetrievalResult` | Dataclass | Structured retrieval output |
| `get_context()` | Helper | Extract text for LLM |
| `filter_by_score()` | Filter | Remove low-quality results |
| `Retriever` | Query Interface | Main search class |
| `RetrieverFactory` | Factory | Manage setup complexity |
| `create_retriever()` | Facade | Simple one-liner API |
