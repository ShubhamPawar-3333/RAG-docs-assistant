# 🔍 Deep Dive: HuggingFace Embeddings Architecture

## 🧠 What Are Embeddings?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT → VECTOR TRANSFORMATION                  │
│                                                                  │
│  "The refund policy allows returns within 30 days"              │
│                           ↓                                      │
│              Embedding Model (Neural Network)                    │
│                           ↓                                      │
│  [0.023, -0.156, 0.892, 0.044, ..., -0.221]  (384 dimensions)   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Embeddings capture **semantic meaning** — similar concepts have similar vectors.

```
"refund policy"     → [0.23, -0.45, 0.12, ...]
"return guidelines" → [0.21, -0.42, 0.15, ...]  ← Similar vectors!
"weather forecast"  → [0.89, 0.12, -0.76, ...]  ← Very different
```

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EmbeddingsManager                            │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │ EMBEDDING_MODELS│    │  HuggingFaceEmbeddings           │    │
│  │ (Registry)      │───►│  _embeddings (lazy loaded)       │    │
│  └─────────────────┘    └──────────────────────────────────┘    │
│                                                                 │
│   Model Aliases:                                                │
│   ├── "all-MiniLM-L6-v2"    → sentence-transformers/...         │
│   ├── "all-mpnet-base-v2"   → sentence-transformers/...         │
│   ├── "bge-small-en-v1.5"   → BAAI/...                          │
│   └── "e5-small-v2"         → intfloat/...                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLETON PATTERN                            │
│                                                                 │
│  get_embeddings() ──► _default_manager ──► Reuse same instance  │
│                                                                 │
│  First call:  Load model (slow, ~5 sec)                         │
│  Next calls:  Return cached instance (instant)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Model Registry

```python
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_seq_length": 256,
        "description": "Fast, good quality, small size (80MB)",
    },
    # ... more models
}
```

### Why a Registry?

| Benefit | Explanation |
|---------|-------------|
| **Alias mapping** | User says "all-MiniLM-L6-v2" instead of full path |
| **Metadata access** | Know dimensions without loading model |
| **Easy switching** | Change model with one string change |
| **Documentation** | Self-documenting available options |

### Model Comparison:

| Model | Dimensions | Size | Speed | Quality |
|-------|------------|------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | 80MB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good |
| `all-mpnet-base-v2` | 768 | 420MB | ⚡⚡ Medium | ⭐⭐⭐⭐ Better |
| `bge-small-en-v1.5` | 384 | 130MB | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ State-of-art |
| `e5-small-v2` | 384 | 130MB | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Excellent |

---

## 🏗️ EmbeddingsManager Class

### Constructor

```python
def __init__(
    self,
    model_name: str = DEFAULT_MODEL,    # Which model to use
    device: str = "cpu",                 # CPU or GPU
    normalize_embeddings: bool = True,   # L2 normalization
    cache_folder: Optional[str] = None,  # Where to cache model
):
```

**Key Design Decisions:**

| Parameter | Purpose |
|-----------|---------|
| `model_name` | Alias or full HuggingFace path |
| `device` | "cpu" for compatibility, "cuda" for speed |
| `normalize_embeddings` | Makes cosine similarity = dot product |
| `cache_folder` | Avoid re-downloading models |

### Why Normalize Embeddings?

```
Without normalization:
  Cosine similarity requires: dot(a,b) / (||a|| * ||b||)
  
With normalization (||v|| = 1):
  Cosine similarity = dot(a,b)  ← Much faster!
```

---

### Lazy Loading Pattern

```python
def get_embeddings(self) -> Embeddings:
    if self._embeddings is None:        # First time?
        self._embeddings = self._create_embeddings()  # Load now
    return self._embeddings             # Return cached
```

**Why Lazy Loading?**

```
❌ Eager Loading (in __init__):
┌─────────────────────────────────────────────────────────────┐
│  import embeddings  ← Takes 5 seconds even if not used!    │
└─────────────────────────────────────────────────────────────┘

✅ Lazy Loading:
┌─────────────────────────────────────────────────────────────┐
│  import embeddings  ← Instant!                              │
│  # Later, only when needed:                                 │
│  embeddings.embed("text")  ← Model loads here               │
└─────────────────────────────────────────────────────────────┘
```

---

### The `_create_embeddings` Method

```python
def _create_embeddings(self) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": self.device}
    encode_kwargs = {"normalize_embeddings": self.normalize_embeddings}
    
    return HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs=model_kwargs,      # Passed to model loading
        encode_kwargs=encode_kwargs,    # Passed to encoding
    )
```

**What Happens Internally:**
1. Download model from HuggingFace Hub (cached)
2. Load into memory (CPU or GPU)
3. Initialize tokenizer
4. Return ready-to-use embeddings

---

### `embed_text` and `embed_texts` Methods

```python
def embed_text(self, text: str) -> List[float]:
    """Single text → single vector"""
    embeddings = self.get_embeddings()
    return embeddings.embed_query(text)

def embed_texts(self, texts: List[str]) -> List[List[float]]:
    """Multiple texts → multiple vectors (batched)"""
    embeddings = self.get_embeddings()
    return embeddings.embed_documents(texts)
```

**Why Two Different Methods?**

| Method | Use Case | Optimization |
|--------|----------|--------------|
| `embed_query` | User's question | Single text |
| `embed_documents` | Bulk ingestion | Batched processing |

---

## 🔄 Singleton Pattern

```python
_default_manager: Optional[EmbeddingsManager] = None

def get_embeddings(model_name: str = DEFAULT_MODEL, ...) -> Embeddings:
    global _default_manager
    
    if _default_manager is None:
        _default_manager = EmbeddingsManager(model_name=model_name)
    
    return _default_manager.get_embeddings()
```

**Why Singleton?**

```
❌ Without Singleton:
Request 1: Load model (5 sec) → Embed → Response
Request 2: Load model (5 sec) → Embed → Response  ← Wasteful!

✅ With Singleton:
Request 1: Load model (5 sec) → Embed → Response
Request 2: Reuse model → Embed → Response  ← Instant!
```

---

## 🔗 Integration with VectorStore

```python
from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import create_vector_store

# 1. Get embeddings (singleton, loaded once)
embeddings = get_embeddings()

# 2. Create vector store with embeddings
store = create_vector_store(embeddings)

# 3. Add documents (embeddings used automatically)
store.add_documents(chunks)

# 4. Search (query embedded automatically)
results = store.similarity_search("What is the refund policy?")
```

---

## 📋 Summary

| Component | Pattern | Purpose |
|-----------|---------|---------|
| `EMBEDDING_MODELS` | Registry | Map aliases to full model names |
| `EmbeddingsManager` | Manager | Configure and hold embedding model |
| `get_embeddings()` | Singleton + Lazy | Efficient model reuse |
| `_create_embeddings()` | Factory | Build HuggingFace instance |
| `embed_text/texts` | Adapter | Unified embedding interface |
| `list_available_models()` | Helper | Discover available options |
