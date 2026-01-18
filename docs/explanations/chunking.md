# 🔍 Deep Dive: Document Chunking Pipeline

## 🧠 Why Chunking Matters in RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE CHUNKING PROBLEM                          │
│                                                                  │
│  Large Document (50,000 tokens)                                  │
│        ↓                                                         │
│  Embedding Model (max 512 tokens)  ← CAN'T PROCESS!             │
│        ↓                                                         │
│  Solution: Split into smaller chunks                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Embedding models have token limits. We must split documents into digestible pieces while preserving semantic meaning.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY PATTERN                              │
│                                                                  │
│  DocumentChunker                                                 │
│       │                                                          │
│       ├── RECURSIVE ──► RecursiveCharacterTextSplitter          │
│       │                 (paragraphs → sentences → words)         │
│       │                                                          │
│       ├── SEMANTIC ───► RecursiveCharacterTextSplitter          │
│       │                 (with regex sentence detection)          │
│       │                                                          │
│       └── TOKEN ──────► SentenceTransformersTokenTextSplitter   │
│                         (actual token counting)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 The ChunkingStrategy Enum

```python
class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"   # Default, most versatile
    SEMANTIC = "semantic"     # Better sentence preservation
    TOKEN = "token"           # Exact token counting
```

**Why Enum?**
- **Type safety** — can't pass invalid strategy
- **IDE autocomplete** — easier to use
- **Self-documenting** — clear what options exist

---

## 🏗️ DocumentChunker Class

### Constructor (`__init__`)

```python
def __init__(
    self,
    chunk_size: int = None,          # Max characters per chunk
    chunk_overlap: int = None,        # Overlap between chunks
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
):
```

**The Overlap Concept (Critical for RAG):**

```
WITHOUT OVERLAP:
┌──────────┐┌──────────┐┌──────────┐
│  Chunk 1 ││  Chunk 2 ││  Chunk 3 │
└──────────┘└──────────┘└──────────┘
     ↑              ↑
     └── Information at boundaries is LOST!

WITH OVERLAP (200 chars):
┌──────────────┐
│   Chunk 1    │
└────────┬─────┘
         │overlap
    ┌────┴─────────┐
    │   Chunk 2    │
    └────────┬─────┘
             │overlap
        ┌────┴─────────┐
        │   Chunk 3    │
        └──────────────┘
        
↑ Sentences at boundaries are preserved in BOTH chunks!
```

**Default values from settings:**
```python
self.chunk_size = chunk_size or settings.chunk_size      # 1000 chars
self.chunk_overlap = chunk_overlap or settings.chunk_overlap  # 200 chars
```

---

### The `_create_splitter` Method (Factory Pattern)

```python
def _create_splitter(self):
    """Create the text splitter based on strategy."""
```

This is a **Factory Method** — creates the right object based on configuration.

#### Strategy 1: RECURSIVE (Default, Best for General Use)

```python
return RecursiveCharacterTextSplitter(
    chunk_size=self.chunk_size,
    chunk_overlap=self.chunk_overlap,
    separators=[
        "\n\n",      # 1. Try paragraphs first
        "\n",        # 2. Then line breaks
        ". ",        # 3. Then sentences
        "! ",        # 4. Exclamations
        "? ",        # 5. Questions
        "; ",        # 6. Semicolons
        ", ",        # 7. Commas
        " ",         # 8. Words
        "",          # 9. Characters (last resort)
    ],
)
```

**How Recursive Splitting Works:**

```
Input: "Hello world. This is a test.\n\nNew paragraph here."

Step 1: Try splitting on "\n\n" (paragraphs)
├── "Hello world. This is a test."  ← Chunk 1
└── "New paragraph here."           ← Chunk 2

If chunks still too big, recursively try next separator...

Step 2: Try splitting on ". " (sentences)
├── "Hello world"                   ← Sub-chunk
└── "This is a test."               ← Sub-chunk
```

**The hierarchy matters!** We want semantically meaningful breaks.

#### Strategy 2: SEMANTIC (Regex-based Sentence Detection)

```python
separators=[
    "\n\n",                    # Paragraphs
    "\n",                      # Lines
    "(?<=[.!?]) ",             # ← REGEX: Split AFTER sentence-ending punctuation
    " ",                       # Words
    "",                        # Characters
],
is_separator_regex=True,       # Enable regex mode
```

**What `(?<=[.!?]) ` means:**
- `(?<=...)` — **Lookbehind assertion** (match position AFTER these chars)
- `[.!?]` — Match period, exclamation, or question mark
- ` ` — Followed by a space

```
"Hello world. This is great! Right?"
              ↑              ↑
              Split HERE     Split HERE
              (after ". ")   (after "! ")
```

This preserves complete sentences better than character-based splitting.

#### Strategy 3: TOKEN (For Model Compatibility)

```python
return SentenceTransformersTokenTextSplitter(
    chunk_overlap=min(self.chunk_overlap, 50),  # Token overlap (not chars)
    tokens_per_chunk=self.chunk_size // 4,      # ~4 chars per token avg
)
```

**Why tokens matter:**
- Models count **tokens**, not characters
- "Hello" = 1 token
- "extraordinary" = 3 tokens (`extra`, `ordin`, `ary`)

This strategy ensures chunks fit exactly in model context windows.

---

### The `chunk_document` Method

```python
def chunk_document(self, document: Document) -> List[Document]:
```

**What it does:**

```python
# 1. Split the document
chunks = self._splitter.split_documents([document])

# 2. Enrich metadata (CRITICAL for RAG!)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = i           # Position in original
    chunk.metadata["total_chunks"] = len(chunks) # Total pieces
    chunk.metadata["chunk_size"] = len(chunk.page_content)
    chunk.metadata["chunking_strategy"] = self.strategy.value
```

**Why metadata enrichment?**

```
User asks: "What's in section 3 of the document?"

Without metadata:
→ Can only say "Here's the answer"

With metadata:
→ "Here's the answer from policies.pdf (chunk 3 of 10)"
```

---

### The `chunk_documents` Method (Batch Processing)

```python
def chunk_documents(self, documents: List[Document]) -> List[Document]:
    all_chunks: List[Document] = []
    
    for document in documents:
        try:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Error chunking: {e}")
            continue  # ← Skip failures, process rest
    
    return all_chunks
```

**Design Decisions:**

| Pattern | Why |
|---------|-----|
| `try/except` per doc | One bad doc doesn't kill entire batch |
| `extend` not `append` | Flatten into single list |
| Logging warnings | Track failures without crashing |

---

### The `get_chunk_stats` Method (Debugging Helper)

```python
def get_chunk_stats(self, chunks: List[Document]) -> dict:
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(sizes) / len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "total_characters": sum(sizes),
    }
```

**Use case:**
```python
stats = chunker.get_chunk_stats(chunks)
# {
#   "total_chunks": 47,
#   "avg_chunk_size": 892.3,
#   "min_chunk_size": 156,      ← Maybe too small?
#   "max_chunk_size": 1000,
#   "total_characters": 41938
# }
```

This helps tune chunk_size parameters.

---

## 🔄 Convenience Function

```python
def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: str = "recursive"  # ← String, not Enum (simpler API)
) -> List[Document]:
```

**Facade Pattern** — simpler interface for common use:

```python
# Without convenience function:
from src.rag.chunking import DocumentChunker, ChunkingStrategy
chunker = DocumentChunker(
    chunk_size=1000,
    strategy=ChunkingStrategy.RECURSIVE
)
chunks = chunker.chunk_documents(docs)

# With convenience function:
from src.rag.chunking import chunk_documents
chunks = chunk_documents(docs, chunk_size=1000, strategy="recursive")
```

---

## 📊 How Chunking Affects RAG Quality

```
┌────────────────────────────────────────────────────────────────┐
│                    CHUNK SIZE TRADEOFFS                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SMALL CHUNKS (200 chars)          LARGE CHUNKS (2000 chars)   │
│  ├── More precise                  ├── More context             │
│  ├── May lose context              ├── May include irrelevant   │
│  ├── More API calls                ├── Fewer API calls          │
│  └── Risk: incomplete answers      └── Risk: noisy answers      │
│                                                                 │
│  SWEET SPOT: 500-1000 chars with 10-20% overlap                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Integration with Other Modules

```python
# Task 4: Load documents
documents = load_documents("data/docs/")

# Task 5: Chunk documents (THIS MODULE)
chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

# Task 6: Create embeddings & store in ChromaDB
vector_store = Chroma.from_documents(chunks, embedding_function)

# Task 8: Retrieve relevant chunks
results = vector_store.similarity_search("What is the refund policy?")
```

---

## 📋 Summary

| Component | Pattern Used | Purpose |
|-----------|--------------|---------|
| `ChunkingStrategy` | Enum | Type-safe strategy selection |
| `DocumentChunker` | Strategy + Factory | Flexible splitter creation |
| `_create_splitter` | Factory Method | Create right splitter for strategy |
| `chunk_document` | Decorator | Enrich with metadata |
| `chunk_documents` | Iterator | Batch processing with fault tolerance |
| `chunk_documents()` | Facade | Simple one-liner API |
