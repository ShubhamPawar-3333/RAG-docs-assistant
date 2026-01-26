# 🔍 Deep Dive: End-to-End Testing

## 🧠 Why E2E Testing for RAG?

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE COMPONENTS                       │
│                                                                  │
│  Each component can work individually but may fail together:    │
│                                                                  │
│  [Loader] → [Chunker] → [Embeddings] → [VectorStore]           │
│                                              ↓                   │
│                              [Retriever] → [LLM] → [Answer]     │
│                                                                  │
│  E2E tests verify the ENTIRE chain works together               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Test Script Architecture

```
scripts/
├── test_e2e.py    # Comprehensive test suite
└── demo.py        # Interactive demo
```

---

## 🎯 test_e2e.py - Comprehensive Test Suite

### Test Flow:

```python
1. test_document_loading()   # Load sample documents
       ↓
2. test_chunking()           # Split into chunks
       ↓
3. test_embeddings()         # Initialize embedding model
       ↓
4. test_vector_store()       # Store chunks with embeddings
       ↓
5. test_retrieval()          # Search for relevant docs
       ↓
6. test_llm()                # Test LLM generation
       ↓
7. test_rag_pipeline()       # Full RAG query
       ↓
8. cleanup_test_collection() # Remove test data
```

### Key Test Methods:

#### `test_document_loading()`
```python
loader = MultiFormatDocumentLoader()
documents = loader.load_directory(sample_dir)
# Verifies: Files are read, metadata is extracted
```

#### `test_chunking()`
```python
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
# Verifies: Documents are split, stats are correct
```

#### `test_embeddings()`
```python
manager = EmbeddingsManager()
vector = manager.embed_text("What is RAG?")
# Verifies: Model loads, produces 384-dim vectors
```

#### `test_vector_store()`
```python
store = create_vector_store(embeddings, collection_name="test_collection")
ids = store.add_documents(chunks)
# Verifies: Chunks are stored, IDs are returned
```

#### `test_retrieval()`
```python
retriever = Retriever(vector_store=store, default_k=3)
result = retriever.retrieve("What is RAG?")
# Verifies: Relevant docs are returned with scores
```

#### `test_llm()`
```python
manager = LLMManager()
response = manager.generate("Say hello")
# Verifies: API key works, model responds
```

#### `test_rag_pipeline()`
```python
pipeline = create_rag_pipeline(collection_name="test_collection")
result = pipeline.query("What is RAG?", include_sources=True)
# Verifies: Full pipeline works, sources included
```

---

## 🎮 demo.py - Interactive Demo

### Usage:
```bash
python scripts/demo.py
```

### Flow:
```
1. Load documents from data/sample_docs/
2. Chunk documents
3. Index in vector store
4. Start interactive Q&A loop
5. User asks questions
6. Pipeline retrieves + generates
7. Display answer with sources
```

### Interactive Loop:
```python
while True:
    question = input("Your question: ")
    result = pipeline.query(question, include_sources=True)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
```

---

## 🔧 Running the Tests

### Full Test Suite:
```bash
# Activate virtual environment
venv\Scripts\activate

# Run E2E tests
python scripts/test_e2e.py
```

### Interactive Demo:
```bash
# Activate virtual environment
venv\Scripts\activate

# Run demo
python scripts/demo.py
```

---

## ✅ Expected Output

```
============================================================
  🚀 RAG Pipeline End-to-End Test Suite
============================================================

============================================================
  1. Testing Document Loading
============================================================
✅ Loaded 1 document(s)
   - sample.txt: 1024 chars

============================================================
  2. Testing Document Chunking
============================================================
✅ Created 3 chunks
   - Average chunk size: 341 chars

============================================================
  3. Testing Embeddings
============================================================
✅ Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2
   - Vector dimensions: 384

============================================================
  4. Testing Vector Store
============================================================
✅ Added 3 chunks to vector store
   - Collection: test_collection
   - Document count: 3

============================================================
  5. Testing Retrieval
============================================================
📝 Query: 'What is RAG?'
   Retrieved 3 documents
   Top score: 0.8542

============================================================
  6. Testing LLM
============================================================
✅ LLM configured: gemini-2.5-flash
   - Test response: Hello, RAG!

============================================================
  7. Testing Complete RAG Pipeline
============================================================
✅ RAG Pipeline created
📝 Testing query: 'What is RAG and what are its benefits?'
💬 Answer: Based on the documentation, RAG is a technique...

============================================================
  8. Cleanup
============================================================
✅ Test collection cleaned up

============================================================
  Test Summary
============================================================
✅ All tests passed!

🎉 Phase 1 Complete: Core RAG Pipeline is working!
```

---

## 📋 Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_e2e.py` | Verify all components | After code changes |
| `demo.py` | Interactive testing | Demo to users |
