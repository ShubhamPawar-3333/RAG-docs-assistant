# 🧪 Project Testing Roadmap

Complete sequential guide to test all components of DocuMind AI.

---

## Prerequisites

```bash
cd "f:\LLM projects\rag-docs-assistant"

# Ensure virtual environment is activated
# Create .env file with your API key
echo "GOOGLE_API_KEY=your_actual_api_key" > .env
```

---

## Phase 1: Unit Tests (5 min)

### 1.1 Run All Unit Tests
```bash
python -m pytest tests/unit/ -v --tb=short
```

**Expected:** All tests pass ✅

### 1.2 Run Specific Test Modules
```bash
# Test caching
python -m pytest tests/unit/test_caching.py -v

# Test async utilities
python -m pytest tests/unit/test_async_utils.py -v

# Test loaders
python -m pytest tests/unit/test_loaders.py -v

# Test embeddings
python -m pytest tests/unit/test_embeddings.py -v
```

---

## Phase 2: Integration Tests (5 min)

### 2.1 Run API Integration Tests
```bash
python -m pytest tests/integration/test_api.py -v
```

**Expected:** Health endpoints, query, ingest tests pass ✅

---

## Phase 3: Backend API Testing (10 min)

### 3.1 Start FastAPI Server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 3.2 Test Health Endpoint
Open browser or use curl:
```bash
curl http://localhost:8000/health
```
**Expected:** `{"status": "healthy", ...}`

### 3.3 Test API Docs
Open: http://localhost:8000/docs

**Expected:** Swagger UI with all endpoints visible

### 3.4 Test Ingest Endpoint (with sample file)
Create a test document:
```bash
echo "RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation." > test_doc.txt
```

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "files=@test_doc.txt" \
  -F "collection_name=test_collection"
```

**Expected:** `{"success": true, "documents_processed": 1, ...}`

### 3.5 Test Query Endpoint
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "collection_name": "test_collection"}'
```

**Expected:** JSON response with `answer` and `sources`

---

## Phase 4: Frontend Testing (10 min)

### 4.1 Start Streamlit (in new terminal)
```bash
streamlit run src/frontend/app.py
```

### 4.2 Open Browser
Navigate to: http://localhost:8501

### 4.3 Test UI Components
| Test | Action | Expected |
|------|--------|----------|
| Page loads | Open URL | Chat interface visible |
| Sidebar | Check left panel | Collection settings, file uploader |
| Upload file | Drag test_doc.txt | Success message |
| Send query | Type "What is RAG?" | Answer with sources |
| Sources shown | Check response | Citations displayed |

---

## Phase 5: RAG Pipeline Testing (10 min)

### 5.1 Python REPL Test
```python
# Start Python
python

# Test imports
from src.rag import create_rag_pipeline, load_documents, chunk_documents
from src.core.config import settings

# Test document loading
docs = load_documents("test_doc.txt")
print(f"Loaded {len(docs)} documents")

# Test chunking
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

# Test pipeline (requires GOOGLE_API_KEY in .env)
pipeline = create_rag_pipeline(collection_name="test_cli")
result = pipeline.query("What is RAG?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

---

## Phase 6: Evaluation Testing (5 min)

### 6.1 Run Promptfoo Evaluation
```bash
cd eval
npx promptfoo eval --config promptfooconfig.yaml
```

**Expected:** Evaluation results with pass/fail metrics

### 6.2 View Results
```bash
npx promptfoo view
```

Opens browser with evaluation dashboard.

---

## Phase 7: End-to-End Flow (10 min)

### 7.1 Full Workflow Test

1. **Start both servers:**
   ```bash
   # Terminal 1: Backend
   uvicorn src.api.main:app --port 8000
   
   # Terminal 2: Frontend
   streamlit run src/frontend/app.py
   ```

2. **Upload documents:**
   - Go to http://localhost:8501
   - Upload a PDF/MD/TXT file via sidebar

3. **Query documents:**
   - Type a question about uploaded content
   - Verify answer is relevant
   - Check sources are cited

4. **Verify observability:**
   - Check terminal logs for traces
   - If Langfuse configured, check dashboard

---

## Quick Validation Checklist

| Component | Test Command | Pass Criteria |
|-----------|--------------|---------------|
| Unit Tests | `pytest tests/unit/ -v` | All green |
| API Health | `curl localhost:8000/health` | 200 OK |
| Swagger UI | Open `/docs` | Renders correctly |
| Streamlit | Open `:8501` | Chat UI loads |
| File Upload | Upload test file | Success message |
| Query | Ask question | Gets answer |
| Sources | Check response | Citations shown |

---

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### API Key Missing
```bash
# Check .env file exists with:
GOOGLE_API_KEY=your_key_here
```

### Port Already in Use
```bash
# Kill existing process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### ChromaDB Errors
```bash
# Reset vector store
rm -rf data/chroma
```

---

## Success Criteria

✅ All unit tests pass  
✅ API endpoints respond correctly  
✅ Frontend loads and is interactive  
✅ File upload works  
✅ Queries return relevant answers  
✅ Sources are properly cited
