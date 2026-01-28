# 🔍 Deep Dive: FastAPI Application Architecture

## 🧠 Why FastAPI for RAG?

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BENEFITS                              │
│                                                                  │
│  ✅ Async support → Efficient I/O for LLM calls                │
│  ✅ Automatic docs → Swagger UI at /docs                        │
│  ✅ Pydantic → Type validation and serialization                │
│  ✅ Streaming → SSE for real-time responses                     │
│  ✅ Production-ready → Used by Netflix, Uber, Microsoft         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Architecture Overview

```
src/api/
├── main.py           # Application entry point
├── models.py         # Pydantic request/response models
├── __init__.py
└── routes/
    ├── __init__.py
    ├── health.py     # Health check endpoints
    ├── query.py      # Query/RAG endpoints
    └── ingest.py     # Document ingestion endpoints
```

---

## 🎯 main.py - Application Entry Point

### FastAPI App Configuration

```python
app = FastAPI(
    title="DocuMind AI",
    description="RAG-Powered Documentation Assistant API",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc alternative
    lifespan=lifespan,     # Startup/shutdown events
)
```

### Lifespan Context Manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting DocuMind AI API...")
    yield
    # Shutdown
    logger.info("Shutting down...")
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # ["localhost:3000", "localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📋 models.py - Pydantic Schemas

### QueryRequest

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection_name: str = Field(default="documents")
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)
```

### QueryResponse

```python
class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: Optional[List[SourceDocument]]
    num_sources: Optional[int]
```

### IngestResponse

```python
class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    collection_name: str
```

---

## 🛣️ Routes

### Health Routes (`/health`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Component-level health |
| `/ready` | GET | Kubernetes readiness |
| `/live` | GET | Kubernetes liveness |

### Query Routes (`/api/query`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/query` | POST | Standard RAG query |
| `/api/query/stream` | POST | Streaming response |

### Ingest Routes (`/api/ingest`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ingest` | POST | Upload files |
| `/api/ingest/text` | POST | Ingest raw text |
| `/api/ingest/{collection}` | DELETE | Delete collection |

---

## 🔄 Query Flow

```
POST /api/query
{
    "question": "What is the refund policy?",
    "collection_name": "documents",
    "top_k": 5,
    "include_sources": true
}
    ↓
get_pipeline(collection_name, top_k)
    ↓
pipeline.query(question, include_sources)
    ↓
{
    "answer": "Based on the documentation...",
    "question": "What is the refund policy?",
    "sources": [...],
    "num_sources": 3
}
```

---

## 📤 Ingest Flow

```
POST /api/ingest
[files: policy.pdf, guide.md]
    ↓
Validate file extensions (.pdf, .md, .txt)
    ↓
Save to temp files
    ↓
MultiFormatDocumentLoader.load_file()
    ↓
DocumentChunker.chunk_documents()
    ↓
VectorStore.add_documents()
    ↓
{
    "success": true,
    "documents_processed": 2,
    "chunks_created": 15
}
```

---

## 🚀 Running the API

```bash
# Development
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📋 Summary

| File | Purpose |
|------|---------|
| `main.py` | App config, CORS, routers |
| `models.py` | Pydantic schemas |
| `routes/health.py` | Health checks |
| `routes/query.py` | RAG query endpoints |
| `routes/ingest.py` | File upload endpoints |
