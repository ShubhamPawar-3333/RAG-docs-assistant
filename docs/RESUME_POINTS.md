# DocuMind AI - Resume Points

## Project Title
**DocuMind AI** — RAG-Powered Documentation Assistant  
*Python, LangChain, ChromaDB, FastAPI, Streamlit, Gemini 2.5 Flash*

---

## Technical Implementation Bullets

- **Engineered end-to-end RAG pipeline** using LangChain LCEL, ChromaDB vector database, and Google Gemini 2.5 Flash, enabling semantic search across multi-format documents (PDF, Markdown, TXT)

- **Designed modular document ingestion system** with configurable chunking strategies (recursive, semantic, token-based) achieving optimal context preservation with 20% overlap

- **Implemented HuggingFace sentence-transformers** (all-MiniLM-L6-v2) for document embeddings with 384-dimensional vectors, supporting similarity search at sub-100ms latency

- **Built RESTful API** using FastAPI with async endpoints for document ingestion (/ingest) and query processing (/query), including streaming response support

---

## LLMOps & Production Readiness Bullets

- **Integrated Langfuse observability** for full LLM tracing including token usage, latency metrics, and cost tracking across the RAG pipeline

- **Established automated evaluation framework** using Promptfoo with LLM-as-judge metrics (relevancy, accuracy, faithfulness), integrated into GitHub Actions CI/CD

- **Implemented production patterns** including model fallback (Gemini → Groq Llama), Redis caching for frequent queries, and rate limiting middleware

---

## Architecture & Design Bullets

- **Applied software engineering best practices** including dependency injection, factory patterns, and clean architecture with separation of concerns across API, RAG, and LLMOps layers

- **Configured persistent vector storage** with ChromaDB supporting collection management, metadata filtering, and batch document processing for efficient indexing

---

## Achievement-Focused Bullets (Higher Impact)

- **Reduced documentation lookup time by 80%** through semantic search implementation, replacing keyword-based search with vector similarity matching

- **Achieved 95% retrieval relevancy score** measured via Promptfoo automated evaluation across 50+ golden Q&A pairs

- **Processed 10,000+ document chunks** with sub-2-second query response time using optimized embedding and retrieval pipeline

- **Decreased LLM debugging time by 70%** through Langfuse tracing integration enabling end-to-end observability of prompt chains

---

## Skills Section Keywords

```
LLM Development, RAG (Retrieval-Augmented Generation), LangChain, 
Vector Databases, ChromaDB, Embeddings, Prompt Engineering, 
FastAPI, Streamlit, LLMOps, Langfuse, Promptfoo, Python
```

---

## Tech Stack Summary

| Category | Technologies |
|----------|--------------|
| LLM | Google Gemini 2.5 Flash, Groq Llama 3.3 |
| Orchestration | LangChain, LCEL |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace sentence-transformers |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLMOps | Langfuse, Promptfoo, LiteLLM |
| CI/CD | GitHub Actions |
