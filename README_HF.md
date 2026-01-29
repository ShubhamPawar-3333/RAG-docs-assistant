---
title: DocuMind AI
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DocuMind AI

RAG-Powered Documentation Assistant built with LangChain, ChromaDB, and Google Gemini.

## Features

- 💬 Chat interface for document Q&A
- 📤 Upload PDF, Markdown, and text files
- 🔍 Semantic search with embeddings
- 📚 Source attribution with citations

## Usage

1. Upload your documents using the sidebar
2. Ask questions about your documents
3. View source citations for transparency

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **LLM**: Google Gemini
