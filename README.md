# RAG Documentation Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered documentation assistant using Retrieval-Augmented Generation (RAG) with full LLMOps integration.

## Features

- 🔍 **Semantic Search** - Find information using natural language queries
- 📚 **Multi-format Support** - PDF, Markdown, and text documents
- ⚡ **Fast Responses** - Sub-2s response times with caching
- 📊 **Full Observability** - Langfuse tracing and monitoring
- 🧪 **Automated Evaluation** - Promptfoo quality testing
- 🔄 **CI/CD Pipeline** - GitHub Actions for testing and deployment

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangChain + LCEL |
| Vector DB | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| Observability | Langfuse |
| Evaluation | Promptfoo |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-docs-assistant.git
cd rag-docs-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run src/frontend/streamlit_app.py
```

## Project Structure

```
rag-docs-assistant/
├── src/
│   ├── api/          # FastAPI backend
│   ├── rag/          # RAG pipeline components
│   ├── llmops/       # LLMOps integrations
│   └── frontend/     # Streamlit UI
├── evals/            # Evaluation datasets & configs
├── tests/            # Unit and integration tests
├── config/           # Configuration files
└── docs/             # Documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.
