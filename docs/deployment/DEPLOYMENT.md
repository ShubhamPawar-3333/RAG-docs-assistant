# 🚀 Deployment Guide: Hugging Face Spaces

## Overview

DocuMind AI is deployed as a **Docker-based Hugging Face Space** running both:
- **FastAPI Backend** (port 8000)
- **Streamlit Frontend** (port 7860 - exposed)

---

## Prerequisites

1. **Hugging Face Account**: [huggingface.co](https://huggingface.co)
2. **Google API Key**: [Google AI Studio](https://aistudio.google.com/apikey)
3. **Langfuse Account** (optional): [langfuse.com](https://langfuse.com)

---

## Deployment Steps

### Step 1: Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure:
   - **Space name**: `documind-ai` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free) or CPU Upgrade for better performance
3. Click **Create Space**

### Step 2: Configure Secrets

In your Space Settings > Repository secrets, add:

| Secret Name | Required | Description |
|-------------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google Gemini API key |
| `LANGFUSE_PUBLIC_KEY` | ❌ No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | ❌ No | Langfuse secret key |
| `LANGFUSE_HOST` | ❌ No | `https://cloud.langfuse.com` |

### Step 3: Deploy via Git

```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/documind-ai
cd documind-ai

# Copy files from project
cp -r /path/to/rag-docs-assistant/* .

# Rename README for HF
cp README_HF.md README.md

# Push to HF
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Monitor Build

1. Go to your Space page
2. Click **Logs** tab
3. Watch the Docker build progress
4. Wait for "Running on..." message

---

## File Structure for Deployment

```
├── Dockerfile          # Multi-service container
├── start.sh            # Launches FastAPI + Streamlit
├── README.md           # HF Space metadata (from README_HF.md)
├── requirements.txt    # Python dependencies
├── src/
│   ├── api/           # FastAPI backend
│   ├── frontend/      # Streamlit app
│   └── rag/           # RAG pipeline
└── data/              # Created at runtime
```

---

## Environment Variables in Docker

The Dockerfile sets these automatically:

```dockerfile
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
```

Secrets from HF are injected as environment variables.

---

## Troubleshooting

### Build Fails

1. Check **Logs** for error messages
2. Common issues:
   - Missing dependencies in `requirements.txt`
   - Syntax errors in Python files
   - Port conflicts

### App Not Loading

1. Ensure port 7860 is exposed
2. Check Streamlit is running in headless mode
3. Verify `start.sh` has execute permissions

### API Errors

1. Check `GOOGLE_API_KEY` is set correctly
2. Verify API key has Gemini access
3. Check rate limits

---

## Local Testing (Optional)

```bash
# Build Docker image
docker build -t documind-ai .

# Run with environment variables
docker run -p 7860:7860 \
  -e GOOGLE_API_KEY=your_key \
  documind-ai

# Access at http://localhost:7860
```

---

## CI/CD with GitHub Actions

The `.github/workflows/deploy.yml` automates deployment:

1. Push to `main` branch
2. GitHub Action triggers
3. Code synced to HF Space
4. Space rebuilds automatically

Required GitHub Secrets:
- `HF_TOKEN`: Hugging Face access token
- `HF_USERNAME`: Your HF username

---

## Monitoring

- **Spaces Logs**: Real-time container logs
- **Langfuse**: LLM call traces (if configured)
- **Space Analytics**: Usage statistics

---

## Resources

- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Project README](./README.md)
