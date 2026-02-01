#!/bin/bash

# Set production environment
export APP_ENV=production
export DEBUG=false

# Start FastAPI backend in background (production mode, no reload)
echo "Starting FastAPI backend (production)..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2 &

# Wait for API to be ready
sleep 5

# Start Streamlit frontend on HF Spaces port
echo "Starting Streamlit frontend (production)..."
streamlit run src/frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --browser.serverAddress aienthussp-documind-ai.hf.space \
    --browser.serverPort 443 \
    --server.enableXsrfProtection true
