"""
Streamlit Frontend Application

DocuMind AI - RAG-Powered Documentation Assistant
Chat interface for querying documents.
"""

import streamlit as st
import requests
from typing import Optional
import time
import os

# Page configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration - use localhost for HF Spaces (both run in same container)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ============== Custom CSS ==============
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
</style>
""", unsafe_allow_html=True)


# ============== Session State ==============
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = "documents"


# ============== Helper Functions ==============
def query_api(question: str, collection: str, top_k: int) -> Optional[dict]:
    """Query the RAG API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={
                "question": question,
                "collection_name": collection,
                "top_k": top_k,
                "include_sources": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def ingest_files(files, collection: str) -> Optional[dict]:
    """Upload files to the API for ingestion."""
    try:
        # Map extensions to MIME types (browsers often report wrong/empty types)
        mime_types = {
            ".pdf": "application/pdf",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
        }
        
        files_data = []
        for f in files:
            ext = "." + f.name.split(".")[-1].lower() if "." in f.name else ""
            content_type = mime_types.get(ext, f.type or "application/octet-stream")
            files_data.append(("files", (f.name, f.getvalue(), content_type)))
        
        response = requests.post(
            f"{API_BASE_URL}/api/ingest",
            files=files_data,
            data={"collection_name": collection},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# ============== Sidebar ==============
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.caption("Start the API with: `uvicorn src.api.main:app --reload`")
    
    st.divider()
    
    # Collection settings
    st.markdown("### 📁 Collection")
    st.session_state.collection_name = st.text_input(
        "Collection Name",
        value=st.session_state.collection_name,
        help="Name of the document collection to query"
    )
    
    # Retrieval settings
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve"
    )
    
    st.divider()
    
    # File upload
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, Markdown, Text"
    )
    
    if uploaded_files:
        if st.button("📥 Ingest Files", type="primary", use_container_width=True):
            with st.spinner("Processing files..."):
                result = ingest_files(uploaded_files, st.session_state.collection_name)
                if result and result.get("success"):
                    st.success(
                        f"✅ Ingested {result['documents_processed']} docs, "
                        f"{result['chunks_created']} chunks"
                    )
                    st.balloons()
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============== Main Content ==============

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 DocuMind AI</h1>
    <p>RAG-Powered Documentation Assistant</p>
</div>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}</strong> 
                            {f"(Score: {source.get('score', 0):.2f})" if source.get('score') else ""}
                            <br>
                            <small>{source.get('content', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_api(
                question=prompt,
                collection=st.session_state.collection_name,
                top_k=top_k,
            )
            
            if result:
                answer = result.get("answer", "Sorry, I couldn't generate a response.")
                sources = result.get("sources", [])
                
                st.markdown(answer)
                
                # Store message with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}</strong>
                                {f"(Score: {source.get('score', 0):.2f})" if source.get('score') else ""}
                                <br>
                                <small>{source.get('content', '')[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                error_msg = "Failed to get a response. Please check the API connection."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# Footer
st.markdown("---")
st.caption(
    "Built with ❤️ using LangChain, ChromaDB, and Google Gemini | "
    f"Collection: **{st.session_state.collection_name}**"
)
