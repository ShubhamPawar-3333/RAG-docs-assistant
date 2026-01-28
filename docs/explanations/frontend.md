# 🔍 Deep Dive: Streamlit Frontend

## 🧠 Why Streamlit for RAG?

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT BENEFITS                            │
│                                                                  │
│  ✅ Python-only → No JS/HTML required                           │
│  ✅ Built-in chat → st.chat_input, st.chat_message             │
│  ✅ File upload → st.file_uploader                              │
│  ✅ Reactive → Auto-refresh on state changes                    │
│  ✅ Free hosting → Streamlit Cloud                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Application Architecture

```
src/frontend/
├── app.py            # Main Streamlit application
└── __init__.py

.streamlit/
└── config.toml       # Theme configuration
```

---

## 🎯 Key Components

### Session State

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = "documents"
```

**Purpose:** Persist data across reruns (Streamlit reruns on every interaction).

---

### Chat Interface

```python
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        result = query_api(prompt, collection, top_k)
        st.markdown(result["answer"])
```

---

### File Upload

```python
uploaded_files = st.file_uploader(
    "Drop files here",
    type=["pdf", "md", "txt"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Ingest"):
    files_data = [("files", (f.name, f.getvalue())) for f in files]
    response = requests.post(f"{API_URL}/api/ingest", files=files_data)
```

---

### Source Attribution

```python
if sources:
    with st.expander("📚 View Sources"):
        for source in sources:
            st.markdown(f"""
            <div class="source-card">
                <strong>Source {i}</strong> (Score: {score})
                <br>{content[:200]}...
            </div>
            """, unsafe_allow_html=True)
```

---

## 🎨 Custom Styling

```python
st.markdown("""
<style>
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 20px 20px 5px 20px;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa, #e4e8eb);
        border-radius: 20px 20px 20px 5px;
    }
    
    .source-card {
        border-left: 4px solid #667eea;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)
```

---

## 🔄 Data Flow

```
User Input (st.chat_input)
    ↓
Add to session_state.messages
    ↓
POST /api/query
    ↓
Display answer + sources
    ↓
Store response in session_state
```

---

## 🚀 Running the App

```bash
# Start API first
uvicorn src.api.main:app --reload

# Then start Streamlit (new terminal)
streamlit run src/frontend/app.py
```

### Access Points:
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `app.py` | Main chat interface |
| `config.toml` | Theme colors & settings |
| Session State | Message persistence |
| `st.chat_input` | User input box |
| `st.file_uploader` | Document upload |
| Custom CSS | Gradient styling |
