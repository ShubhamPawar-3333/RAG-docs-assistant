# 🔍 Deep Dive: Langfuse Observability

## 🧠 Why Langfuse for LLMOps?

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGFUSE BENEFITS                             │
│                                                                  │
│  ✅ Full tracing → See every step of your RAG pipeline          │
│  ✅ Token tracking → Monitor usage and costs                    │
│  ✅ Latency metrics → Identify bottlenecks                      │
│  ✅ Free tier → 50k observations/month                          │
│  ✅ LangChain native → Drop-in callback handler                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Architecture

```
RAG Query
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Langfuse Trace                                                  │
│  ├── Span: "retrieval" (input: query, output: documents)       │
│  └── Generation: "llm-call" (model, tokens, latency)           │
└─────────────────────────────────────────────────────────────────┘
    ↓
Langfuse Dashboard → Analytics, Costs, Debugging
```

---

## 🎯 Key Components

### LangfuseTracer Class

```python
tracer = LangfuseTracer(trace_name="rag-query")
tracer.start_trace(user_id="user-123", tags=["production"])

# Track retrieval
span = tracer.span("retrieval", input_data={"query": question})
# ... retrieval logic ...
span.end(output={"num_docs": 5})

# Track LLM generation
tracer.generation(
    name="answer-generation",
    model="gemini-2.5-flash",
    input=prompt,
    output=response,
    usage={"input_tokens": 500, "output_tokens": 200},
)

tracer.end_trace(status="success")
```

---

### LangChain Callback Handler

```python
from src.llmops import get_langfuse_callback_handler

handler = get_langfuse_callback_handler()

# Attach to chain
chain = prompt | llm | parser
chain_with_tracing = chain.with_config(callbacks=[handler])

# Every invocation is now traced automatically
response = chain_with_tracing.invoke({"question": "..."})
```

---

### Decorator for Easy Tracing

```python
from src.llmops import trace_rag_query

@trace_rag_query
def process_query(question: str, _tracer=None):
    # _tracer is injected automatically
    span = _tracer.span("custom-step", input_data=question)
    # ... your logic ...
    span.end()
    return result
```

---

## 🔄 Integration with Pipeline

```python
# In pipeline.py _build_chain()
chain = prompt | llm | output_parser

# Add Langfuse if configured
if is_langfuse_enabled():
    chain = create_traced_chain(chain, trace_name="rag-pipeline")
```

---

## ⚙️ Configuration

**.env file:**
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Get keys from:** https://cloud.langfuse.com

---

## 📊 What Gets Tracked

| Metric | Description |
|--------|-------------|
| Trace | Full request lifecycle |
| Spans | Individual pipeline steps |
| Generations | LLM calls with tokens |
| Latency | Time per component |
| Costs | Token costs per model |
| Metadata | User ID, session, tags |

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `LangfuseTracer` | Manual trace control |
| `get_langfuse_callback_handler()` | LangChain integration |
| `create_traced_chain()` | Wrap chains with tracing |
| `@trace_rag_query` | Decorator for functions |
| `is_langfuse_enabled()` | Check if configured |
