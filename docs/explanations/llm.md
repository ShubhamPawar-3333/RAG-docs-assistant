# 🔍 Deep Dive: LLM Module (Gemini Integration)

## 🧠 Role of LLM in RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                                  │
│                                                                  │
│  Query → Retriever → Context → LLM → Answer                    │
│                                  ↑                               │
│                          THIS MODULE                             │
│                                                                  │
│  The LLM takes retrieved context and generates human answers    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM MODULE                                    │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │   LLM_MODELS    │    │         LLMManager               │   │
│  │   (Registry)    │───►│   ┌─────────────────────────┐    │   │
│  └─────────────────┘    │   │ ChatGoogleGenerativeAI │    │   │
│                         │   │ _llm (lazy loaded)      │    │   │
│  ┌─────────────────┐    │   └─────────────────────────┘    │   │
│  │  LLMProvider    │    │                                   │   │
│  │  (Enum)         │    │   Methods:                        │   │
│  └─────────────────┘    │   - get_llm()                     │   │
│                         │   - generate()                    │   │
│                         │   - generate_with_context()       │   │
│                         └──────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Convenience Functions:                                      ││
│  │  - get_llm()  (singleton)                                   ││
│  │  - generate_response()                                      ││
│  │  - list_available_models()                                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 LLMProvider Enum

```python
class LLMProvider(Enum):
    GEMINI = "gemini"
    GROQ = "groq"  # For future fallback support
```

**Why Enum?**
- Type safety
- IDE autocomplete
- Easy to extend for fallback providers

---

## 📋 LLM_MODELS Registry

```python
LLM_MODELS = {
    "gemini-2.5-flash": {
        "provider": LLMProvider.GEMINI,
        "model_name": "gemini-2.5-flash",
        "description": "Fast, efficient for most tasks",
        "context_window": 1_000_000,  # 1 million tokens!
        "free_tier": True,
    },
    "gemini-2.5-pro": {...},
    "gemini-2.0-flash": {...},
}
```

### Model Comparison:

| Model | Speed | Quality | Context | Use Case |
|-------|-------|---------|---------|----------|
| `gemini-2.5-flash` | ⚡⚡⚡ | ⭐⭐⭐ | 1M | Default, fast responses |
| `gemini-2.5-pro` | ⚡⚡ | ⭐⭐⭐⭐ | 1M | Complex reasoning |
| `gemini-2.0-flash` | ⚡⚡⚡ | ⭐⭐⭐ | 1M | Latest stable |

---

## 🏗️ LLMManager Class

### Constructor
```python
def __init__(
    self,
    model_name: str = None,      # Default: gemini-2.5-flash
    temperature: float = None,    # Default: 0.3 (focused)
    max_tokens: int = None,       # Default: 2048
    api_key: Optional[str] = None # From settings or param
):
```

**Configuration from settings.py:**
```python
self.model_name = model_name or settings.default_model
self.temperature = temperature or settings.temperature
self.api_key = api_key or settings.google_api_key
```

### `get_llm()` — Lazy Loading
```python
def get_llm(self) -> BaseChatModel:
    if self._llm is None:
        self._llm = self._create_llm()
    return self._llm
```

**Why Lazy?**
```
❌ Without lazy loading:
import llm  ← API connection made immediately (slow)

✅ With lazy loading:
import llm  ← Instant
...
llm.generate("Hi")  ← Connection made here, only when needed
```

### `_create_llm()` — LangChain Integration
```python
def _create_llm(self) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=self.model_name,
        temperature=self.temperature,
        max_output_tokens=self.max_tokens,
        google_api_key=self.api_key,
        convert_system_message_to_human=True,  # Gemini quirk
    )
```

**`convert_system_message_to_human=True`:**
Gemini handles system prompts differently. This flag ensures compatibility with LangChain's message format.

---

### `generate()` — Basic Generation
```python
def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
```

**Flow:**
```python
messages = []
if system_prompt:
    messages.append(SystemMessage(content=system_prompt))
messages.append(HumanMessage(content=prompt))

response = llm.invoke(messages)
return response.content
```

**Example:**
```python
manager = LLMManager()
answer = manager.generate(
    "What is RAG?",
    system_prompt="You are a helpful AI teacher."
)
```

---

### `generate_with_context()` — RAG-Specific
```python
def generate_with_context(
    self,
    question: str,
    context: str,
    system_prompt: Optional[str] = None,
) -> str:
```

**The Core RAG Pattern:**
```python
prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""
```

**Default System Prompt:**
```
You are a helpful documentation assistant. 
Answer questions based on the provided context. 
If the context doesn't contain the answer, say so honestly.
Be concise but comprehensive.
```

**Usage:**
```python
# After retrieval
context = retriever.retrieve_with_context("refund policy")

# Generate answer using context
answer = llm_manager.generate_with_context(
    question="What is the refund policy?",
    context=context
)
```

---

## 🔄 Singleton Pattern

```python
_default_llm_manager: Optional[LLMManager] = None

def get_llm(model_name=None, temperature=None, force_new=False):
    global _default_llm_manager
    
    if force_new or _default_llm_manager is None:
        _default_llm_manager = LLMManager(...)
    
    return _default_llm_manager.get_llm()
```

**Benefits:**
- Model loaded once per process
- Efficient API connection reuse
- Consistent configuration

---

## 🔗 Integration Example

```python
# Complete RAG flow
from src.rag import create_retriever, LLMManager

# 1. Create retriever
retriever = create_retriever()

# 2. Get context
result = retriever.retrieve("What is the refund policy?")
context = result.get_context()

# 3. Generate answer
llm = LLMManager()
answer = llm.generate_with_context(
    question="What is the refund policy?",
    context=context
)

print(answer)
# "Based on the documentation, refunds are processed within 7 days..."
```

---

## 📋 Summary

| Component | Pattern | Purpose |
|-----------|---------|---------|
| `LLMProvider` | Enum | Type-safe provider selection |
| `LLM_MODELS` | Registry | Model configurations |
| `LLMManager` | Manager | Configure and use Gemini |
| `get_llm()` | Singleton + Lazy | Efficient model reuse |
| `generate()` | Basic API | Simple text generation |
| `generate_with_context()` | RAG Helper | Context-aware generation |
| `list_available_models()` | Helper | Discover options |
