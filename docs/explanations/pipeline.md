# 🔍 Deep Dive: RAG Pipeline with LCEL

## 🧠 What is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to compose LangChain components into chains using the pipe (`|`) operator.

```python
# LCEL syntax
chain = prompt | llm | output_parser

# Equivalent to
def chain(input):
    prompted = prompt.format(input)
    response = llm.generate(prompted)
    return output_parser.parse(response)
```

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                                  │
│                                                                  │
│  User Query                                                      │
│      ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    RETRIEVAL STAGE                           ││
│  │  Retriever.retrieve(query) → RetrievalResult → context      ││
│  └─────────────────────────────────────────────────────────────┘│
│      ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    GENERATION STAGE (LCEL)                   ││
│  │  {context, question} → prompt → LLM → output_parser         ││
│  └─────────────────────────────────────────────────────────────┘│
│      ↓                                                           │
│  Answer with Sources                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 DEFAULT_RAG_PROMPT

```python
DEFAULT_RAG_PROMPT = """You are a helpful documentation assistant.
Answer the question based ONLY on the following context.
If the context doesn't contain enough information to answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
```

**Key Elements:**
| Element | Purpose |
|---------|---------|
| Role instruction | Sets assistant behavior |
| "ONLY" constraint | Prevents hallucination |
| Honesty instruction | Admits when info is missing |
| `{context}` | Placeholder for retrieved docs |
| `{question}` | Placeholder for user query |

---

## 🏗️ RAGPipeline Class

### Constructor

```python
def __init__(
    self,
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = None,
    temperature: float = None,
    top_k: int = None,
    prompt_template: str = None,
):
```

**Fully Configurable:**
- Database collection
- Embedding model
- LLM model and temperature
- Retrieval depth (top_k)
- Custom prompts

### Lazy Loading Properties

```python
@property
def retriever(self) -> Retriever:
    if self._retriever is None:
        self._retriever = create_retriever(...)
    return self._retriever

@property
def chain(self):
    if self._chain is None:
        self._chain = self._build_chain()
    return self._chain
```

**Why Lazy?**
- Fast initialization
- Resources loaded only when needed
- Efficient memory usage

---

### `_build_chain()` — LCEL Chain Construction

```python
def _build_chain(self):
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(self.prompt_template)
    
    # Get LLM
    llm = self.llm_manager.get_llm()
    
    # Output parser
    output_parser = StrOutputParser()
    
    # Build the chain using LCEL
    chain = prompt | llm | output_parser
    
    return chain
```

**Data Flow:**
```
Input: {"context": "...", "question": "..."}
    ↓
ChatPromptTemplate.from_template()
    → Formatted prompt string
    ↓
ChatGoogleGenerativeAI
    → AIMessage with response
    ↓
StrOutputParser
    → Plain string answer
```

---

### `query()` — Main Query Method

```python
def query(
    self,
    question: str,
    include_sources: bool = False,
) -> Dict[str, Any]:
```

**Flow:**
```
Step 1: Retrieve
retrieval_result = self.retriever.retrieve(query=question)

Step 2: Format Context
context = retrieval_result.get_context()

Step 3: Generate Answer
answer = self.chain.invoke({
    "context": context,
    "question": question,
})

Step 4: Return Response
{
    "answer": "Based on the docs...",
    "question": "What is...?",
    "sources": [...],  # If include_sources=True
}
```

**Source Attribution:**
```python
if include_sources:
    response["sources"] = [
        {
            "content": doc.page_content[:200] + "...",
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in zip(...)
    ]
```

Why sources? — Users can verify answers against original documents.

---

### `stream()` — Streaming Responses

```python
def stream(self, question: str) -> Iterator[str]:
    # Retrieve context first
    context = retrieval_result.get_context()
    
    # Stream from LLM
    for chunk in self.chain.stream({
        "context": context,
        "question": question,
    }):
        yield chunk
```

**Why Streaming?**
- Better UX — users see response building
- Lower perceived latency
- Works with chat interfaces

**Usage:**
```python
for chunk in pipeline.stream("What is RAG?"):
    print(chunk, end="", flush=True)
```

---

## 🏭 RAGPipelineBuilder (Builder Pattern)

```python
pipeline = (
    RAGPipelineBuilder()
    .with_collection("my_docs")
    .with_model("gemini-2.5-pro")
    .with_temperature(0.5)
    .with_top_k(10)
    .build()
)
```

**Why Builder Pattern?**
| Benefit | Explanation |
|---------|-------------|
| Fluent API | Readable, chainable |
| Optional params | Set only what you need |
| Validation | Can validate before build |
| Immutable result | Built object is configured |

### Builder Methods:

```python
.with_collection(name)      # Set ChromaDB collection
.with_embedding_model(model) # Set embedding model
.with_model(model)          # Set LLM model
.with_temperature(temp)      # Set temperature
.with_top_k(k)              # Set retrieval count
.with_prompt(template)      # Set custom prompt
.build()                    # Create RAGPipeline
```

---

## 🔄 Convenience Function

```python
def create_rag_pipeline(
    collection_name: str = "documents",
    llm_model: str = None,
    top_k: int = None,
) -> RAGPipeline:
```

**Simple one-liner:**
```python
pipeline = create_rag_pipeline()
answer = pipeline.query("What is RAG?")
```

---

## 🔗 Complete Usage Example

```python
from src.rag import create_rag_pipeline, load_documents, chunk_documents

# 1. Ingest documents (one-time)
docs = load_documents("data/docs/")
chunks = chunk_documents(docs)
# store.add_documents(chunks)  # Add to vector store

# 2. Create pipeline
pipeline = create_rag_pipeline(collection_name="my_docs")

# 3. Query
result = pipeline.query(
    "What is the refund policy?",
    include_sources=True
)

print(result["answer"])
# "Based on the documentation, refunds are processed within 7 business days..."

for source in result["sources"]:
    print(f"- {source['metadata']['file_name']} (score: {source['score']:.2f})")
```

---

## 📋 Summary

| Component | Pattern | Purpose |
|-----------|---------|---------|
| `DEFAULT_RAG_PROMPT` | Template | Standard RAG instruction |
| `RAGPipeline` | Facade | Complete RAG pipeline |
| `_build_chain()` | Factory | Create LCEL chain |
| `query()` | Query | Full retrieval + generation |
| `stream()` | Iterator | Streaming responses |
| `RAGPipelineBuilder` | Builder | Fluent pipeline configuration |
| `create_rag_pipeline()` | Factory | Quick pipeline creation |
