# 🔍 Deep Dive: Document Loaders Architecture

## 📐 Overall Design Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY PATTERN                              │
│                                                                  │
│  User Input → MultiFormatDocumentLoader → Appropriate Loader    │
│                                                                  │
│  "doc.pdf"  ──┐                                                  │
│  "notes.md" ──┼──► SUPPORTED_EXTENSIONS dict ──► Right Loader   │
│  "data.txt" ──┘                                                  │
└─────────────────────────────────────────────────────────────────┘
```

We use the **Strategy Pattern** — one class (`MultiFormatDocumentLoader`) that delegates to the right strategy (PDF/MD/TXT loader) based on file extension.

---

## 🏗️ Class Architecture

```python
class MultiFormatDocumentLoader:
    """
    Central orchestrator for loading documents of various formats.
    """
```

### 1️⃣ Class-Level Configuration

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,   # LangChain's PDF loader
    ".md": TextLoader,     # Markdown read as plain text (see note below)
    ".markdown": TextLoader,
    ".txt": TextLoader,    # Plain text
}
```

> **Note:** `.md` was previously mapped to `UnstructuredMarkdownLoader`. That loader
> pulls in `unstructured`, which downloads a spaCy model at runtime on first use and
> fails hard offline / behind strict TLS, making every Markdown ingest return HTTP 500.
> Markdown is already text, so `TextLoader` is used instead. See
> [`docs/fixes/2026-09-05-ingestion-and-event-loop-blocking.md`](../fixes/2026-09-05-ingestion-and-event-loop-blocking.md).

**Why this design?**
- **Dictionary as registry** — O(1) lookup for file type → loader mapping
- **Easy to extend** — Add new format: just add one line
- **Decoupled** — Loader classes are injected, not hardcoded in methods

```python
# Adding a new format later:
SUPPORTED_EXTENSIONS[".html"] = BSHTMLLoader  # One-line addition!
```

---

### 2️⃣ The `load_file` Method (Core Logic)

```python
def load_file(self, file_path: Union[str, Path]) -> List[Document]:
```

**Flow Diagram:**
```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Input Path │───►│  Validation  │───►│  Extension      │
│  (str/Path) │    │  - exists?   │    │  Extraction     │
└─────────────┘    │  - is file?  │    │  (.pdf .md .txt)│
                   └──────────────┘    └────────┬────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY SELECTION                        │
│                                                              │
│  extension → SUPPORTED_EXTENSIONS[extension] → LoaderClass  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOADER INSTANTIATION                      │
│                                                              │
│  if .txt: TextLoader(path, encoding="utf-8")               │
│  else:    LoaderClass(path)                                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                METADATA ENRICHMENT                           │
│                                                              │
│  doc.metadata["source"] = full_path                         │
│  doc.metadata["file_name"] = "document.pdf"                 │
│  doc.metadata["file_type"] = ".pdf"                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                    Return List[Document]
```

**Key Engineering Decisions:**

| Decision | Reasoning |
|----------|-----------|
| `Union[str, Path]` input | Flexibility — user can pass either |
| Convert to `Path` immediately | Consistent API for all operations |
| Validate before processing | Fail fast — clear error messages |
| Enrich metadata | **Critical for RAG** — need to cite sources |

---

### 3️⃣ The `load_directory` Method (Bulk Processing)

```python
def load_directory(
    self,
    directory_path: Union[str, Path],
    recursive: bool = True,           # Search subdirectories?
    extensions: Optional[List[str]] = None  # Filter by type?
) -> List[Document]:
```

**Flow:**
```python
# 1. Use glob patterns based on recursive flag
pattern = "**/*" if recursive else "*"

# 2. For each extension (.pdf, .md, .txt)
for ext in extensions:
    # 3. Find matching files
    files = directory_path.glob(f"{pattern}{ext}")
    
    # 4. Load each file (reusing load_file method)
    for file in files:
        docs = self.load_file(file)  # ← DRY principle!
```

**Why `**/*` pattern?**
```
data/
├── docs/
│   ├── guide.md        ← Matched with **/*.md (recursive)
│   └── sub/
│       └── deep.md     ← Also matched (recursive)
├── readme.md           ← Matched
└── notes.txt           ← Matched with **/*.txt
```

**Error Handling Philosophy:**
```python
try:
    docs = self.load_file(file_path)
except Exception as e:
    logger.warning(f"Skipping file {file_path}: {e}")
    continue  # ← Don't fail entire batch for one bad file
```

This is **graceful degradation** — one corrupt PDF shouldn't stop processing 100 other files.

---

### 4️⃣ The `load_files` Method (Explicit List)

```python
def load_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
```

When the user already knows exactly which files to load:
```python
loader.load_files([
    "chapter1.pdf",
    "chapter2.pdf", 
    "appendix.md"
])
```

Same graceful error handling — skip failures, continue with rest.

---

## 🎯 Convenience Function

```python
def load_documents(path: Union[str, Path], recursive: bool = True) -> List[Document]:
```

**Why a standalone function?**

```python
# Without convenience function:
loader = MultiFormatDocumentLoader()
if path.is_file():
    docs = loader.load_file(path)
elif path.is_dir():
    docs = loader.load_directory(path)

# With convenience function:
docs = load_documents(path)  # ← One line!
```

This follows the **Facade Pattern** — hide complexity behind simple interface.

---

## 📦 LangChain Document Structure

Each document returned is a LangChain `Document` object:

```python
Document(
    page_content="The actual text content of the document...",
    metadata={
        "source": "data/sample_docs/sample.pdf",
        "file_name": "sample.pdf",
        "file_type": ".pdf",
        "page": 1,  # Added by PDF loader
        # ... other loader-specific metadata
    }
)
```

**Why is metadata important for RAG?**

```
User asks: "What is the refund policy?"

RAG system returns:
┌───────────────────────────────────────────────────────┐
│ Answer: Refunds are processed within 7 days...        │
│                                                       │
│ Source: policies/refund-policy.pdf (Page 3)          │  ← Metadata!
└───────────────────────────────────────────────────────┘
```

Without metadata, you can't tell the user WHERE the information came from.

---

## 🔧 Error Handling Design

```python
class DocumentLoaderError(Exception):
    """Custom exception for document loading errors."""
    pass
```

**Why custom exceptions?**

```python
# User code can catch specific errors:
try:
    docs = load_documents("data/")
except DocumentLoaderError as e:
    # Handle our errors specifically
    show_user_friendly_message(e)
except ValueError as e:
    # Handle unsupported format
    suggest_supported_formats()
```

**Exception hierarchy:**
```
Exception
├── DocumentLoaderError      ← File not found, access denied
└── ValueError               ← Unsupported format
```

---

## 📋 Summary

| Component | Design Pattern | Purpose |
|-----------|---------------|---------|
| `SUPPORTED_EXTENSIONS` | Registry | Map extension → loader class |
| `load_file` | Strategy | Delegate to right loader |
| `load_directory` | Iterator | Process multiple files |
| `load_documents` | Facade | Simple one-liner interface |
| `DocumentLoaderError` | Custom Exception | Clean error handling |

This is **production-grade code** because:
1. ✅ Single Responsibility — each method does one thing
2. ✅ Open/Closed — add formats without modifying existing code
3. ✅ Liskov Substitution — all loaders return same `Document` type
4. ✅ Graceful degradation — one failure doesn't break everything
5. ✅ Rich metadata — enables source citation in RAG
