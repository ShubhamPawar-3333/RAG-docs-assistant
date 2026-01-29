# 🔍 Deep Dive: Promptfoo RAG Evaluation

## 🧠 Why Promptfoo?

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPTFOO BENEFITS                            │
│                                                                  │
│  ✅ Declarative YAML → Easy test case management               │
│  ✅ LLM-as-Judge → GPT evaluates answer quality                 │
│  ✅ CI/CD ready → GitHub Actions integration                    │
│  ✅ Multi-provider → Test across different models               │
│  ✅ Free & open → No vendor lock-in                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Evaluation Architecture

```
Golden Test Cases (YAML)
    ↓
Promptfoo Config
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  For each test:                                                  │
│  1. Inject vars into prompt template                            │
│  2. Call LLM provider (Gemini)                                  │
│  3. Run assertions (contains, llm-rubric, relevance)           │
│  4. Record pass/fail, latency, cost                            │
└─────────────────────────────────────────────────────────────────┘
    ↓
JSON Results → Markdown Report
```

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `eval/promptfooconfig.yaml` | Main config with prompts & tests |
| `eval/golden_tests.yaml` | Expected Q&A pairs |
| `eval/runner.py` | Python wrapper for automation |
| `eval/results/` | Output directory for results |

---

## 📋 Test Categories

### 1. Factual Accuracy
```yaml
- description: "Factual retrieval"
  vars:
    question: "What is RAG?"
    context: "RAG combines retrieval with generation..."
  assert:
    - type: contains
      value: "retrieval"
    - type: llm-rubric
      value: "Answer correctly explains RAG"
```

### 2. Faithfulness (No Hallucination)
```yaml
- description: "Should not hallucinate"
  assert:
    - type: not-contains
      value: "PostgreSQL"
    - type: llm-rubric
      value: "Answer only uses facts from context"
```

### 3. Relevance
```yaml
- description: "Answer should be relevant"
  assert:
    - type: relevance
      threshold: 0.7
```

### 4. Edge Cases
```yaml
- description: "Handle empty context"
  vars:
    question: "What is X?"
    context: ""
  assert:
    - type: llm-rubric
      value: "Admits information is not available"
```

---

## 🔄 Assertion Types

| Type | Purpose |
|------|---------|
| `contains` | Response includes text |
| `not-contains` | Response excludes text |
| `contains-any` | Response includes any of list |
| `llm-rubric` | GPT judges quality |
| `relevance` | Semantic similarity score |
| `cost` | Max cost per query |
| `latency` | Max response time |

---

## 🚀 Running Evaluations

```bash
# Install Promptfoo (one-time)
npm install -g promptfoo

# Run evaluation
cd eval
npx promptfoo eval

# View results in browser
npx promptfoo view

# Run with Python wrapper
python eval/runner.py
```

---

## 📊 Sample Output

```
┌──────────────────────────────────────────────────────────────┐
│ RAG Evaluation Results                                        │
├──────────────────────────────────────────────────────────────┤
│ Total Tests: 10                                               │
│ Passed: 9                                                     │
│ Failed: 1                                                     │
│ Pass Rate: 90%                                                │
│ Avg Latency: 1,234ms                                          │
│ Total Cost: $0.0089                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔗 CI/CD Integration

```yaml
# .github/workflows/eval.yml
- name: Run RAG Evaluation
  run: |
    npx promptfoo eval -c eval/promptfooconfig.yaml
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `promptfooconfig.yaml` | Test definitions |
| `golden_tests.yaml` | Expected behaviors |
| `runner.py` | Python automation |
| `llm-rubric` | AI-powered evaluation |
| CI/CD | Automated quality checks |
