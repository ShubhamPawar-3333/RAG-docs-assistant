# 🔍 Deep Dive: GitHub Actions CI/CD

## 🧠 Why GitHub Actions?

```
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS BENEFITS                       │
│                                                                  │
│  ✅ Native GitHub → No external CI service needed              │
│  ✅ Free tier → 2,000 min/month for public repos               │
│  ✅ Secrets → Secure API key management                        │
│  ✅ Matrix builds → Test multiple Python versions              │
│  ✅ Artifacts → Save evaluation results                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  On Pull Request:                                                │
│  ┌──────┐    ┌──────┐    ┌──────────┐    ┌───────┐             │
│  │ Lint │ → │ Test │ → │ Evaluate │ → │ Build │             │
│  └──────┘    └──────┘    └──────────┘    └───────┘             │
│                              ↓                                   │
│                    PR Comment with Results                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  On Merge to Main:                                               │
│  ┌────────────┐    ┌──────────────────┐                        │
│  │ Deploy API │ → │ Deploy Frontend │                        │
│  └────────────┘    └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Workflow Files

| File | Trigger | Purpose |
|------|---------|---------|
| `.github/workflows/ci.yml` | PR, push | Lint, test, evaluate |
| `.github/workflows/deploy.yml` | merge to main | Deploy to prod |
| `.github/dependabot.yml` | weekly | Keep deps updated |

---

## 📋 CI Pipeline Jobs

### 1. Lint & Type Check
```yaml
- name: Run Ruff linter
  run: ruff check src/ --output-format=github

- name: Run type check
  run: mypy src/ --ignore-missing-imports
```

### 2. Unit Tests
```yaml
- name: Run tests
  run: pytest tests/ -v --cov=src
  env:
    GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

### 3. RAG Evaluation (PRs only)
```yaml
- name: Run Promptfoo evaluation
  run: npx promptfoo eval -c promptfooconfig.yaml

- name: Comment PR with results
  uses: actions/github-script@v7
  # Posts pass rate to PR comments
```

### 4. Build Check
```yaml
- name: Verify imports
  run: |
    python -c "from src.rag import RAGPipeline"
    python -c "from src.api.main import app"
```

---

## 🚀 Deploy Pipeline

### Deployment Options
```yaml
# Render (recommended)
- name: Deploy to Render
  run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK }}"

# Railway
- uses: bervProject/railway-deploy@main

# Docker Hub
- run: docker push username/documind-api:latest
```

---

## ⚙️ Required Secrets

Add these in GitHub → Settings → Secrets → Actions:

| Secret | Purpose |
|--------|---------|
| `GOOGLE_API_KEY` | For Gemini API calls |
| `RENDER_DEPLOY_HOOK` | Trigger Render deploy |
| `RAILWAY_TOKEN` | Railway deployment |

---

## 🔄 Dependabot

Automatically updates:
- Python dependencies (weekly)
- GitHub Actions versions (weekly)
- npm/Promptfoo (monthly)

---

## 📊 PR Comment Example

```
## 🧪 RAG Evaluation Results

| Metric | Value |
|--------|-------|
| Tests Passed | 9/10 |
| Pass Rate | 90.0% |

View full results →
```

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `ci.yml` | Test on every PR |
| `deploy.yml` | Deploy on merge |
| `dependabot.yml` | Auto-update deps |
| Secrets | Secure credentials |
| Artifacts | Save eval results |
