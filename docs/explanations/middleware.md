# 🔍 Deep Dive: Streaming & Error Handling

## 🧠 Why Streaming for RAG?

```
┌─────────────────────────────────────────────────────────────────┐
│                    WITHOUT STREAMING                            │
│                                                                  │
│  User asks → [3-5 second wait...] → Full answer appears        │
│                                                                  │
│                    WITH STREAMING (SSE)                          │
│                                                                  │
│  User asks → Answer | streams | word | by | word | instantly    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Middleware Architecture

```
Request → RateLimitMiddleware → RequestLoggingMiddleware → ErrorHandlerMiddleware → Route
                                                                    ↓
Response ← RateLimitMiddleware ← RequestLoggingMiddleware ← ErrorHandlerMiddleware ←
```

---

## 🎯 ErrorHandlerMiddleware

```python
class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Error: {e}")
            return JSONResponse(status_code=500, content={...})
```

**Purpose:** Catch all unhandled exceptions and return proper JSON errors.

---

## 📊 RequestLoggingMiddleware

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        return response
```

**Purpose:** Log all requests with timing for observability.

---

## ⏱️ RateLimitMiddleware

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute=60):
        # Track requests per IP in memory
    
    async def dispatch(self, request, call_next):
        if too_many_requests:
            return JSONResponse(status_code=429, content={...})
        return await call_next(request)
```

**Purpose:** Prevent abuse with simple rate limiting (60 req/min default).

---

## 🌊 Server-Sent Events (SSE) Streaming

### Endpoint

```python
@router.post("/query/stream")
async def stream_query(request: QueryRequest):
    async def generate():
        for chunk in pipeline.stream(request.question):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### SSE Format

```
data: Based on
data:  the 
data: documentation
data: ,
data:  refunds
data:  are
data: ...
data: [DONE]
```

### Frontend Consumption

```javascript
const eventSource = new EventSource('/api/query/stream');

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
    } else {
        // Append chunk to UI
        responseDiv.textContent += event.data;
    }
};
```

---

## 📋 Summary

| Component | Purpose |
|-----------|---------|
| `ErrorHandlerMiddleware` | Global error catching |
| `RequestLoggingMiddleware` | Request timing logs |
| `RateLimitMiddleware` | Abuse prevention |
| SSE Streaming | Real-time response delivery |
| `X-Process-Time` header | Performance monitoring |
