# Nimbus Analytics — Events API Reference (v2)

## Base URL

```
https://api.nimbusanalytics.com/v2
```

## Authentication

Every request must include a project API key as a bearer token:

```
Authorization: Bearer <API_KEY>
```

Requests without a valid key return `401 Unauthorized`. Keys are created in
**Workspace settings → API keys** and can be scoped to a single project.

## Rate limits

- **600 requests per minute** per project.
- Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header
  (in seconds).
- The Events API is exempt from rate limiting for batches under 100 events.

## Send events

`POST /events`

- Maximum **500 events per request**.
- Each event `timestamp` must be **ISO 8601 in UTC** (for example `2026-09-05T12:30:00Z`).
- Events older than 90 days are rejected with `422 Unprocessable Entity`.

```bash
curl -X POST https://api.nimbusanalytics.com/v2/events \
  -H "Authorization: Bearer $NIMBUS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"name": "checkout_completed", "user_id": "u_123", "timestamp": "2026-09-05T12:30:00Z"}
    ]
  }'
```

```python
import requests

requests.post(
    "https://api.nimbusanalytics.com/v2/events",
    headers={"Authorization": f"Bearer {key}"},
    json={"events": [{"name": "checkout_completed", "user_id": "u_123",
                      "timestamp": "2026-09-05T12:30:00Z"}]},
)
```

## Other endpoints

| Method | Path                | Purpose                              |
|--------|---------------------|--------------------------------------|
| GET    | `/funnels/{id}`     | Retrieve a funnel definition + results |
| GET    | `/dashboards`       | List dashboards in the project        |
| POST   | `/export`           | Start an async CSV/JSON export job     |
| GET    | `/export/{job_id}`  | Check export job status               |

Export jobs are asynchronous: `POST /export` returns a `job_id`, and the result file
URL is available from `GET /export/{job_id}` once `status` is `completed`.
