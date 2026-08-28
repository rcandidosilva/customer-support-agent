---
id: kb-031
title: API rate limits
category: api
tags: [rate-limit, 429, throttle, quota, retry-after, burst]
audience: developer
last_reviewed: 2026-06-05
---

# Rate limits

Limits are per workspace, not per token.

| Plan       | Sustained | Burst |
|------------|-----------|-------|
| Starter    | 10 req/s  | 30    |
| Team       | 30 req/s  | 100   |
| Business   | 100 req/s | 300   |
| Enterprise | Negotiated in the order form | |

Every response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and
`X-RateLimit-Reset`. A `429` includes `Retry-After` in seconds.

## Guidance

- Retry `429` and `5xx` with exponential backoff and jitter. Do not retry `4xx` otherwise.
- The bulk export endpoints (`/v2/exports`) have a separate limit of 5 concurrent jobs and
  do not consume the request budget.
- Sustained-limit increases above the plan tier require a plan change or, on Enterprise, an
  amendment to the order form. Support cannot raise limits on request.
