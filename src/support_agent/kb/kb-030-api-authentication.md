---
id: kb-030
title: API authentication and tokens
category: api
tags: [api, token, auth, 401, scopes, rotation]
audience: developer
last_reviewed: 2026-06-05
---

# API authentication

Northwind's REST API lives at `https://api.northwind-analytics.com/v2`.

## Tokens

Create tokens under **Settings → API tokens**. Send them as
`Authorization: Bearer nw_live_...`.

- Token values are shown **once** at creation and cannot be retrieved afterward.
- Tokens are scoped per workspace, with scopes `read:data`, `write:data`, `read:admin`.
- Tokens do not expire by default. Setting an expiry is Business and Enterprise only.
- Tokens are **not** revoked by password reset or SSO enforcement. Rotate them explicitly.

## Common errors

| Status | Meaning |
|--------|---------|
| `401 invalid_token` | Token revoked, deleted, or from a different workspace |
| `403 insufficient_scope` | Token lacks the scope for this endpoint |
| `403 workspace_read_only` | Workspace is in dunning read-only state; see [kb-012] |
| `404` on a resource you can see in the UI | Token belongs to a different workspace |

## Deprecated v1

The `v1` API was retired on **2026-01-31**. `v1` endpoints return `410 Gone`. There is no
extension path; the v1-to-v2 mapping is in the developer docs.
