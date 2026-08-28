---
id: kb-020
title: Setting up data connectors
category: integrations
tags: [connector, sync, postgres, snowflake, bigquery, ip-allowlist]
audience: admin
last_reviewed: 2026-06-10
---

# Data connectors

Northwind syncs from Postgres, MySQL, Snowflake, BigQuery, Redshift, and S3 (CSV/Parquet).

## Setup

1. **Data → Connectors → New connector**.
2. Provide read-only credentials. Northwind never issues writes against source systems.
3. Allowlist our egress IPs: `52.19.44.0/24` and `34.242.11.0/24` (EU region), or
   `44.208.19.0/24` and `54.156.72.0/24` (US region).
4. Select the schemas to sync and a sync frequency.

## Sync frequency by plan

| Plan       | Minimum interval |
|------------|------------------|
| Starter    | 24 hours         |
| Team       | 6 hours          |
| Business   | 1 hour           |
| Enterprise | 5 minutes        |

Requesting a shorter interval than your plan allows silently clamps to the plan minimum —
the connector shows the effective interval, not the requested one.

## Connection failures

`connection refused` almost always means the IP allowlist step was missed or the source
firewall changed. `permission denied for schema` means the read-only role lacks `USAGE` on
that schema.
