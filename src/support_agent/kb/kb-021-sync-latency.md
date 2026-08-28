---
id: kb-021
title: Sync delays and stale dashboards
category: integrations
tags: [sync, latency, stale, backfill, freshness, delay]
audience: customer
last_reviewed: 2026-06-22
---

# Why is my dashboard stale?

Every dashboard shows a **freshness badge** with the timestamp of the last completed sync
for its underlying tables. Hover it to see the connector and the sync duration.

## Expected causes

- **The plan's sync interval.** See [kb-020]. A Team-plan connector will never be fresher
  than 6 hours.
- **A backfill is running.** New connectors and newly added tables backfill fully before
  incremental sync begins. Backfills of tables over 100M rows can take 12+ hours and pause
  incremental syncs for that connector.
- **Source-side long transactions.** Northwind reads from a consistent snapshot; a
  long-running transaction on the source can hold that snapshot back.

## When it is not expected

If the freshness badge is older than **3x your sync interval** and no backfill is shown on
**Data → Connectors → (connector) → Activity**, that is a fault. Sync workers are not
customer-visible and support must check the internal sync queue for the workspace.
