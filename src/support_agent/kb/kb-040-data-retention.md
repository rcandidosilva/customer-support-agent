---
id: kb-040
title: Data retention, deletion, and export
category: security
tags: [retention, deletion, gdpr, dsar, export, subprocessors]
audience: admin
last_reviewed: 2026-06-25
---

# Data retention and deletion

| Event | Retention |
|-------|-----------|
| Plan downgrade to free | Synced data retained 90 days |
| Workspace suspension (non-payment) | Deleted at day 135 |
| Explicit workspace deletion | Purged within 30 days, backups within 60 |
| SCIM deprovisioned user | Dashboards retained 30 days |
| Audit logs | 400 days (Enterprise), 90 days (Business), 30 days otherwise |

## Data subject requests

GDPR/CCPA access and erasure requests must be submitted by a workspace admin through
**Settings → Privacy → Data requests**, or by emailing `privacy@northwind-analytics.com`.
Northwind responds within 30 days. Support agents must route these to the privacy team and
must not action them directly.

## Export

Admins can export all workspace data as JSON + Parquet under **Settings → Privacy →
Export workspace**. Exports over 50 GB are delivered as a signed S3 URL valid for 7 days.

Sub-processor list and the current SOC 2 Type II report are at
`northwind-analytics.com/trust`. NDAs for the SOC 2 report are handled by the security team.
