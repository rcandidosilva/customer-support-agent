---
id: kb-012
title: Failed payments and dunning
category: billing
tags: [payment-failed, card-declined, dunning, suspension, 3ds]
audience: admin
last_reviewed: 2026-05-30
---

# Failed payments

When a charge fails, Northwind retries on days **1, 3, 7, and 14**. The billing contact is
emailed after each failure.

## What happens to the workspace

- Days 1-14: full access, in-app warning banner.
- Day 15: the workspace is **read-only**. Dashboards render; scheduled reports, API writes,
  and data syncs are paused.
- Day 45: the workspace is suspended and data is queued for deletion at day 135.

Updating the card under **Settings → Billing → Payment method** triggers an immediate retry
and restores access within a few minutes if it succeeds.

## Common decline causes

- **3-D Secure challenge not completed** — the bank required verification and the browser
  tab was closed. Retry from the billing page and complete the bank prompt.
- **Card issuer blocks recurring international charges.** Northwind charges from Ireland;
  some issuers require the cardholder to authorize this once.
- **Insufficient authorization amount** on virtual/prepaid cards.

Support cannot see full card numbers or retry a payment on the customer's behalf.
