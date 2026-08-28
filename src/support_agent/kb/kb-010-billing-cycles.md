---
id: kb-010
title: Billing cycles, proration, and invoices
category: billing
tags: [invoice, proration, billing-cycle, upgrade, downgrade, seats]
audience: admin
last_reviewed: 2026-06-18
---

# Billing cycles and proration

Northwind bills monthly or annually in advance. The cycle anchor is the date the paid
subscription started, not the first of the month.

## Mid-cycle seat changes

- **Adding seats** charges a prorated amount immediately for the remainder of the cycle.
- **Removing seats** does not refund. The seat count drops at the next renewal, and the
  credit is applied against that renewal invoice.

## Plan upgrades and downgrades

- **Upgrade** takes effect immediately; the unused portion of the current plan is credited
  against the new plan's prorated charge.
- **Downgrade** takes effect at the end of the current cycle. Feature access continues
  until then.

## Invoices

Invoices are available under **Settings → Billing → Invoices** and are emailed to the
billing contact. Invoice PDFs can be regenerated with an updated VAT/tax ID, a purchase
order number, or a corrected company address at any time by the billing admin.

Invoices cannot be re-issued to a different legal entity after payment. That requires a
credit note, which Support must raise.
