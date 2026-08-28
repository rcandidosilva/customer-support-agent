---
id: kb-003
title: Lost MFA device and recovery codes
category: account
tags: [mfa, 2fa, totp, recovery, locked-out]
audience: customer
last_reviewed: 2026-04-21
---

# Lost MFA device

## If you have recovery codes

Enter one of your 10 single-use recovery codes at the MFA prompt, then re-enroll a new
device immediately under **Settings → Security → Two-factor authentication**.

## If you do not have recovery codes

A **workspace admin** must reset your MFA enrollment: **Settings → Members → (member) →
Reset two-factor**. The member receives an email and must re-enroll at next sign-in.

## If you are the only admin and are locked out

Northwind Support can reset MFA for a sole admin, but only after identity verification:

- The request must come from the email address on the account.
- Support verifies the last 4 digits of the billing card **or** a recent invoice number.
- The reset is applied within one business day. This cannot be expedited and cannot be
  performed over chat.

Support agents cannot see or set passwords, and cannot bypass MFA for non-admin members.
