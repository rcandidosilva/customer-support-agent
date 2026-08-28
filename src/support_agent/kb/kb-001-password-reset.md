---
id: kb-001
title: Resetting your Northwind Analytics password
category: account
tags: [password, login, reset, mfa, sso]
audience: customer
last_reviewed: 2026-05-14
---

# Resetting your password

1. Go to **app.northwind-analytics.com/login** and choose **Forgot password**.
2. Enter the email address on your account. A reset link is sent within 2 minutes.
3. The link expires after **60 minutes**. Requesting a new link invalidates the previous one.
4. Choose a password of at least 12 characters. Reuse of your last 5 passwords is blocked.

## If the reset email never arrives

- Check spam, and allowlist `no-reply@northwind-analytics.com`.
- Reset emails are only sent to addresses that already exist on a workspace. For privacy
  reasons the "Forgot password" screen shows the same confirmation either way.
- Workspaces with **SSO enforcement** enabled cannot use password reset at all. Members must
  sign in through their identity provider; see [kb-002].

## Notes

- Resetting a password does **not** sign you out of active sessions. To end all sessions,
  use **Settings → Security → Sign out everywhere**.
- Password reset does not reset multi-factor enrollment. Lost MFA devices require an
  admin to reset enrollment; see [kb-003].
