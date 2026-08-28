---
id: kb-002
title: Configuring SAML SSO
category: account
tags: [sso, saml, okta, azure-ad, provisioning, enforcement]
audience: admin
last_reviewed: 2026-06-02
---

# Configuring SAML SSO

SAML 2.0 single sign-on is available on the **Business** and **Enterprise** plans. It is not
available on Starter or Team.

## Setup

1. **Settings → Security → Single sign-on → Configure**.
2. Copy the ACS URL and Entity ID into your identity provider (Okta, Entra ID, and Google
   Workspace are documented; any SAML 2.0 IdP works).
3. Paste the IdP metadata XML back into Northwind and select **Verify**.
4. Verification requires at least one successful test login before SSO can be enforced.

## Enforcement

Turning on **Require SSO for all members** disables password login for everyone in the
workspace, including existing sessions, within 5 minutes.

- Service accounts and API tokens are **not** affected by SSO enforcement.
- At least one break-glass admin can be exempted under **Security → SSO exemptions**. We
  strongly recommend configuring one before enforcing.

## SCIM provisioning

SCIM 2.0 user provisioning and deprovisioning is **Enterprise only**. Deprovisioning a user
through SCIM suspends the account and revokes tokens but retains their saved dashboards for
30 days.
