# Web Labeling Campus Invite Auth Direction

## Decision

For the near-term campus pilot, use campus/VPN firewall access plus signed per-user invite links instead of full SSO.

This keeps deployment lightweight while avoiding the weakest pattern: trusting only `expected_user=<id>` in the URL.

## Threat model

Campus network access is a coarse access gate. It establishes that a browser is coming from inside the campus/VPN boundary, but it does not identify the user to the web-labeling app.

Therefore, firewall-only access is not sufficient for personalized assignments. Without an identity signal or signed invite, one user could change a URL from:

```text
/my-datasets?expected_user=alice
```

to:

```text
/my-datasets?expected_user=bob
```

and attempt to view Bob's queue.

## Proposed pilot model

Use links like:

```text
/my-datasets?invite=<signed_token>
```

The signed token should resolve the labeler identity server-side. The browser should not be trusted to assert identity with a plain query parameter.

Suggested token claims:

```json
{
  "user": "campus_id",
  "scope": "personal_queue",
  "expires_at_utc": "2026-07-01T00:00:00Z",
  "nonce": "random-or-store-backed-id"
}
```

The server then uses the resolved token user to filter assignments and tasks.

## Authorization rules

- The assignment store remains authoritative for which recordings a user may access.
- Signed links are entry hints and identity resolvers, not write authority.
- Browser mutations must still require an active assignment for the resolved user.
- Browser mutations must still require a current task session and target token.
- The mutation target must still be resolved server-side from assignment/task scope.
- Users never receive direct Zarr write authority.
- Intermediate CSV or handoff files remain metadata/control-plane artifacts, not label-write targets.

## Access tiers

Recommended order:

1. SSO/proxy header identity for production or broad rollout.
2. Campus firewall plus signed per-user invite tokens for a trusted pilot.
3. Campus firewall plus plain `expected_user=<id>` only for local debugging.
4. Direct task links without identity/assignment checks are not recommended.

## Operational workflow

1. Operator assigns one recording to one campus user ID.
2. Operator creates tasks against explicit training-Zarr runs.
3. Operator generates a signed per-user invite link.
4. Labeler opens the invite link from campus/VPN network.
5. Server resolves the user from the signed invite.
6. Server displays only assigned recordings/tasks for that resolved user.
7. Server writes label mutations only to assigned task-scoped training Zarr targets.

## Implementation checklist

- Add invite-token creation CLI for one user or all assigned users.
- Add invite-token verification to `/my-datasets`, `/my-work`, and task-open flows.
- Resolve `user` from a valid invite before falling back to local/dev `--user` configuration.
- Reject expired, malformed, wrong-scope, or unknown-user invites.
- Preserve the existing `expected_user` guard as a debugging/support check, but do not treat it as identity.
- Include invite identity in authorization context diagnostics.
- Keep signed direct task links as entry hints only; task open must recheck resolved user and assignment ownership.
- Add operator-facing documentation for generating and sharing invite links.

## Open questions

- Should invite tokens be stateless HMAC tokens or store-backed one-time/nonced invites?
- What expiration period is appropriate for campus pilots?
- Should invite tokens be revocable individually, or is assignment deactivation enough for revocation?
- Should the invite URL include `expected_user` as a human-readable guard in addition to the token?

## Recommendation

Start with expiring signed HMAC invite tokens plus assignment-store checks. Add store-backed revocation only if pilots reveal a need to revoke links independently from deactivating assignments.

## Invite and session lifetime policy

Separate invite tokens from task session tokens.

### Invite token

The invite token is a medium-lived entry credential for a user's personal queue.

Recommended pilot policy:

- TTL: 7 days.
- Scope: personal queue/work entry for one resolved campus user ID.
- Expiration behavior: show a clear expired-link page and ask the user to request a refreshed link from the operator.
- Revocation behavior: assignment deactivation must block access even if the invite has not expired.

### Task session token

The task session token is a short-lived editing credential created when a user opens a task.

Recommended pilot policy:

- TTL: 4-8 hours.
- Scope: one task/open session and one server-resolved target token.
- Mutation behavior: saves require the current unexpired task session and current target token.
- Expiration behavior: save is rejected; user returns to the personal queue and reopens the task to create a fresh session.

### Refresh strategy

Start with operator refresh rather than automatic refresh.

Operator refresh means the operator regenerates handoff/invite links every few days or when a user reports an expired link. This is simpler, auditable, and safer for the initial campus pilot.

Do not implement rolling automatic refresh initially. Rolling refresh can be added later if link churn becomes operationally annoying.

Possible future refresh modes:

- Rolling refresh: if a valid invite is near expiry, the server returns a fresh invite for the same user.
- Store-backed refresh: invites carry a nonce stored in the sidecar DB so individual invite links can be rotated or revoked.

### Current recommendation

Use:

- 7-day signed invite tokens.
- 4-8 hour task sessions.
- Operator-generated refreshed links.
- Assignment ownership checks on every queue, task-open, and mutation request.
