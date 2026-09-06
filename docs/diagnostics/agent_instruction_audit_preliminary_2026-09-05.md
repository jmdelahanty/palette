# Agent-instruction audit: Palette and Codex configuration

> Superseded for audit scope and recommendations on 2026-09-06 by the
> [completed Palette instruction audit](agent_instruction_audit_2026-09-05.md),
> now included in this uncommitted landing draft. The user confirmed Palette
> only; the broader-root question below is no longer pending. The original
> config-change record is retained as history, not an action in this pass.
> Follow-up status: [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

Date: 2026-09-05. Preliminary, read-only instruction review; no skills or
existing AGENTS.md rules were rewritten as a result of this audit.

## Scope and source limits

The requested `~/Projects` directory does not exist on this workstation.
`~/gitrepos` does exist, but a question about substituting that broader root
is outstanding. This is **not** an audit of every repository or installed skill.

Inspected the original Palette root AGENTS.md in full and the isolated
main-based draft. An explicit hidden/ignored-file search, excluding nested
`.claude/worktrees`, Git internals, node_modules, and virtualenv caches, found
one source-owned AGENTS.md and no SKILL.md. It also exposed two installed
marimo AGENTS.md copies under `.pixi/envs`; those third-party environment files
were not audited or changed. Global/plugin skill bodies were not inventoried;
the visible session catalog alone is insufficient to claim that audit.

Read Eric Provencher's article text, [Rethinking skills and prompts for GPT-6
Astra](https://x.com/pvncher/status/2095991462416490862), through the public
[FxTwitter response](https://api.fxtwitter.com/pvncher/status/2095991462416490862)
after X returned 403. The response identifies article
`2095989703967125509`, published 2026-09-04. Embedded example images were not
inspected. Its relevant advice: narrow skill triggers, route task-specific
details, avoid mandatory broad reading for small changes, and define safe
completion boundaries. This is useful review guidance, not authority to relax
data-integrity or security requirements.

## Ranked findings

| Rank | Evidence in original Palette AGENTS.md | Finding and recommendation |
|---|---|---|
| 1 | Git Push Rule, line 44; worktree ownership, lines 76–93 | The push-only example hardcodes the original checkout in `git -C`. Copied into another worktree, it targets that checkout's branch instead of the caller's. Scope the example explicitly or use a helper that verifies the intended worktree, branch, and exact commit. Preserve the SSH-key requirement and separate permission to push. No push was performed. |
| 2 | Collaboration preservation/testing instructions, lines 95–152 | Several requirements read as unconditional for any change: preservation tests before implementation, broad negative cases, benchmarks, and public-path integration exercises. Keep these for the relevant contract/behavior changes, but explicitly scope them so a typo or evidence-only document update does not trigger the entire program. Never turn proportional local validation into a waiver of required CI. |
| 3 | Sandbox fallback, lines 234–237; test policy, lines 259–272; outside-sandbox notes, lines 302–312 | The execution-environment guidance is spread across three sections. Consolidate common routing and keep the actual exceptions: pytest outside the sandbox on an allowed workstation, no campus-login-node/LSF tests, explicit mutable/published metadata modes, metadata-file fallback, and GPU/marimo limitations. These solve real environment failures; deleting them as generic scaffolding would be wrong. |
| 4 | Full root file: 318 lines / 21,678 bytes; draft: 324 lines / 22,050 bytes | The root is substantial but below the default 32 KiB combined project-instruction cap. There is no evidence here that it was truncated. Keep a short always-applicable safety/contract core and conditional links for detailed implementation playbooks; do not move critical constraints into optional reading without a reliable trigger. Counts alone do not prove a rule is unnecessary. |
| 5 | Authority roles, lines 179–232; consolidation and mask rules, lines 239–257 and 282–300 | Keep these domain-specific constraints. A more capable model still cannot infer Palette's accepted supplier roles, canonical/authoritative distinction, immutable-publication policy, or dense-mask editing authority. Shorten duplicate wording only after traceable equivalence review. |

Line numbers refer to the original uncommitted root inspected during this
audit. The draft preserves six newer main lines that prohibit pytest on campus
login nodes and LSF test submissions, shifting later anchors. The current
collaboration section is also an audit target: having drafted it does not
exempt it from scoping and simplification.

## Codex context-management setting

The installed CLI reports `codex-cli 0.153.3` and ChatGPT sign-in. Official
[configuration documentation](https://learn.chatgpt.com/docs/config-file/config-reference)
documents `features.context_management.experimental_mode` as an off-by-default
experiment using notes and searchable history instead of repeatedly reducing
context to one summary. It requires ChatGPT sign-in on Plus, Pro, or Pro Lite.
The subscription tier was not inspected.

Enabled the user's exact setting in `~/.codex/config.toml` after a byte-for-byte
backup to `~/.codex/config.toml.before-context-management-20260905-f976e668.bak`.
Both files retain mode 0600. TOML parsing and an exact-byte comparison establish
that the only insertion was:

```toml
[features.context_management]
experimental_mode = true
```

`codex features list` now reports context_management true and under development.
That verifies local configuration, not account entitlement or activation in
this already-running session. Use a fresh session to exercise it. Reverting
means removing only this inserted table after checking for subsequent edits;
do not blindly restore the entire backup over later settings changes.

This option does not enlarge authorization, enforce repository contracts, or
make an oversized skill catalog harmless. Official [skill guidance](https://learn.chatgpt.com/docs/build-skills)
describes initial metadata loading and on-use instructions; the [AGENTS.md
guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md) documents
instruction discovery and size limits.

## Remaining audit, pending root confirmation

Inventory source-owned AGENTS.md, AGENTS.override.md, and skill roots under the
approved projects directory. Distinguish local skills from installed vendor
copies and duplicate worktrees. Read every in-scope instruction body before
claiming a complete audit. Evaluate trigger overlap, contradictory scope,
stale paths, mandatory reading, safe local completion boundaries, and retained
domain contracts. Record per-file keep/scope/consolidate recommendations before
requesting authorization for instruction edits. No new fix queue is created.
