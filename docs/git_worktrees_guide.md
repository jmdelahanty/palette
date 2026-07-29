# Git worktrees — a safe-use guide

Practical reference for using `git worktree` without stranding work. Written for this
repo's workflow (a solo maintainer running parallel agents that shouldn't collide on one
branch). Reference material, not a point-in-time diagnostic.

## Mental model (the whole thing)

A git repo has two separable parts people conflate:

1. **The repository** — the object database: every commit, branch, and blob. Lives in **one**
   place, the `.git` directory. There is only ever one.
2. **A working tree** — an actual directory of checked-out files on one branch, that you edit.

One repository can have **many** working trees attached, all sharing the same `.git` object
store. Each has its own branch checked out and its own copy of the files.

```
~/gitrepos/
├── palette/                 ← main worktree (branch: sun)
│   ├── .git/                ← THE repository: all objects, refs, history (one of these)
│   │   └── worktrees/
│   │       └── feature-x/   ← tiny bookkeeping for a linked tree (its HEAD, index)
│   └── src/ docs/ …         ← main's checked-out files
└── palette-feature-x/       ← linked worktree (branch: feature-x) — a SEPARATE folder
    ├── .git                 ← a one-line FILE: "gitdir: …/palette/.git/worktrees/feature-x"
    └── src/ docs/ …         ← its OWN second copy of the working files
```

- **Duplicated on disk:** the working files (each worktree gets its own copy).
- **Shared, not duplicated:** the `.git` object store (history). So N worktrees =
  1× history + N× working files — cheaper than N clones, but not free.
- The worktree path is **your choice** — it's the first argument to `git worktree add`,
  nothing automatic. Put it as a **sibling**, never nested inside the repo.

A commit made in any worktree lands in the shared `.git` immediately. **Committed work is
never lost by deleting a worktree directory** — only *uncommitted* edits live solely in the
folder.

## The safe lifecycle

```bash
git worktree add ../palette-x -b feature-x   # create: new dir + new branch, as a sibling
git worktree list                            # your map — run whenever unsure where you are
#   ...work and COMMIT inside ../palette-x (it's a normal repo on branch feature-x)...
git worktree remove ../palette-x             # tears down BOTH halves; refuses if uncommitted
git switch sun && git merge feature-x        # bring the work back when ready
git worktree prune                           # only if a worktree dir was deleted by hand
```

## Two rules that make orphans impossible

1. **Always tear down with `git worktree remove`, never `rm -rf`.** `remove` refuses if the
   tree has uncommitted/untracked files (`fatal: contains modified or untracked files`) and
   removes both the directory and git's bookkeeping together. Orphans only happen when the
   directory and the bookkeeping get separated — i.e. when you delete a worktree folder by
   hand.
2. **Commit before you remove.** Committed work is safe in the shared `.git`; uncommitted
   edits in a hand-deleted folder are the only thing genuinely at risk.

## "Where am I?" — your seatbelt

```bash
git worktree list                            # the truth about every working tree + branch
git rev-parse --git-dir --git-common-dir     # equal => main worktree; differ => linked
```

Visual tell: in the **main** worktree, `.git` is a **directory**; in a **linked** worktree
it's a one-line **file** pointing back to the main repo.

Built-in guard you can rely on: **a branch can be checked out in only one worktree at a
time.** Git refuses to check out a branch already live elsewhere — so two parallel streams
are forced onto separate branches and cannot stomp each other.

## Recovering an orphaned worktree

Symptom: a directory whose `.git` is a *file* pointing into `…/palette/.git/worktrees/<name>`,
but `git worktree list` does **not** show it. Git has forgotten it; the files remain.

```bash
# Is the work preserved as a branch in the shared repo?
git branch -a | grep <name>
# If yes: the commits are safe — just check out / merge that branch; rm the stale dir.
# If no:  any work exists ONLY as loose files in that folder. Inspect before deleting:
diff -rq <orphan-dir> .            # what's actually different from the live tree
# Decide keep-vs-delete from facts, then:
rm -rf <orphan-dir>                # safe: it's already disconnected from git
git worktree prune                 # clear any dangling bookkeeping
```

A separate full **clone** (its own `.git` *directory*, e.g. a `*.bak`) is self-contained —
deleting it with `rm -rf` cannot affect the main repo.

## Pattern for parallel agents

One worktree per agent stream, each on its own branch, merged back to `sun` when ready:

```bash
git worktree add ../palette-pipeline -b pipeline-stream
git worktree add ../palette-docs     -b docs-stream
```

No interleaving on a single branch; each agent is isolated to its own directory and branch.
(This repo's agent harness can also do this automatically via the Agent tool's
`isolation: "worktree"` option, which spins up a temp worktree and cleans it up afterward.)

## Commit-pinned cluster worktrees

The workstation's `/tmp` worktrees are not a stable cluster runtime surface.
When several agents need to submit LSF jobs concurrently, leave the primary
`/groups/.../gitrepos/palette` checkout on its existing branch and deploy each
clean committed agent worktree separately:

```bash
scripts/deploy_palette_cluster_worktree.sh \
  --source-repo /tmp/palette-my-feature
```

The helper:

1. requires a clean source branch;
2. pushes the exact source commit without updating the shared checkout;
3. fetches that branch into the shared Palette object store;
4. creates a detached worktree below the shared `palette-worktrees` root;
5. includes the short commit in the destination name;
6. locks the worktree against accidental Git pruning;
7. verifies that the shared checkout's branch and HEAD did not move;
8. verifies the deployed commit from `login1-citrus-poller` by default;
9. invokes that worktree's `scripts/py` and verifies that `fisheye` imports from
   the deployed `src` tree rather than another checkout's editable install.

The final output includes an exact value such as:

```text
palette_groups_repo_env=PALETTE_GROUPS_REPO=/groups/.../palette-worktrees/my-feature-0123abcd
```

Pass that absolute path to cluster submitters through `--palette-repo` or
`PALETTE_GROUPS_REPO`. Submitted jobs must also record the full commit and
refuse a dirty or mismatched checkout.

A new source commit gets a new deployment path. Do not fast-forward or repoint
an existing cluster worktree while jobs may still be using it. After their jobs
and retained evidence no longer need the checkout, unlock and remove old
deployments through Git rather than deleting their directories directly:

```bash
git -C /groups/.../gitrepos/palette worktree unlock /groups/.../<deployment>
git -C /groups/.../gitrepos/palette worktree remove /groups/.../<deployment>
```
