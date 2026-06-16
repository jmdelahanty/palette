# Login-Node Palette Branch Divergence - 2026-06-16

## Purpose

Document the observed divergence between the workstation Palette checkout and
the Palette checkout visible from the Janelia login node. This is a handoff for
a follow-up agent to investigate whether useful work exists only on the login
node branch and whether any of it should be reconciled into the canonical
GitHub branch.

## Context

While wiring the Citrus staging poller to submit jobs through `login1`, the
workstation needed a repo-managed LSF submit wrapper:

```text
scripts/submit_citrus_session_import_bsub.sh
```

The wrapper was committed locally on the workstation and pushed to GitHub:

```text
493fc93 Add Citrus session LSF submit wrapper
```

The login-node checkout is at:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette
```

It is also on a branch named `sun`, but it is not fast-forwardable from
`origin/sun`.

## Observed State

Workstation checkout:

```bash
cd /home/delahantyj@hhmi.org/gitrepos/palette
git branch --show-current
git rev-parse --short HEAD
```

Observed:

```text
sun
493fc93
```

Login-node checkout:

```bash
ssh login1-citrus-poller \
  'cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
   git branch --show-current &&
   git rev-parse --short HEAD &&
   git status --short'
```

Observed:

```text
sun
7d011a8
?? scripts/submit_citrus_session_import_bsub.sh
```

The submit wrapper is untracked on the login-node checkout because a full
fast-forward pull was not safe. It was restored from `origin/sun` only for the
purpose of allowing the Citrus poller to call it.

## Pull Attempt

This command was attempted on the login-node checkout:

```bash
git pull --ff-only origin sun
```

It failed with:

```text
fatal: Not possible to fast-forward, aborting.
```

## Divergence Summary

The following comparison was run from the login-node checkout after fetching
`origin/sun`:

```bash
git rev-list --left-right --count 493fc93...HEAD
```

Observed:

```text
763 740
```

Interpretation:

- `763` commits are reachable from workstation/GitHub `493fc93` and not from
  login-node `HEAD`.
- `740` commits are reachable from login-node `HEAD` and not from
  workstation/GitHub `493fc93`.

This is a substantial branch divergence, not a small unpushed delta.

## Recent Login-Node-Only Commit Subjects

Command:

```bash
git log --oneline --right-only --cherry-pick 493fc93...HEAD --max-count=30
```

Observed recent subjects included:

```text
9d61d06 analysis: add body-frame analytics workflows
eb456d0 benchmarks: archive roi inference cache smoke reports
9966e01 Add detection profile registry sync and training data card aggregation
ab21872 Add eye-mask parity tooling, contracts, and provenance groundwork
b1faa18 more thorough documentation and automation of model generation
37d19ad Update training registry for detect and keypoints and scripting workflow
7e36d66 updated provenance and trianing
cb4f103 Expand retune/refinement tooling, zarr schema docs, and diagnostics
a09b144 updated exporting to tensorrt and registry for datasets and models
653eba6 eye angles now no longer use ferets
001012d eye mask creation and editing, audits for keypoints and eye masks manual version
7a00b67 fixed import and malformed dataset chaser states in h5
091f370 fixing alignment correctly I think
1f02b45 fixing bad interpolation method for chaser states
c48684c working stimulus interpolation chaser states
8003aff centroids computed at import for plotting in red, columnar format for interpolated/filtered datasets
54ac830 swim bouts and movement analysis aligned correctly it seems
86b19b2 eye angle smoothing applied
26861ff can't get events saved in a zarr friendly way, moving on...
7dc4795 added lots of refinements for metadata provenance etc in the library
1540b12 added visualization, imports for h5 data, and diagnostics
c7bdc9a refinement stages now unified with data provenance and attributes I think, ready to start making analysis scripts
0f079f0 refining eye segmentation methods
5cc45df eye mask training and validation seems to work, speedy with cache
a319317 can train eye mask segmentation, validation fails, pretty slow. Time to cache
ec52e49 keypoint visualization with new structure working
55eaca1 update for including benchmarks
5375c1d adding clean zarr v3 and refactoring organization of fisheye
02d1a6e adding corrected keypoint tracking
2f4d7b0 messy repo update for trying to get plots of speed/distance/etc by group...
```

## Changed-Path Summary

Command:

```bash
git diff --name-only 493fc93..HEAD |
  grep -v '^\.venv/' |
  awk -F/ 'NF==1{print $1; next} {print $1"/"$2}' |
  sort | uniq -c | sort -nr | head -40
```

Observed high-level summary:

```text
233 src/fisheye
 33 tests/unit
 17 runs/benchmarks
 10 docs/diagnostics
```

Other notable paths included tracked local artifacts/model files:

```text
yolo11*.pt
yolov8*.pt
gpu_results.json
keypoint_nudge_flags.json
keypoint_manual_list.txt
eye_mask_manual_list.txt
eye_mask_frame_flags.json
todo.txt
.venv/
```

Those paths should be treated with caution. Their presence suggests the
login-node history may contain local environment artifacts or generated outputs
that should not be blindly merged.

## Temporary Action Taken For Citrus Poller

To avoid rewriting or merging the divergent login-node checkout, only the new
submit wrapper was restored from `origin/sun`:

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git fetch origin sun &&
  git restore --source origin/sun -- scripts/submit_citrus_session_import_bsub.sh &&
  chmod 755 scripts/submit_citrus_session_import_bsub.sh &&
  bash -n scripts/submit_citrus_session_import_bsub.sh
'
```

This left the wrapper untracked in the login-node checkout:

```text
?? scripts/submit_citrus_session_import_bsub.sh
```

That was intentional: it avoided merging or rebasing the divergent `sun` branch
while allowing the workstation cron poller to call a repo-shaped wrapper path.

## Follow-Up Update After Workstation Push

After the workstation committed and pushed additional cleanup/hardening work,
the canonical remote moved from `493fc93` to:

```text
e3253b9 utils: fail closed on missing refined detect parent
```

The workstation checkout is now in sync with `origin/sun` at `e3253b9`.

The login-node checkout was fetched again:

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git fetch origin sun &&
  git branch -vv &&
  git rev-parse --short origin/sun &&
  git rev-list --left-right --count origin/sun...HEAD
'
```

Observed:

```text
* sun  7d011a8 [origin/sun: ahead 740, behind 768] review: shard proxy video generation
e3253b9
768 740
```

Interpretation:

- The login-node checkout is still not fast-forwardable.
- This is not just the recent Citrus wrapper being unpushed. It is a large
  historical divergence.
- The login-node branch appears to contain pre-cleanup or pre-history-rewrite
  history. Its diff against `origin/sun` includes tracked local artifacts such
  as `.venv/`, model weights, benchmark outputs, and JSON sidecars.
- Do not merge this branch back into `origin/sun`.

Current recommended reconciliation:

1. Treat GitHub `origin/sun` as canonical.
2. Preserve the login-node branch before changing it:
   create a backup branch or bundle named with the old HEAD, e.g.
   `backup/login-sun-20260616-7d011a8`.
3. Replace the login-node working checkout with a fresh clone of `origin/sun`,
   or move the current directory aside and clone fresh into the original path.
4. Keep the Citrus poller path stable:
   `/groups/johnson/johnsonlab/jeremy/gitrepos/palette`.
5. Only salvage individual files from the backup after an explicit path-level
   review. Do not salvage `.venv/`, model weights, generated plots, benchmark
   outputs, or JSON sidecars into source history.

## Investigation Questions

1. Why does the login-node branch named `sun` point at `7d011a8` while
   GitHub/workstation `sun` points at `493fc93`?
2. Are the login-node-only commits already present elsewhere under another
   branch name, tag, or historical remote?
3. Which login-node-only commits contain real work that should be preserved?
4. Which login-node-only paths are generated artifacts or local-environment
   mistakes that should not be migrated?
5. Is the login-node checkout still used for production LSF submissions? If so,
   should it be replaced by a fresh clone of the canonical GitHub branch after
   salvage is complete?

## Suggested Read-Only Commands For Follow-Up Agent

Run these first. Do not reset, merge, rebase, or clean until the useful content
has been inventoried.

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git remote -v &&
  git branch -vv &&
  git status --short &&
  git log --oneline --decorate --graph --all --max-count=80
'
```

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git for-each-ref --format="%(refname:short) %(objectname:short) %(committerdate:iso8601) %(subject)" refs/heads refs/remotes |
  sort
'
```

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git log --oneline --right-only --cherry-pick origin/sun...HEAD --max-count=200
'
```

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  git diff --stat origin/sun..HEAD &&
  git diff --name-status origin/sun..HEAD
'
```

## Safety Notes

- Do not run `git reset --hard`, `git clean`, `git pull`, `git merge`, or
  `git rebase` on the login-node checkout until the branch contents have been
  classified.
- Do not assume tracked `.venv`, model weights, plots, or JSON sidecars are
  intended source assets.
- Keep the Citrus poller dependency in mind: the workstation poller currently
  calls
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette/scripts/submit_citrus_session_import_bsub.sh`
  on `login1-citrus-poller`.
