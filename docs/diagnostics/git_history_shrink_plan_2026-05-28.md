# Git History Shrink Plan

Last verified: 2026-05-28

## Status

The safe repository hygiene cleanup has already landed at `HEAD`: local
artifacts are no longer tracked, and `.gitignore` blocks them from being added
again. That cleanup does not shrink `.git`, because the old blobs remain in
history.

Current packed Git storage:

```text
size-pack: 955.97 MiB
```

Current tracked files matching the cleanup artifact categories:

```text
0
```

## Rewrite Strategy

Use an explicit path-list rewrite, not a broad file-extension rewrite. The
reviewed path list is:

```text
docs/diagnostics/git_history_cleanup_paths_2026-05-28.txt
```

The list contains 41 explicit paths/path-prefixes. In the current history it
matches 731 historical blob paths, totaling about 450.3 MiB of uncompressed blob
payload.

The list intentionally targets generated/local/runtime artifacts:

- Local environments and dependency artifacts: `.venv/`, `decord`, Decord wheel.
- Generated run output trees: `runs/`, `src/runs/`, plot/output directories.
- Root model/checkpoint/engine/zip artifacts.
- Local manual-review flags and diagnostic outputs.

The list intentionally does not include large notebook or fixture-like files
found in history, such as `keypoints_to_boundingbox.ipynb`,
`raw_video_to_boundingbox.ipynb`, or `test_frames/*.jpg`. Those can be reviewed
as a separate second pass if needed.

## Read-Only Audit Commands

List largest historical blobs:

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1 == "blob" {path=$0; sub(/^[^ ]+ [^ ]+ [^ ]+ /, "", path); print $3 "\t" path}' \
  | sort -nr \
  | head -80
```

Check current storage:

```bash
git count-objects -vH
```

Confirm the target paths are no longer tracked at `HEAD`:

```bash
git ls-files .venv decord '*.pt' '*.whl' runs src/runs alignment_diagnostics detection_plots \
  src/fisheye/analysis/plots src/fisheye/visualization/plots \
  eye_mask_frame_flags.json keypoint_nudge_flags.json gpu_results.json \
  eye_mask_manual_list.txt keypoint_manual_list.txt todo.txt
```

## Disposable-Clone Rewrite Procedure

Do not run this in the active working clone first. Use a fresh mirror clone:

```bash
cd /tmp
git clone --mirror git@github.com:jmdelahanty/palette.git palette-history-cleanup.git
cd palette-history-cleanup.git
git filter-repo \
  --paths-from-file /home/delahantyj@hhmi.org/gitrepos/palette/docs/diagnostics/git_history_cleanup_paths_2026-05-28.txt \
  --invert-paths
git count-objects -vH
```

If `git-filter-repo` is not installed, install it outside this repository or use
a managed environment that already provides it. Do not add it as a Palette
dependency.

## Verification Before Force-Push

After the rewrite in the disposable clone:

1. Verify the target paths are absent from all history:

   ```bash
   git rev-list --objects --all \
     | rg -f /home/delahantyj@hhmi.org/gitrepos/palette/docs/diagnostics/git_history_cleanup_paths_2026-05-28.txt
   ```

   This should print no matches, except for path-list entries that are directory
   prefixes and require manual prefix checks.

2. Verify expected source/docs still exist at rewritten `HEAD`:

   ```bash
   git ls-tree -r --name-only HEAD | rg '^(src/|tests/|docs/|configs/|scripts/)'
   ```

3. Make a normal clone from the rewritten mirror and run static validation:

   ```bash
   git clone /tmp/palette-history-cleanup.git /tmp/palette-history-cleanup-work
   cd /tmp/palette-history-cleanup-work
   git status --short
   git diff --check
   scripts/py -m py_compile src/fisheye/shared/zarr_discovery.py
   ```

4. Only after review, push with lease from the rewritten mirror:

   ```bash
   git push --force-with-lease origin --all
   git push --force-with-lease origin --tags
   ```

## Collaboration Risk

This operation changes commit hashes for all rewritten refs. After the force
push, collaborators should make a new clone or hard-reset local branches to the
new history. Any unmerged local work must be rebased carefully.

Do not run the rewrite while cluster jobs or agent branches are expected to push
back into the old history.
