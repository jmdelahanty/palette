#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/push_and_update_groups_checkout.sh [options]

Push the current Palette branch from the workstation checkout, then fast-forward
the shared /groups checkout to the pushed branch.

Options:
  --repo PATH             Workstation repository path
                          (default: $PALETTE_WORKSTATION_REPO or
                           /home/delahantyj@hhmi.org/gitrepos/palette)
  --groups-repo PATH      Shared cluster/login-node repository path
                          (default: $PALETTE_GROUPS_REPO or
                           /groups/johnson/johnsonlab/jeremy/gitrepos/palette)
  --branch NAME           Branch to push/update (default: current branch in --repo)
  --remote NAME           Git remote (default: $PALETTE_GIT_REMOTE or origin)
  --ssh-key PATH          SSH key for GitHub pushes/fetches
                          (default: $PALETTE_GIT_SSH_KEY or Palette workstation key)
  --require-clean         Fail if the workstation checkout has uncommitted changes
  --skip-push             Do not push; only fast-forward the shared checkout
  --skip-groups-update    Push only; do not update the shared checkout
  --dry-run               Print actions without running git push/fetch/merge
  -h, --help              Show this help

Environment:
  PALETTE_WORKSTATION_REPO
  PALETTE_GROUPS_REPO
  PALETTE_GIT_REMOTE
  PALETTE_GIT_SSH_KEY

Notes:
  - The /groups checkout must already be on the target branch.
  - The /groups checkout must have no staged or tracked working-tree changes.
  - The update is a fetch followed by merge --ff-only; non-fast-forward state
    is treated as an error.
EOF
}

log() {
  printf '%s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

REPO="${PALETTE_WORKSTATION_REPO:-/home/delahantyj@hhmi.org/gitrepos/palette}"
GROUPS_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
REMOTE="${PALETTE_GIT_REMOTE:-origin}"
SSH_KEY="${PALETTE_GIT_SSH_KEY:-/home/delahantyj@hhmi.org/.ssh/delahantyj-ws1-git-id_ed25519}"
BRANCH=""
REQUIRE_CLEAN=0
SKIP_PUSH=0
SKIP_GROUPS_UPDATE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || fail "--repo requires a path"
      REPO="$2"
      shift 2
      ;;
    --groups-repo)
      [[ $# -ge 2 ]] || fail "--groups-repo requires a path"
      GROUPS_REPO="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || fail "--branch requires a name"
      BRANCH="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || fail "--remote requires a name"
      REMOTE="$2"
      shift 2
      ;;
    --ssh-key)
      [[ $# -ge 2 ]] || fail "--ssh-key requires a path"
      SSH_KEY="$2"
      shift 2
      ;;
    --require-clean)
      REQUIRE_CLEAN=1
      shift
      ;;
    --skip-push)
      SKIP_PUSH=1
      shift
      ;;
    --skip-groups-update)
      SKIP_GROUPS_UPDATE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -d "$REPO/.git" ]] || fail "Not a git repository: $REPO"
[[ -f "$SSH_KEY" ]] || fail "SSH key not found: $SSH_KEY"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
fi
[[ "$BRANCH" != "HEAD" ]] || fail "Detached HEAD in workstation repo; pass --branch explicitly."

WORKTREE_STATUS="$(git -C "$REPO" status --short)"
if [[ -n "$WORKTREE_STATUS" ]]; then
  log "warning: workstation checkout has uncommitted changes; only committed changes will be pushed."
  if [[ "$REQUIRE_CLEAN" -eq 1 ]]; then
    printf '%s\n' "$WORKTREE_STATUS" >&2
    fail "Workstation checkout is dirty and --require-clean was set."
  fi
fi

LOCAL_HEAD="$(git -C "$REPO" rev-parse --short HEAD)"
log "workstation_repo=$REPO"
log "groups_repo=$GROUPS_REPO"
log "remote=$REMOTE"
log "branch=$BRANCH"
log "local_head=$LOCAL_HEAD"

export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes"

if [[ "$SKIP_PUSH" -eq 0 ]]; then
  run git -C "$REPO" push "$REMOTE" "HEAD:$BRANCH"
fi

if [[ "$SKIP_GROUPS_UPDATE" -eq 0 ]]; then
  [[ -d "$GROUPS_REPO/.git" ]] || fail "Not a git repository: $GROUPS_REPO"

  GROUPS_BRANCH="$(git -C "$GROUPS_REPO" rev-parse --abbrev-ref HEAD)"
  [[ "$GROUPS_BRANCH" == "$BRANCH" ]] || {
    fail "Shared checkout is on branch '$GROUPS_BRANCH', expected '$BRANCH'. Switch it manually first."
  }

  git -C "$GROUPS_REPO" diff --quiet || fail "Shared checkout has unstaged tracked changes: $GROUPS_REPO"
  git -C "$GROUPS_REPO" diff --cached --quiet || fail "Shared checkout has staged changes: $GROUPS_REPO"

  GROUPS_BEFORE="$(git -C "$GROUPS_REPO" rev-parse --short HEAD)"
  run git -C "$GROUPS_REPO" fetch "$REMOTE" "$BRANCH"
  run git -C "$GROUPS_REPO" merge --ff-only FETCH_HEAD
  log "groups_before=$GROUPS_BEFORE"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    GROUPS_AFTER="$(git -C "$GROUPS_REPO" rev-parse --short HEAD)"
    log "groups_after=$GROUPS_AFTER"
  else
    log "dry_run: shared checkout not updated"
  fi
fi
