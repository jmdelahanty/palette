#!/usr/bin/env bash
set -euo pipefail
umask 0002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_REPO="${PALETTE_WORKSTATION_REPO:-$REPO_ROOT}"
GROUPS_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
DEPLOY_ROOT="${PALETTE_GROUPS_WORKTREE_ROOT:-}"
DESTINATION=""
BRANCH=""
REMOTE="${PALETTE_GIT_REMOTE:-origin}"
SSH_KEY="${PALETTE_GIT_SSH_KEY:-/home/delahantyj@hhmi.org/.ssh/delahantyj-ws1-git-id_ed25519}"
VERIFY_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOCK_TIMEOUT_SECONDS=120
SKIP_PUSH=0
SKIP_HOST_VERIFY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_palette_cluster_worktree.sh [options]

Push one clean committed Palette worktree without changing the shared /groups
checkout, then create a detached, commit-pinned /groups worktree for LSF jobs.
The destination is idempotent for the same commit and is locked against Git
worktree pruning.

Options:
  --source-repo PATH       Workstation worktree to deploy
                           (default: this script's repository)
  --groups-repo PATH       Shared Palette repository/object store
                           (default: $PALETTE_GROUPS_REPO or the standard path)
  --deploy-root PATH       Parent for commit-pinned cluster worktrees
                           (default: <groups-repo-parent>/palette-worktrees)
  --destination PATH       Exact destination below --deploy-root
                           (default: <deploy-root>/<branch-leaf>-<short-commit>)
  --branch NAME            Branch to push/fetch (default: source branch)
  --remote NAME            Git remote in both repositories (default: origin)
  --ssh-key PATH           Workstation SSH key used for push
  --verify-host HOST       Confirm the worktree from this host after deployment
                           (default: login1-citrus-poller)
  --skip-push              Fetch an already-pushed branch without pushing
  --skip-host-verify       Do not SSH to the verification host
  --lock-timeout SECONDS   Wait for another deployment helper (default: 120)
  --dry-run                Validate and print actions without mutation
  -h, --help               Show this help

The source may be a primary checkout or a linked Git worktree. It must be clean.
The shared checkout's branch and HEAD are verified unchanged after deployment.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

canonical_existing_path() {
  local path="$1"
  realpath -- "$path"
}

canonical_future_path() {
  local path="$1"
  realpath -m -- "$path"
}

resolve_git_path() {
  local base="$1"
  local value="$2"
  if [[ "$value" == /* ]]; then
    canonical_future_path "$value"
  else
    canonical_future_path "$base/$value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-repo)
      [[ $# -ge 2 ]] || fail "--source-repo requires a path"
      SOURCE_REPO="$2"
      shift 2
      ;;
    --groups-repo)
      [[ $# -ge 2 ]] || fail "--groups-repo requires a path"
      GROUPS_REPO="$2"
      shift 2
      ;;
    --deploy-root)
      [[ $# -ge 2 ]] || fail "--deploy-root requires a path"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --destination)
      [[ $# -ge 2 ]] || fail "--destination requires a path"
      DESTINATION="$2"
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
    --verify-host)
      [[ $# -ge 2 ]] || fail "--verify-host requires a host"
      VERIFY_HOST="$2"
      shift 2
      ;;
    --skip-push)
      SKIP_PUSH=1
      shift
      ;;
    --skip-host-verify)
      SKIP_HOST_VERIFY=1
      shift
      ;;
    --lock-timeout)
      [[ $# -ge 2 ]] || fail "--lock-timeout requires seconds"
      LOCK_TIMEOUT_SECONDS="$2"
      shift 2
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
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "$LOCK_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  fail "--lock-timeout must be a positive integer"
[[ "$REMOTE" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || \
  fail "--remote must be a safe Git remote name"
if [[ "$SKIP_HOST_VERIFY" -eq 0 ]]; then
  [[ "$VERIFY_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9._@-]*$ ]] || \
    fail "--verify-host must be a safe SSH host"
fi
command -v flock >/dev/null 2>&1 || fail "flock is required"
command -v realpath >/dev/null 2>&1 || fail "realpath is required"

SOURCE_TOP="$(git -C "$SOURCE_REPO" rev-parse --show-toplevel 2>/dev/null)" || \
  fail "source is not a Git worktree: $SOURCE_REPO"
SOURCE_REPO="$(canonical_existing_path "$SOURCE_TOP")"
GROUPS_TOP="$(git -C "$GROUPS_REPO" rev-parse --show-toplevel 2>/dev/null)" || \
  fail "shared path is not a Git worktree: $GROUPS_REPO"
GROUPS_REPO="$(canonical_existing_path "$GROUPS_TOP")"
[[ "$SOURCE_REPO" != "$GROUPS_REPO" ]] || \
  fail "source and shared repositories must be different worktrees"
git -C "$SOURCE_REPO" remote get-url "$REMOTE" >/dev/null 2>&1 || \
  fail "source repository has no remote named '$REMOTE'"
git -C "$GROUPS_REPO" remote get-url "$REMOTE" >/dev/null 2>&1 || \
  fail "shared repository has no remote named '$REMOTE'"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$SOURCE_REPO" symbolic-ref --quiet --short HEAD)" || \
    fail "source is detached; check out the branch before deployment"
fi
git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 || \
  fail "invalid branch name: $BRANCH"

SOURCE_BRANCH="$(git -C "$SOURCE_REPO" symbolic-ref --quiet --short HEAD)" || \
  fail "source worktree must be on a branch"
[[ "$SOURCE_BRANCH" == "$BRANCH" ]] || \
  fail "source is on '$SOURCE_BRANCH', not requested branch '$BRANCH'"

SOURCE_STATUS="$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf '%s\n' "$SOURCE_STATUS" >&2
  fail "source worktree must be clean before cluster deployment"
fi

COMMIT="$(git -C "$SOURCE_REPO" rev-parse HEAD)"
SHORT_COMMIT="$(git -C "$SOURCE_REPO" rev-parse --short=8 HEAD)"
BRANCH_LEAF="${BRANCH##*/}"
BRANCH_SLUG="$(
  printf '%s' "$BRANCH_LEAF" \
    | tr -c 'A-Za-z0-9._-' '-' \
    | sed -e 's/^-*//' -e 's/-*$//'
)"
[[ -n "$BRANCH_SLUG" ]] || fail "branch does not produce a safe deployment slug"

if [[ -z "$DEPLOY_ROOT" ]]; then
  DEPLOY_ROOT="$(dirname -- "$GROUPS_REPO")/palette-worktrees"
fi
DEPLOY_ROOT="$(canonical_future_path "$DEPLOY_ROOT")"
case "$DEPLOY_ROOT/" in
  "$GROUPS_REPO/"*) fail "deploy root cannot be inside the shared checkout" ;;
esac

if [[ -z "$DESTINATION" ]]; then
  DESTINATION="$DEPLOY_ROOT/${BRANCH_SLUG}-${SHORT_COMMIT}"
fi
DESTINATION="$(canonical_future_path "$DESTINATION")"
[[ "$DESTINATION" != "$DEPLOY_ROOT" ]] || \
  fail "destination must be below the deploy root"
case "$DESTINATION/" in
  "$DEPLOY_ROOT/"*) ;;
  *) fail "destination must be below deploy root $DEPLOY_ROOT" ;;
esac
[[ "$DESTINATION" != "$GROUPS_REPO" ]] || \
  fail "destination cannot replace the shared checkout"

if [[ "$SKIP_PUSH" -eq 0 && ! -f "$SSH_KEY" ]]; then
  fail "SSH key not found: $SSH_KEY"
fi
if [[ "$SKIP_HOST_VERIFY" -eq 0 && -z "$VERIFY_HOST" ]]; then
  fail "verification host cannot be empty unless --skip-host-verify is used"
fi
printf -v GIT_SSH_COMMAND_VALUE 'ssh -i %q -o IdentitiesOnly=yes' "$SSH_KEY"

GROUPS_BRANCH_BEFORE="$(git -C "$GROUPS_REPO" rev-parse --abbrev-ref HEAD)"
GROUPS_HEAD_BEFORE="$(git -C "$GROUPS_REPO" rev-parse HEAD)"
GROUPS_COMMON_RAW="$(git -C "$GROUPS_REPO" rev-parse --git-common-dir)"
GROUPS_COMMON_DIR="$(resolve_git_path "$GROUPS_REPO" "$GROUPS_COMMON_RAW")"

printf 'status=%s\n' "$([[ "$DRY_RUN" -eq 1 ]] && printf planned || printf deploying)"
printf 'source_repo=%s\n' "$SOURCE_REPO"
printf 'source_branch=%s\n' "$BRANCH"
printf 'palette_commit=%s\n' "$COMMIT"
printf 'groups_repo=%s\n' "$GROUPS_REPO"
printf 'groups_branch=%s\n' "$GROUPS_BRANCH_BEFORE"
printf 'groups_head=%s\n' "$GROUPS_HEAD_BEFORE"
printf 'deploy_root=%s\n' "$DEPLOY_ROOT"
printf 'palette_repo=%s\n' "$DESTINATION"
printf 'verify_host=%s\n' "$([[ "$SKIP_HOST_VERIFY" -eq 1 ]] && printf skipped || printf '%s' "$VERIFY_HOST")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$SKIP_PUSH" -eq 0 ]]; then
    print_command env \
      "GIT_SSH_COMMAND=$GIT_SSH_COMMAND_VALUE" \
      git -C "$SOURCE_REPO" push "$REMOTE" \
      "$COMMIT:refs/heads/$BRANCH"
  fi
  print_command git -C "$GROUPS_REPO" fetch "$REMOTE" "refs/heads/$BRANCH"
  print_command mkdir -p "$DEPLOY_ROOT"
  print_command git -C "$GROUPS_REPO" worktree add --detach \
    "$DESTINATION" "$COMMIT"
  print_command git -C "$GROUPS_REPO" worktree lock --reason \
    "Palette cluster deployment $BRANCH@$COMMIT" "$DESTINATION"
  if [[ "$SKIP_HOST_VERIFY" -eq 0 ]]; then
    print_command ssh -o BatchMode=yes "$VERIFY_HOST" \
      "verify $DESTINATION at $COMMIT"
  fi
  printf 'status=planned\n'
  exit 0
fi

if [[ "$SKIP_PUSH" -eq 0 ]]; then
  print_command env \
    "GIT_SSH_COMMAND=$GIT_SSH_COMMAND_VALUE" \
    git -C "$SOURCE_REPO" push "$REMOTE" \
    "$COMMIT:refs/heads/$BRANCH"
  GIT_SSH_COMMAND="$GIT_SSH_COMMAND_VALUE" \
    git -C "$SOURCE_REPO" push "$REMOTE" \
    "$COMMIT:refs/heads/$BRANCH"
fi

LOCK_PATH="$GROUPS_COMMON_DIR/palette-cluster-worktree-deploy.lock"
exec 9>"$LOCK_PATH"
flock -w "$LOCK_TIMEOUT_SECONDS" 9 || \
  fail "timed out waiting for cluster deployment lock: $LOCK_PATH"

run git -C "$GROUPS_REPO" fetch "$REMOTE" "refs/heads/$BRANCH"
FETCHED_COMMIT="$(git -C "$GROUPS_REPO" rev-parse FETCH_HEAD)"
[[ "$FETCHED_COMMIT" == "$COMMIT" ]] || \
  fail "fetched branch resolved to $FETCHED_COMMIT, expected $COMMIT"

DEPLOYMENT_STATUS="deployed"
if [[ -e "$DESTINATION" ]]; then
  EXISTING_TOP="$(git -C "$DESTINATION" rev-parse --show-toplevel 2>/dev/null)" || \
    fail "destination exists but is not a Git worktree: $DESTINATION"
  EXISTING_TOP="$(canonical_existing_path "$EXISTING_TOP")"
  [[ "$EXISTING_TOP" == "$DESTINATION" ]] || \
    fail "destination resolves to a different worktree root: $EXISTING_TOP"
  EXISTING_COMMON_RAW="$(git -C "$DESTINATION" rev-parse --git-common-dir)"
  EXISTING_COMMON="$(resolve_git_path "$DESTINATION" "$EXISTING_COMMON_RAW")"
  [[ "$EXISTING_COMMON" == "$GROUPS_COMMON_DIR" ]] || \
    fail "destination belongs to another Git repository"
  EXISTING_COMMIT="$(git -C "$DESTINATION" rev-parse HEAD)"
  [[ "$EXISTING_COMMIT" == "$COMMIT" ]] || \
    fail "destination is at $EXISTING_COMMIT, expected $COMMIT"
  EXISTING_STATUS="$(git -C "$DESTINATION" status --porcelain --untracked-files=all)"
  [[ -z "$EXISTING_STATUS" ]] || \
    fail "existing deployment worktree is dirty: $DESTINATION"
  DEPLOYMENT_STATUS="already_deployed"
else
  run mkdir -p "$DEPLOY_ROOT"
  run git -C "$GROUPS_REPO" worktree add --detach "$DESTINATION" "$COMMIT"
fi

DEPLOY_GIT_RAW="$(git -C "$DESTINATION" rev-parse --git-dir)"
DEPLOY_GIT_DIR="$(resolve_git_path "$DESTINATION" "$DEPLOY_GIT_RAW")"
if [[ ! -f "$DEPLOY_GIT_DIR/locked" ]]; then
  run git -C "$GROUPS_REPO" worktree lock --reason \
    "Palette cluster deployment $BRANCH@$COMMIT" "$DESTINATION"
fi

DEPLOY_COMMIT="$(git -C "$DESTINATION" rev-parse HEAD)"
DEPLOY_STATUS="$(git -C "$DESTINATION" status --porcelain --untracked-files=all)"
[[ "$DEPLOY_COMMIT" == "$COMMIT" ]] || fail "deployed commit changed unexpectedly"
[[ -z "$DEPLOY_STATUS" ]] || fail "deployed worktree is unexpectedly dirty"
[[ -x "$DESTINATION/scripts/py" ]] || \
  fail "deployed worktree lacks executable scripts/py"

GROUPS_BRANCH_AFTER="$(git -C "$GROUPS_REPO" rev-parse --abbrev-ref HEAD)"
GROUPS_HEAD_AFTER="$(git -C "$GROUPS_REPO" rev-parse HEAD)"
[[ "$GROUPS_BRANCH_AFTER" == "$GROUPS_BRANCH_BEFORE" ]] || \
  fail "shared checkout branch changed during deployment"
[[ "$GROUPS_HEAD_AFTER" == "$GROUPS_HEAD_BEFORE" ]] || \
  fail "shared checkout HEAD changed during deployment"

if [[ "$SKIP_HOST_VERIFY" -eq 0 ]]; then
  printf -v Q_DESTINATION '%q' "$DESTINATION"
  printf -v Q_COMMIT '%q' "$COMMIT"
  REMOTE_COMMAND="repo=$Q_DESTINATION; expected=$Q_COMMIT; "
  REMOTE_COMMAND+='test -x "$repo/scripts/py" && '
  REMOTE_COMMAND+='actual=$(git -C "$repo" rev-parse HEAD) && '
  REMOTE_COMMAND+='test "$actual" = "$expected" && '
  REMOTE_COMMAND+='test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"'
  run ssh -o BatchMode=yes "$VERIFY_HOST" "$REMOTE_COMMAND"
fi

printf 'status=%s\n' "$DEPLOYMENT_STATUS"
printf 'palette_repo=%s\n' "$DESTINATION"
printf 'palette_commit=%s\n' "$COMMIT"
printf 'groups_checkout_unchanged=true\n'
printf 'worktree_locked=true\n'
printf 'palette_groups_repo_env=PALETTE_GROUPS_REPO=%q\n' "$DESTINATION"
