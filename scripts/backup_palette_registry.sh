#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/backup_palette_registry.sh [options]

Create a verified SQLite backup of the Palette registry.

Options:
  --registry PATH       Registry SQLite path
                        (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --backup-dir DIR      Directory for backup files
                        (default: $PALETTE_REGISTRY_BACKUP_DIR or
                         /groups/ahrens/ahrenslab/jeremy/zebrobot/backups)
  --days-to-keep N      Delete palette_registry_*.sqlite backups older than N days
                        after a successful backup
                        (default: $PALETTE_REGISTRY_BACKUP_DAYS_TO_KEEP or 7)
  -h, --help            Show this help

Cron example:
  0 2 * * * cd /home/delahantyj@hhmi.org/gitrepos/palette && scripts/backup_palette_registry.sh >> /home/delahantyj@hhmi.org/palette_registry_backup.log 2>&1
EOF
}

log() {
  printf '%s: %s\n' "$(date)" "$*"
}

fail() {
  log "ERROR - $*" >&2
  exit 1
}

validate_registry() {
  local label="$1"
  local path="$2"
  local integrity
  local foreign_keys
  integrity="$(sqlite3 -readonly "$path" "PRAGMA integrity_check;" 2>&1)" || {
    fail "$label integrity_check failed: $integrity"
  }
  if [[ "$integrity" != "ok" ]]; then
    fail "$label integrity_check returned: $integrity"
  fi
  foreign_keys="$(sqlite3 -readonly "$path" "PRAGMA foreign_key_check;" 2>&1)" || {
    fail "$label foreign_key_check failed: $foreign_keys"
  }
  if [[ -n "$foreign_keys" ]]; then
    fail "$label foreign_key_check returned: $foreign_keys"
  fi
}

REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
BACKUP_DIR="${PALETTE_REGISTRY_BACKUP_DIR:-/groups/ahrens/ahrenslab/jeremy/zebrobot/backups}"
DAYS_TO_KEEP="${PALETTE_REGISTRY_BACKUP_DAYS_TO_KEEP:-7}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --days-to-keep)
      DAYS_TO_KEEP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$DAYS_TO_KEEP" =~ ^[0-9]+$ ]]; then
  fail "--days-to-keep must be a non-negative integer."
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  fail "sqlite3 is required."
fi

if [[ ! -f "$REGISTRY" ]]; then
  fail "Registry not found: $REGISTRY"
fi

if [[ ! -s "$REGISTRY" ]]; then
  fail "Registry is empty: $REGISTRY"
fi

validate_registry "Source registry" "$REGISTRY"

mkdir -p "$BACKUP_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="$BACKUP_DIR/palette_registry_${TIMESTAMP}.sqlite"
TMP_BACKUP="${BACKUP_FILE}.tmp.$$"

cleanup_tmp() {
  rm -f "$TMP_BACKUP"
}
trap cleanup_tmp EXIT

log "Starting Palette registry backup."
log "REGISTRY=$REGISTRY"
log "BACKUP_FILE=$BACKUP_FILE"

if ! sqlite3 "$REGISTRY" ".backup '$TMP_BACKUP'"; then
  fail "sqlite3 backup command failed."
fi

if [[ ! -s "$TMP_BACKUP" ]]; then
  fail "Backup file is missing or empty: $TMP_BACKUP"
fi

validate_registry "Backup" "$TMP_BACKUP"

mv "$TMP_BACKUP" "$BACKUP_FILE"
trap - EXIT

BACKUP_SIZE="$(du -h "$BACKUP_FILE" | cut -f1)"
log "Backup successful - $BACKUP_FILE ($BACKUP_SIZE) via sqlite3 backup; full integrity and foreign keys verified"

DELETED_COUNT=0
while IFS= read -r old_backup; do
  rm -f "$old_backup"
  DELETED_COUNT=$((DELETED_COUNT + 1))
done < <(find "$BACKUP_DIR" -name "palette_registry_*.sqlite" -type f -mtime +"$DAYS_TO_KEEP" -print)

if [[ "$DELETED_COUNT" -gt 0 ]]; then
  log "Removed $DELETED_COUNT old backup(s)."
fi

log "Current backups:"
ls -lh "$BACKUP_DIR"/palette_registry_*.sqlite 2>/dev/null | tail -5 || true
