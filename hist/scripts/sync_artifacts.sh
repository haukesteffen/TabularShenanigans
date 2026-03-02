#!/usr/bin/env bash
set -euo pipefail

# Example pull from remote runner into local artifacts directory.
# Usage:
#   scripts/sync_artifacts.sh user@host:/path/to/project/artifacts artifacts

REMOTE_PATH="${1:-}"
LOCAL_PATH="${2:-artifacts}"

if [[ -z "$REMOTE_PATH" ]]; then
  echo "Usage: $0 <remote_path> [local_path]"
  exit 1
fi

rsync -avz "$REMOTE_PATH" "$LOCAL_PATH"
