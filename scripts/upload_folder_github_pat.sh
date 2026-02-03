#!/usr/bin/env bash
set -euo pipefail

# Upload a WHOLE folder to GitHub via Contents API (HTTPS + PAT),
# with flexible mapping from local_dir -> remote_dir (repo path).
#
# This is the right tool when:
# - You don't want to git-push the whole repo (e.g., repo contains huge weights).
# - You want to map a local folder to an arbitrary folder path in the target repo.
#
# It uploads files one-by-one by calling `upload_one_file_github_pat.sh`.
# Note: this can be slow if the folder contains many files (API calls per file).
#
# Usage:
#   export GITHUB_USER=feng1201
#   export GITHUB_TOKEN=...   # DO NOT paste token into chat
#   ./scripts/upload_folder_github_pat.sh \
#     --repo feng1201/agent \
#     --local_dir /finance_ML/fengninghui/LatentMAS/scripts \
#     --remote_dir scripts \
#     --branch main \
#     --message_prefix "Sync scripts"
#
# Default excludes: common weight/checkpoint extensions.

REPO=""
LOCAL_DIR=""
REMOTE_DIR=""
BRANCH="main"
MESSAGE_PREFIX="Sync folder"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2;;
    --local_dir) LOCAL_DIR="${2:-}"; shift 2;;
    --remote_dir) REMOTE_DIR="${2:-}"; shift 2;;
    --branch) BRANCH="${2:-main}"; shift 2;;
    --message_prefix) MESSAGE_PREFIX="${2:-Sync folder}"; shift 2;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0;;
    *)
      echo "[error] Unknown arg: $1" >&2
      exit 2;;
  esac
done

if [[ -z "$REPO" || -z "$LOCAL_DIR" ]]; then
  echo "[error] Missing required args: --repo OWNER/REPO --local_dir /path/to/dir" >&2
  exit 2
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "[error] Local dir not found: $LOCAL_DIR" >&2
  exit 2
fi

# Normalize remote dir (no leading slash; no trailing slash except empty)
REMOTE_DIR="${REMOTE_DIR#/}"
REMOTE_DIR="${REMOTE_DIR%/}"

# Ensure credentials exist once (child script will prompt otherwise, but that would be painful in a loop)
if [[ -z "${GITHUB_USER:-}" ]]; then
  read -r -p "GitHub username: " GITHUB_USER
  export GITHUB_USER
fi
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  read -r -s -p "GitHub PAT (input hidden): " GITHUB_TOKEN
  echo
  export GITHUB_TOKEN
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOAD_ONE="$SCRIPT_DIR/upload_one_file_github_pat.sh"
if [[ ! -x "$UPLOAD_ONE" ]]; then
  echo "[error] Missing helper script: $UPLOAD_ONE" >&2
  exit 2
fi

echo "[info] Repo: $REPO"
echo "[info] Local dir: $LOCAL_DIR"
echo "[info] Remote dir: ${REMOTE_DIR:-.} (repo root if empty)"
echo "[info] Branch: $BRANCH"

# Exclude weight/checkpoint-like files by default
EXCLUDE_REGEX='\\.(safetensors|bin|pt|pth|ckpt|gguf|onnx)$'

mapfile -t FILES < <(find "$LOCAL_DIR" -type f | sort)
total="${#FILES[@]}"
echo "[info] Found $total files (before exclude)"

uploaded=0
skipped=0
failed=0

for f in "${FILES[@]}"; do
  rel="${f#"$LOCAL_DIR"/}"
  # Skip excluded extensions
  if [[ "$f" =~ $EXCLUDE_REGEX ]]; then
    skipped=$((skipped+1))
    continue
  fi

  if [[ -z "$REMOTE_DIR" ]]; then
    remote_path="$rel"
  else
    remote_path="$REMOTE_DIR/$rel"
  fi

  msg="$MESSAGE_PREFIX: $remote_path"

  echo "[upload] $f -> $remote_path"
  if "$UPLOAD_ONE" --repo "$REPO" --local "$f" --remote "$remote_path" --branch "$BRANCH" --message "$msg"; then
    uploaded=$((uploaded+1))
  else
    failed=$((failed+1))
    echo "[error] Failed uploading: $f" >&2
    # continue uploading other files
  fi
done

echo "[done] uploaded=$uploaded skipped=$skipped failed=$failed"
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi


