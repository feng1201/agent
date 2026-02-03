#!/usr/bin/env bash
set -euo pipefail

# Push ONLY a specified folder (or file path) to a GitHub repo via git commit + HTTPS PAT push.
#
# Why this helps:
# - If your working tree contains huge model weights/checkpoints, don't stage them.
# - Only the paths you `git add` will be included in the commit, so push stays fast.
#
# Prereqs:
# - Use HTTPS + PAT push helper: ./scripts/push_github_pat.sh
# - Provide PAT via env (recommended):
#     export GITHUB_USER=feng1201
#     export GITHUB_TOKEN=...   # do NOT paste into chat
#
# Usage:
#   ./scripts/push_folder_pat.sh --repo feng1201/agent --path scripts --message "Update scripts"
#
# Notes:
# - If weight files were already tracked previously, they can still be pushed.
#   This script will warn if the specified path contains tracked weight-like files.

REPO=""
BRANCH="main"
REMOTE_NAME="feng1201"
TARGET_PATH=""
MESSAGE="Update subset"
PUSH_FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2;;
    --branch) BRANCH="${2:-main}"; shift 2;;
    --remote) REMOTE_NAME="${2:-feng1201}"; shift 2;;
    --path) TARGET_PATH="${2:-}"; shift 2;;
    --message) MESSAGE="${2:-Update subset}"; shift 2;;
    --force) PUSH_FORCE="true"; shift 1;;
    -h|--help)
      sed -n '1,180p' "$0"
      exit 0;;
    *)
      echo "[error] Unknown arg: $1" >&2
      exit 2;;
  esac
done

if [[ -z "$REPO" || -z "$TARGET_PATH" ]]; then
  echo "[error] Missing required args: --repo OWNER/REPO --path PATH" >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] Not inside a git repo. Run from repo root." >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -e "$TARGET_PATH" ]]; then
  echo "[error] Path not found: $TARGET_PATH" >&2
  exit 2
fi

echo "[info] Repo root: $REPO_ROOT"
echo "[info] Staging only: $TARGET_PATH"

# Warn if the target path contains *tracked* weight-like files (meaning they might already be in history).
tracked_weight_count="$(
  git ls-files -- "$TARGET_PATH" \
    | grep -E '\\.(safetensors|bin|pt|pth|ckpt|gguf|onnx)$' \
    | wc -l | tr -d ' '
)"
if [[ "$tracked_weight_count" != "0" ]]; then
  echo "[warn] Found $tracked_weight_count tracked weight-like files under $TARGET_PATH."
  echo "[warn] If these were committed before, they can still be pushed. Consider removing them from git history."
fi

# Stage only the requested subset (does NOT stage other folders).
git add -A -- "$TARGET_PATH"

echo "[info] git status after staging:"
git status -sb

# Commit if there is something staged
if git diff --cached --quiet; then
  echo "[info] No staged changes under $TARGET_PATH. Nothing to commit."
  exit 0
fi

git commit -m "$MESSAGE"

PUSH_ARGS=(--repo "$REPO" --branch "$BRANCH" --remote "$REMOTE_NAME")
if [[ "$PUSH_FORCE" == "true" ]]; then
  PUSH_ARGS+=(--force)
fi

# Delegate the actual push + PAT handling to the existing helper script.
./scripts/push_github_pat.sh "${PUSH_ARGS[@]}"


