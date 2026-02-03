#!/usr/bin/env bash
set -euo pipefail

# Upload ONE file to GitHub via Contents API (HTTPS + PAT), without pushing the whole repo.
#
# Notes / limits:
# - This uses GitHub "Create or update file contents" API.
# - File size is limited (GitHub API has a small payload limit; large files won't work).
# - PAT must have repo write permission (e.g. "Contents: Read and write").
#
# Usage examples:
#   # upload local file README.md to repo root README.md
#   export GITHUB_TOKEN=...   # DO NOT paste token into chat
#   export GITHUB_USER=feng1201
#   ./scripts/upload_one_file_github_pat.sh \
#     --repo feng1201/laten_grpo4agent \
#     --local /finance_ML/fengninghui/LatentMAS/README.md \
#     --remote README.md \
#     --branch main \
#     --message "Update README"
#
# If GITHUB_USER/GITHUB_TOKEN are not set, the script will prompt.

REPO=""
LOCAL=""
REMOTE=""
BRANCH="main"
MESSAGE="Update file via API"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2;;
    --local) LOCAL="${2:-}"; shift 2;;
    --remote) REMOTE="${2:-}"; shift 2;;
    --branch) BRANCH="${2:-main}"; shift 2;;
    --message) MESSAGE="${2:-Update file via API}"; shift 2;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0;;
    *)
      echo "[error] Unknown arg: $1" >&2
      exit 2;;
  esac
done

if [[ -z "$REPO" || -z "$LOCAL" || -z "$REMOTE" ]]; then
  echo "[error] Missing required args: --repo OWNER/REPO --local /path/file --remote path/in/repo" >&2
  exit 2
fi

if [[ ! -f "$LOCAL" ]]; then
  echo "[error] Local file not found: $LOCAL" >&2
  exit 2
fi

GITHUB_USER="${GITHUB_USER:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ -z "$GITHUB_USER" ]]; then
  read -r -p "GitHub username: " GITHUB_USER
fi
if [[ -z "$GITHUB_TOKEN" ]]; then
  read -r -s -p "GitHub PAT (input hidden): " GITHUB_TOKEN
  echo
fi
if [[ -z "$GITHUB_USER" || -z "$GITHUB_TOKEN" ]]; then
  echo "[error] Missing GITHUB_USER or GITHUB_TOKEN" >&2
  exit 2
fi

OWNER="${REPO%%/*}"
REPO_NAME="${REPO#*/}"
API_BASE="https://api.github.com"
API_PATH="/repos/${OWNER}/${REPO_NAME}/contents/${REMOTE}"
API_URL="${API_BASE}${API_PATH}"

auth_header="Authorization: Bearer ${GITHUB_TOKEN}"
ua_header="User-Agent: upload_one_file_github_pat"

# Temp files (use mktemp to avoid bash $$ expansion issues and races)
GH_GET_FILE="$(mktemp)"
GH_PUT_FILE="$(mktemp)"
PAYLOAD_FILE="$(mktemp)"
cleanup() {
  rm -f "$GH_GET_FILE" "$GH_PUT_FILE" "$PAYLOAD_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# Base64-encode file content (single line, no wrapping)
content_b64="$(base64 -w 0 "$LOCAL" 2>/dev/null || base64 "$LOCAL" | tr -d '\n')"

# Check if file exists to get its SHA (required when updating)
existing_sha=""
http_code="$(
  curl -sS -o "$GH_GET_FILE" -w '%{http_code}' \
    -H "$auth_header" -H "$ua_header" \
    "${API_URL}?ref=${BRANCH}" || true
)"

if [[ "$http_code" == "200" ]]; then
  # Extract "sha" field from JSON without jq.
  existing_sha="$(python - "$GH_GET_FILE" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("sha", ""))
PY
  )"
elif [[ "$http_code" == "404" ]]; then
  existing_sha=""
else
  echo "[error] Failed to query existing file. HTTP ${http_code}" >&2
  sed -n '1,120p' "$GH_GET_FILE" >&2 || true
  exit 1
fi

# Build request JSON safely via python (avoid bash quoting/substitution issues).
# We pass values via environment variables to avoid brittle quoting.
export UPLOAD_MESSAGE="$MESSAGE"
export UPLOAD_CONTENT_B64="$content_b64"
export UPLOAD_BRANCH="$BRANCH"
export UPLOAD_USER="$GITHUB_USER"
export UPLOAD_SHA="$existing_sha"
export UPLOAD_PAYLOAD_FILE="$PAYLOAD_FILE"

python - <<'PY'
import json
import os

message = os.environ["UPLOAD_MESSAGE"]
content_b64 = os.environ["UPLOAD_CONTENT_B64"]
branch = os.environ["UPLOAD_BRANCH"]
user = os.environ["UPLOAD_USER"]
sha = os.environ.get("UPLOAD_SHA") or ""
payload_file = os.environ["UPLOAD_PAYLOAD_FILE"]

payload = {
    "message": message,
    "content": content_b64,
    "branch": branch,
    "committer": {"name": user, "email": f"{user}@users.noreply.github.com"},
}
if sha:
    payload["sha"] = sha

with open(payload_file, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

echo "[info] Uploading $LOCAL -> ${REPO}:${REMOTE} (branch=${BRANCH})"
if [[ -n "$existing_sha" ]]; then
  echo "[info] Mode: update (sha=${existing_sha})"
else
  echo "[info] Mode: create"
fi

put_code="$(
  curl -sS -o "$GH_PUT_FILE" -w '%{http_code}' \
    -X PUT \
    -H "$auth_header" -H "$ua_header" -H "Content-Type: application/json" \
    --data-binary "@${PAYLOAD_FILE}" \
    "${API_URL}" || true
)"

if [[ "$put_code" == "200" || "$put_code" == "201" ]]; then
  # Print a small success summary
  python - "$GH_PUT_FILE" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data=json.load(f)
commit=data.get("commit",{})
print("[ok] done")
print("commit:", commit.get("sha",""))
print("html_url:", data.get("content",{}).get("html_url",""))
PY
else
  echo "[error] Upload failed. HTTP ${put_code}" >&2
  sed -n '1,200p' "$GH_PUT_FILE" >&2 || true
  exit 1
fi


# cd /finance_ML/fengninghui/LatentMAS

# export GITHUB_USER=feng1201
# export GITHUB_TOKEN=g  # 不要贴到聊天里

# ./scripts/upload_one_file_github_pat.sh \
#   --repo feng1201/agent \
#   --local /finance_ML/fengninghui/LatentMAS/README.md \
#   --remote README.md \
#   --branch main \
#   --message "Update README"