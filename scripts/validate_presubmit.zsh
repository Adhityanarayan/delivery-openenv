#!/usr/bin/env bash
#
# validate_presubmit.zsh — OpenEnv pre-submission checks (reference / official-style)
#
# Prerequisites:
#   - Docker:  https://docs.docker.com/get-docker/
#   - curl
#   - openenv CLI:  pip install openenv-core  (or uv sync)
#
# Note: Shebang is bash; safe to run as:  bash scripts/validate_presubmit.zsh ...
#       or:  chmod +x scripts/validate_presubmit.zsh && ./scripts/validate_presubmit.zsh ...
#
# Usage:
#   ./scripts/validate_presubmit.zsh <ping_url> [repo_dir]
#
#   ping_url   Your Space API base, e.g. https://adhiawesome-delivery-openenv.hf.space
#   repo_dir   Path to repo root (default: current directory)
#
# Examples:
#   ./scripts/validate_presubmit.zsh https://adhiawesome-delivery-openenv.hf.space
#   ./scripts/validate_presubmit.zsh https://adhiawesome-delivery-openenv.hf.space /path/to/OpenEnvHack
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   HF Space API base (e.g. https://your-name-your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: .)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} (GET /health, POST /reset) ..."

# Do not redirect stderr into the same file as -o (that corrupts the body / code).
HEALTH_CODE="000"
HEALTH_BODY=$(portable_mktemp "validate-health")
CLEANUP_FILES+=("$HEALTH_BODY")
if HEALTH_CODE=$(curl -sS -o "$HEALTH_BODY" -w "%{http_code}" "$PING_URL/health" --max-time 20); then
  :
else
  HEALTH_CODE="000"
fi
[ -z "$HEALTH_CODE" ] && HEALTH_CODE="000"

if [ "$HEALTH_CODE" != "200" ]; then
  fail "GET /health returned HTTP $HEALTH_CODE (expected 200)"
  hint "Open $PING_URL/docs in a browser; confirm the Space is Running."
  stop_at "Step 1"
fi

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE="000"
if HTTP_CODE=$(curl -sS -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 60); then
  :
else
  HTTP_CODE="000"
fi
[ -z "$HTTP_CODE" ] && HTTP_CODE="000"

if [ "$HTTP_CODE" = "200" ]; then
  if ! grep -q '"observation"' "$CURL_OUTPUT" 2>/dev/null; then
    fail "POST /reset returned 200 but JSON missing 'observation'"
    head -c 400 "$CURL_OUTPUT" | sed 's/^/  /' || true
    stop_at "Step 1"
  fi
elif [ "$HTTP_CODE" = "000" ]; then
  fail "POST /reset not reachable (connection failed or timed out)"
  hint "Try: curl -sS -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' $PING_URL/reset"
  stop_at "Step 1"
else
  fail "POST /reset returned HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

pass "HF Space live: GET /health -> 200, POST /reset -> 200 with observation"

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t delivery-openenv-validate "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install: pip install openenv-core   (or uv sync in this repo)"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate . 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && printf "%s\n" "$VALIDATE_OUTPUT" | sed 's/^/  /'
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  hint "Ensure pyproject.toml has [project.scripts] server = \"server.app:main\" and uv.lock exists (run: uv lock)."
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
