#!/usr/bin/env bash
set -euo pipefail

# scripts/run_e2e_local.sh
# Lokales End-to-End Testscript
# - baut Images (docker compose build)
# - startet Compose (docker compose up -d)
# - wartet auf services (web, redis)
# - enqueued einen RQ-Job und pollt mit exponential backoff
# - lädt result.png herunter, speichert artifacts und compose-logs
# - fährt Compose wieder runter

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACTS_DIR="$ROOT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo "[e2e] Building images (docker compose build)"
docker compose build --pull

echo "[e2e] Starting services (docker compose up -d)"
docker compose up -d

# helper: wait for HTTP endpoint
wait_for_http() {
  local url="$1"
  local retries=${2:-30}
  local delay=${3:-2}
  for i in $(seq 1 "$retries"); do
    if curl -sS --fail "$url" >/dev/null 2>&1; then
      echo "[e2e] $url is ready"
      return 0
    fi
    echo "[e2e] waiting for $url... ($i/$retries)"
    sleep "$delay"
  done
  echo "[e2e] timeout waiting for $url"
  return 1
}

# wait for web
wait_for_http "http://localhost:8000/" 60 2

# wait for redis (try connecting via docker to container named redis, fallback to sleep)
if docker ps --format '{{.Names}}' | grep -q '^redis$'; then
  echo "[e2e] checking redis container readiness"
  for i in $(seq 1 30); do
    if docker exec redis redis-cli ping >/dev/null 2>&1; then
      echo "[e2e] redis ready"
      break
    fi
    echo "[e2e] waiting for redis... ($i/30)"
    sleep 2
  done
else
  echo "[e2e] redis container not found by name 'redis' — continuing after short wait"
  sleep 2
fi

# ensure jq exists
if ! command -v jq >/dev/null 2>&1; then
  echo "[e2e] jq not found. On Debian/Ubuntu you can 'sudo apt install jq'."
  echo "[e2e] Continuing but JSON parsing failures may occur."
fi

# enqueue RQ job
echo "[e2e] enqueueing RQ job"
RESP=$(curl -sS -X POST -H "Content-Type: application/json" -d '{"params":{}}' http://localhost:8000/api/generate_rq || true)
JOB_ID=""
if command -v jq >/dev/null 2>&1; then
  JOB_ID=$(echo "$RESP" | jq -r '.job_id // empty')
else
  # crude extraction if jq not present
  JOB_ID=$(echo "$RESP" | sed -n 's/.*\"job_id\"[: "]*\([^",}]*\).*/\1/p')
fi

if [ -z "$JOB_ID" ]; then
  echo "[e2e] Failed to enqueue job: $RESP"
  docker compose logs > "$ARTIFACTS_DIR/compose.log" || true
  exit 1
fi

echo "[e2e] enqueued job: $JOB_ID"

# poll with exponential backoff (max attempts ~12)
attempt=0
delay=1
STATUS=""
while [ $attempt -lt 12 ]; do
  STATUS=$(curl -sS "http://localhost:8000/api/rq/job/${JOB_ID}" | (command -v jq >/dev/null 2>&1 && jq -r '.status // empty' || sed -n 's/.*\"status\"[: "]*\([^",}]*\).*/\1/p')) || true
  echo "[e2e] status=$STATUS (attempt=$attempt)"
  if [ "$STATUS" = "finished" ]; then
    echo "[e2e] job finished"
    break
  fi
  attempt=$((attempt+1))
  sleep $delay
  delay=$((delay*2))
done

if [ "$STATUS" != "finished" ]; then
  echo "[e2e] job did not finish in time"
  docker compose logs > "$ARTIFACTS_DIR/compose.log" || true
  exit 2
fi

# download result
echo "[e2e] downloading result"
if curl -sS -o "$ARTIFACTS_DIR/result.png" "http://localhost:8000/api/job/${JOB_ID}/download"; then
  if [ ! -s "$ARTIFACTS_DIR/result.png" ]; then
    echo "[e2e] downloaded file is empty"
    docker compose logs > "$ARTIFACTS_DIR/compose.log" || true
    exit 3
  fi
  echo "[e2e] result saved to $ARTIFACTS_DIR/result.png"
else
  echo "[e2e] failed to download result"
  docker compose logs > "$ARTIFACTS_DIR/compose.log" || true
  exit 4
fi

# collect logs and teardown
docker compose logs > "$ARTIFACTS_DIR/compose.log" || true

echo "[e2e] tearing down"
docker compose down -v || true

echo "[e2e] done. Artifacts in $ARTIFACTS_DIR"
