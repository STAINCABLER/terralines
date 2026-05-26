#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/run_locust_headless.sh [BASE_URL]
BASE_URL=${1:-http://127.0.0.1:8000}
USERS=${2:-50}
SPAWN_RATE=${3:-10}
RUN_TIME=${4:-1m}

echo "Running locust headless against $BASE_URL (users=$USERS spawn_rate=$SPAWN_RATE duration=$RUN_TIME)"
python -m pip install --quiet locust
locust -f tests/locustfile.py --headless -u "$USERS" -r "$SPAWN_RATE" -t "$RUN_TIME" --host "$BASE_URL" --csv=artifacts/locust_report || true
echo "Locust finished. Reports in artifacts/locust_report*"
