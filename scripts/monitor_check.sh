#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/monitor_check.sh [BASE_URL]
BASE_URL=${1:-http://127.0.0.1:8000}
echo "Running synthetic checks against $BASE_URL"
curl -fsS --max-time 15 "$BASE_URL/" -o /tmp/monitor_root.html
curl -fsS --max-time 10 "$BASE_URL/health" -o /tmp/monitor_health.txt
if ! grep -q '"status"[[:space:]]*:[[:space:]]*"ok"' /tmp/monitor_health.txt; then
	echo "health endpoint did not report status=ok"
	cat /tmp/monitor_health.txt
	exit 1
fi
echo "root size: $(wc -c < /tmp/monitor_root.html)"
echo "health:"
cat /tmp/monitor_health.txt
echo "Artifacts saved to /tmp/monitor_*"
