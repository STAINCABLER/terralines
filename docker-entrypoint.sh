#!/bin/sh
set -eu

REDIS_URL="${REDIS_URL:-redis://redis:6379/0}"

# Generate a 64-char secret key if not provided
if [ -z "${TERRALINES_SECRET_KEY:-}" ]; then
  TERRALINES_SECRET_KEY=$(python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)
  export TERRALINES_SECRET_KEY
  echo "[entrypoint] Generated TERRALINES_SECRET_KEY (auto)" >&2
fi
wait_for_redis() {
  timeout="${TERRALINES_REDIS_WAIT_SECONDS:-30}"
  python - "$REDIS_URL" "$timeout" <<'PY'
import sys
import time

from redis import Redis

redis_url = sys.argv[1]
timeout = int(sys.argv[2])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        Redis.from_url(redis_url).ping()
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1)

print(f"Redis not ready after {timeout}s: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

case "${TERRALINES_WORKER:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    wait_for_redis
    exec rq worker terralines --url "$REDIS_URL"
    ;;
esac

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec gunicorn --workers 1 --threads 2 --timeout 120 --capture-output --log-level info --bind 0.0.0.0:8000 app:app