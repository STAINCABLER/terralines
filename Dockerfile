#
# BUILDER IMAGE
#
FROM python:3.13-slim-trixie AS terralines-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Set environment variables for build-time configuration
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TERRALINES_RESULTS_DIR=/tmp/terralines_results
ENV PYTHONPATH=/app

# Refresh Debian packages in the build stage so the runtime image inherits current security fixes.
RUN apt-get update \
    && apt-get -y full-upgrade \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv and install dependencies in the builder image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application sources into builder, then precompile them and drop the
# source files so the runtime image only receives bytecode plus static assets.
COPY ./app/app.py ./app/generator.py ./app/job_queue.py ./app/worker.py ./app/tasks.py ./app/index.html ./
COPY ./app/static ./static
COPY ./app/templates ./templates
RUN python -m compileall -q -b /app \
    && find /app -type f -name '*.py' -delete


#
# RUNTIME IMAGE
#
# Hardened runtime switch: move runtime to the newer Debian trixie variant.
FROM python:3.13-slim-trixie AS terralines-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Use the same MPL config dir and results dir
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TERRALINES_RESULTS_DIR=/tmp/terralines_results
ENV HOST=0.0.0.0 \
    PORT=8000

# Refresh Debian packages in the runtime stage to reduce base-image CVEs.
RUN apt-get update \
    && apt-get -y full-upgrade \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Copy venv from builder
COPY --from=terralines-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the compiled runtime artefacts and static files.
COPY --from=terralines-builder /app /app
COPY ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
WORKDIR /app

# Set environment variables for runtime configuration
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TERRALINES_RESULTS_DIR=/tmp/terralines_results
ENV PYTHONPATH=/app

ENV TERRALINES_JOB_WORKERS=1
ENV TERRALINES_MAX_CONCURRENCY=1
ENV TERRALINES_RATE_LIMIT_WINDOW_SECONDS=60
ENV TERRALINES_RATE_LIMIT_MAX_REQUESTS=30
ENV TERRALINES_TRUSTED_PROXIES="127.0.0.1,::1"

# Create results dir with correct ownership
RUN mkdir -p ${TERRALINES_RESULTS_DIR} && chown 10001:10001 ${TERRALINES_RESULTS_DIR}

EXPOSE 8000

USER appuser

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD /bin/sh -c 'if [ "${TERRALINES_WORKER:-false}" = "true" ]; then exit 0; fi; python -c "import json, urllib.request, sys;\ntry:\n    body = urllib.request.urlopen(\"http://127.0.0.1:8000/health\", timeout=2).read().decode(\"utf-8\")\n    data = json.loads(body)\n    sys.exit(0 if data.get(\"status\") == \"ok\" else 1)\nexcept Exception:\n    sys.exit(1)"'