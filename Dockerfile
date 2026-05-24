FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ENV HOST=0.0.0.0 \
    PORT=8000

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py generator.py index.html ./
COPY static ./static
COPY template ./template

EXPOSE 8000

USER appuser

# Run gunicorn with debug logging and capture worker output so CI can surface exceptions
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--capture-output", "--log-level", "debug", "--bind", "0.0.0.0:8000", "app:app"]

# Simple healthcheck using Python (no extra packages needed)
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import sys,urllib.request as r
try:
        r.urlopen('http://127.0.0.1:8000/', timeout=2)
        sys.exit(0)
except:
        sys.exit(1)"