 
FROM python:3.12-slim AS builder
 
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
 
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md /build/
RUN uv lock --check --project /build
RUN uv export --frozen --no-dev --no-editable --no-emit-project --format requirements-txt -o requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir=/wheels --require-hashes -r requirements.txt

COPY idi /build/idi
RUN pip wheel --no-cache-dir --wheel-dir=/wheels --no-deps .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

 
WORKDIR /app
 
RUN apt-get update \
    && apt-get install -y --no-install-recommends libffi8 \
    && rm -rf /var/lib/apt/lists/*
 
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels idi \
    && rm -rf /wheels \
    && pip cache purge
 
ENTRYPOINT ["python", "-m", "idi.ian.cli"]

