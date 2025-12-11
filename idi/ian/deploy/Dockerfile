# IAN Node Dockerfile
# Multi-stage build for minimal production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Copy source
COPY . .

# Build package
RUN pip wheel --no-cache-dir --wheel-dir=/wheels .


# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.12-slim as production

# Labels
LABEL org.opencontainers.image.title="IAN Node"
LABEL org.opencontainers.image.description="Intelligent Augmentation Network - Decentralized L2 Node"
LABEL org.opencontainers.image.version="1.0.0"

# Create non-root user
RUN groupadd -r ian && useradd -r -g ian ian

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy configuration
COPY deploy/config/ /app/config/

# Create directories
RUN mkdir -p /app/data /app/logs /app/keys && \
    chown -R ian:ian /app

# Switch to non-root user
USER ian

# Environment variables
ENV IAN_CONFIG_PATH=/app/config/production.yaml
ENV IAN_DATA_DIR=/app/data
ENV IAN_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Ports
# 8000 - REST API
# 9000 - P2P TCP
# 9001 - WebSocket
EXPOSE 8000 9000 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command
CMD ["python", "-m", "idi.ian.cli", "node", "start", "--config", "/app/config/production.yaml"]


# =============================================================================
# Stage 3: Development
# =============================================================================
FROM production as development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy \
    ipython

USER ian

ENV IAN_CONFIG_PATH=/app/config/development.yaml
ENV IAN_LOG_LEVEL=DEBUG

CMD ["python", "-m", "idi.ian.cli", "node", "start", "--config", "/app/config/development.yaml"]
