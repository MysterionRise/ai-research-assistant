# ============================================================================
# ARIA Research Assistant - Dockerfile
# ============================================================================
# Multi-stage build for optimized production image
# ============================================================================

# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && \
    pip install build && \
    pip install .

# =========================
# Stage 2: Production
# =========================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="ARIA Team <aria-team@company.com>" \
    version="0.1.0" \
    description="ARIA - AI Research Intelligence Assistant"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    APP_HOME=/app \
    APP_USER=aria \
    APP_GROUP=aria

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For health checks
    curl \
    # For PDF processing
    poppler-utils \
    # For PostgreSQL
    libpq5 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 ${APP_GROUP} \
    && useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code and build files
COPY --chown=${APP_USER}:${APP_GROUP} pyproject.toml README.md ./
COPY --chown=${APP_USER}:${APP_GROUP} src/ ./src/
COPY --chown=${APP_USER}:${APP_GROUP} alembic/ ./alembic/
COPY --chown=${APP_USER}:${APP_GROUP} alembic.ini ./

# Create necessary directories
RUN mkdir -p /data/aria /var/log/aria \
    && chown -R ${APP_USER}:${APP_GROUP} /data/aria /var/log/aria

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "aria.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =========================
# Stage 3: Development
# =========================
FROM production AS development

# Switch to root for installing dev dependencies
USER root

# Install development dependencies
RUN pip install ".[dev]"

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER ${APP_USER}

# Override command for development
CMD ["uvicorn", "aria.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
