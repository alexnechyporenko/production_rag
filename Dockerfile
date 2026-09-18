# syntax=docker/dockerfile:1.7
# Production RAG — application container
#
# Multi-stage build: the final image only contains the Python runtime + the
# app source + production deps. No compilers, no test tools, no caches.
#
# Build context = project root (docker-compose.yml sets `context: .`).

ARG PYTHON_VERSION=3.12

# ----------------------------------------------------------------------------
# Stage 1: builder
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build-time deps for asyncpg + pypdf native bits.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project so we can `pip install .` and let pyproject.toml
# be the single source of truth for production dependencies.
COPY pyproject.toml ./
COPY app/ ./app/
COPY benchmarks/ ./benchmarks/
COPY datasets/ ./datasets/
COPY scripts/ ./scripts/

# Install the package (and its declared production deps) into /install so the
# runtime stage can copy just the installed site-packages.
RUN pip install --upgrade pip \
    && pip install --prefix=/install .

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1

# Runtime libs for asyncpg + pdf parsing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed deps from builder.
COPY --from=builder /install /usr/local

# Copy application source.
WORKDIR /app
COPY app/ ./app/
COPY benchmarks/ ./benchmarks/
COPY datasets/ ./datasets/
COPY scripts/ ./scripts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run with uvicorn. For production, set LLM_PROVIDER=openai, EMBEDDING_PROVIDER=openai,
# and DATABASE_URL=postgresql+asyncpg://... pointing at the postgres container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
