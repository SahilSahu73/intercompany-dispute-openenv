FROM python:3.12-slim

WORKDIR /app

# Install uv and curl (for health check)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cached layer)
COPY pyproject.toml uv.lock ./
COPY openenv.yaml ./

# Install dependencies without project itself (better caching)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY . .

# Install project
RUN uv sync --frozen --no-dev

# Expose OpenEnv port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the environment server
CMD ["uv", "run", "server"]
