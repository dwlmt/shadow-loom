# syntax=docker/dockerfile:1.7
# =====================================================================
# Shadow-Loom — production image
# =====================================================================
# Suitable for Railway, Fly.io, Cloud Run, or any PaaS that injects
# ``$PORT`` and provides a managed Postgres ``$DATABASE_URL``.
#
# Build:   docker build -t shadow-loom .
# Run:     docker run -e DATABASE_URL=... -e STORAGE_SECRET=... -p 8080:8080 shadow-loom
# =====================================================================

FROM python:3.13-slim

# ── System deps ──────────────────────────────────────────────────
# build-essential: needed only for any C extensions that wheel-fail.
# libpq5: psycopg[binary] ships its own libpq, but keeping the runtime
#         lib makes `psql` debugging available.
# curl:   used for the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (1000 keeps HF Spaces compatibility too) ───────
RUN useradd -m -u 1000 app
WORKDIR /app

# ── Project metadata + source ────────────────────────────────────
COPY pyproject.toml ./
COPY README.md ./
COPY shadow_loom/         shadow_loom/
COPY shadow_loom_ui/      shadow_loom_ui/
COPY shadow_loom_mcp/     shadow_loom_mcp/
COPY example_worlds/      example_worlds/
COPY sample_plots/        sample_plots/

# ── Install ──────────────────────────────────────────────────────
# Set INSTALL_EXTRAS at build time to add optional dependency groups,
# e.g.:  docker build --build-arg INSTALL_EXTRAS=research -t shadow-loom .
# Available extras: research (Tavily provider). See pyproject.toml.
ARG INSTALL_EXTRAS=""
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -n "$INSTALL_EXTRAS" ]; then \
        pip install --no-cache-dir ".[$INSTALL_EXTRAS]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# ── Runtime ──────────────────────────────────────────────────────
USER app

# Railway / Fly / Cloud Run inject ``$PORT``. Fall back to 7860 for
# local ``docker run`` parity with Hugging Face Spaces.
ENV PORT=7860
EXPOSE 7860

# These are *defaults* — every deployable knob comes from the
# platform's environment (Railway "Variables" tab, Fly secrets, etc.).
# Never bake secrets into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UI_HOST=0.0.0.0 \
    PIPELINE_USE_CAUSAL_ENGINE=true \
    PIPELINE_SKIP_AUDIT=false

# Healthcheck — Railway uses this for zero-downtime swaps.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/" >/dev/null || exit 1

CMD ["python", "-m", "shadow_loom_ui"]
