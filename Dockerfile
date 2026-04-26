FROM python:3.13-slim

# HF Spaces requires user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY shadow_loom/ shadow_loom/
COPY shadow_loom_ui/ shadow_loom_ui/
COPY shadow_loom_mcp/ shadow_loom_mcp/
COPY sample_plots/ sample_plots/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Switch to non-root user
USER user

# NiceGUI serves on port 7860 (HF Spaces default)
EXPOSE 7860

ENV STORAGE_SECRET="shadow-loom-hf-spaces"
ENV DATABASE_URL="sqlite:////tmp/shadow_loom.db"

CMD ["python", "-m", "shadow_loom_ui.app"]
