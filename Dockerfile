# Grounded QA Service API image. Built by `docker compose up`; see docker-compose.yml.
# Base images are pinned by digest; Python packages come from uv.lock; Hugging Face models are
# fetched at their pinned revisions at build time, so the container runs offline.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/opt/hf
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY gqa/__init__.py gqa/hf_models.py gqa/
RUN HF_HUB_OFFLINE=0 /opt/venv/bin/python -c "from gqa.hf_models import fetch_pinned, require_pins; fetch_pinned(); require_pins()"

FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
RUN useradd --create-home --uid 1000 gqa
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=gqa:gqa /opt/hf /opt/hf
WORKDIR /app
COPY gqa ./gqa
COPY config ./config
COPY data/corpus ./data/corpus
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    GQA_MODEL_BASE_URL=http://ollama:11434/v1
USER gqa
EXPOSE 8000
CMD ["uvicorn", "gqa.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
