# Use a lightweight Python base image for the runtime artifact.
FROM python:3.11-slim AS runtime-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt fastapi uvicorn pydantic

COPY ./hierarchical_lm /app/hierarchical_lm
COPY ./deployment/app.py /app/app.py

LABEL org.opencontainers.image.source="https://github.com/CoreyLeath-code/-Hierarchical-Language-Model" \
      org.opencontainers.image.description="Hierarchical Language Model research and inference container" \
      org.opencontainers.image.licenses="MIT"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
