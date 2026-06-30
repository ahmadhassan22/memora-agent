FROM python:3.11-slim

WORKDIR /app

# Build tools for any compiled dependencies (chromadb pulls in a few,
# e.g. hnswlib) that may not have prebuilt wheels for every platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first — better Docker layer caching, so
# rebuilds after a code change don't re-download every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code only — the Streamlit UI runs locally, pointed at this
# container's public address (see deploy/deploy_notes.md).
COPY app/ ./app/

# ChromaDB writes here. This directory is mounted as a volume in
# docker-compose.yml so memories survive container restarts.
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]