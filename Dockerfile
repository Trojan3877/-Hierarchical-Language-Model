# Use a slim Python base
FROM python:3.11-slim

WORKDIR /app

# Install system deps for FAISS & transformers
RUN apt-get update && apt-get install -y \
    git build-essential libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache dir for sentence-transformers
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentencetransformers

# Copy the app
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Default entrypoint for API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
