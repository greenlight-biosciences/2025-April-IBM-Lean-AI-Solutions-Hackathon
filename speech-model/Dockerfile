# Dockerfile
FROM python:3.10-slim

# System deps for audio handling
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir fastapi uvicorn torch torchaudio transformers

# Install git for pip to fetch from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/huggingface/transformers.git

# Copy app
WORKDIR /app
COPY app.py /app/

# Expose OpenAI-style audio endpoint
EXPOSE 8005

# Launch FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
