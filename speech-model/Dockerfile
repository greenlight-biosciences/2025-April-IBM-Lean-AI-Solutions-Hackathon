# Dockerfile
FROM glbdmcontainer.azurecr.io/granite3-speech-model:0.1

# Set Hugging Face cache directory as a volume
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
VOLUME ["/app/huggingface_cache"]

# Copy app
WORKDIR /app
COPY app.py /app/

# Expose OpenAI-style audio endpoint
EXPOSE 8005

# Launch FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
