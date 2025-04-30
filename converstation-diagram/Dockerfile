# Use an official slim Python runtime as a parent image  
FROM python:3.10-slim AS base

# Set working directory  
WORKDIR /app  

# Install system dependencies: FFmpeg for audio conversion, build tools for any Python wheels  
RUN apt-get update \  
    && apt-get install -y --no-install-recommends \  
       ffmpeg \            
       build-essential \   
    && rm -rf /var/lib/apt/lists/*  

# Copy requirements first to leverage Docker cache  
COPY requirements.txt ./  

# Install Python dependencies  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy application code  
COPY app.py ./  

# Expose Streamlit port  
EXPOSE 8501  

# Run the Streamlit app  
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]  