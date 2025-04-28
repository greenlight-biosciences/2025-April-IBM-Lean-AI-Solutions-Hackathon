# Dockerfile

FROM python:3.10-slim

# 1) Install system deps: graphviz (dot) + build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         graphviz \
         libgraphviz-dev \
         pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 2) Set workdir
WORKDIR /app

# 3) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your application code
COPY . .

# 5) Expose port & default command (adjust module:path as needed)
EXPOSE 8000
CMD ["python", "graphviz_server.py"]
