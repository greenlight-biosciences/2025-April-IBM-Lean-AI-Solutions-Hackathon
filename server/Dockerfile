# Dockerfile

FROM python:3.10-slim

# Combine apt-get commands into a single RUN statement for optimization
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         graphviz \
         libgraphviz-dev \
         pkg-config \
         tk-dev \
         libtk8.6 \
         libtcl8.6 \
    && rm -rf /var/lib/apt/lists/*

# Add missing dependencies for tksvg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         cmake \
         build-essential \
         tcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure scikit-build is installed before other dependencies
RUN pip install --no-cache-dir scikit-build

# 2) Set workdir
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Upgrade setuptools and wheel to the latest versions
RUN pip install --no-cache-dir --upgrade setuptools wheel

# 3) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your application code
COPY . .

# Install pytest for testing
RUN pip install --no-cache-dir pytest

# Run tests during the build process
# RUN pytest

# 5) Expose port & default command (adjust module:path as needed)
EXPOSE 8000
CMD ["python", "graphviz_server.py"]
