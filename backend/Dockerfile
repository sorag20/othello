# Stage 1: Build edax for the target Linux architecture
FROM debian:bookworm-slim AS edax-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/abulmo/edax-reversi.git /edax-src

WORKDIR /edax-src/src

# Build for the native Linux architecture.
# Linux binary is named lEdax-<arch>; rename to mEdax for a fixed path.
RUN UARCH=$(uname -m) && \
    if [ "$UARCH" = "aarch64" ]; then BUILD_ARCH=armv8-a; \
    elif [ "$UARCH" = "x86_64" ]; then BUILD_ARCH=x86-64-modern; \
    else BUILD_ARCH=armv8-a; fi && \
    mkdir -p ../bin && \
    make build ARCH=$BUILD_ARCH CC=gcc OS=linux && \
    mv ../bin/lEdax-$BUILD_ARCH ../bin/mEdax && \
    ls -la ../bin/

# Stage 2: Main Python application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy edax binary from build stage (data files are mounted via docker-compose volume)
COPY --from=edax-builder /edax-src/bin/ /edax/bin/

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller image size)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn for production serving
RUN pip install --no-cache-dir gunicorn==23.0.0

# Copy application code
COPY . .

# Create directory for model weights
RUN mkdir -p models/weights

# Expose the Flask port
EXPOSE 5000

# Run the Flask application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "30", "app:create_app()"]
