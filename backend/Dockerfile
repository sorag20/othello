FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
