# ==============================================================================
# Production Dockerfile for Banking Fraud Detection FastAPI Backend
# ==============================================================================

FROM python:3.11-slim

# Set environment variables for production execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies using layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, pre-trained model artifacts, and logging directory
COPY app/ /app/app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY logs/ /app/logs/
COPY api.py /app/api.py

# Ensure logs directory exists and is writable
RUN mkdir -p /app/logs

# Expose backend API port
EXPOSE 8000

# Docker healthcheck querying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server execution (no reload mode)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
