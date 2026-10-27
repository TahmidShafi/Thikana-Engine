# =============================================================================
# Thikana-Engine Production Dockerfile
# Optimized for Docker, Hugging Face Spaces (Docker Space), and Cloud Run.
# =============================================================================

FROM python:3.12-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Install minimal OS runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for security and Hugging Face Spaces compatibility (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy dependency manifest and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project directories
COPY --chown=user:user data/ ./data/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user src/ ./src/
COPY --chown=user:user app/ ./app/

# Switch to non-root user
USER user

# Warm up gazetteer and model during build/startup
RUN python -c "from src.pipeline import ThikanaPipeline; p = ThikanaPipeline(); p.parse('House 1, Road 1, Dhanmondi, Dhaka')"

# Healthcheck to ensure container is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Expose default port (7860 for Hugging Face Spaces, 8000 for standard Docker)
EXPOSE 7860 8000

# Run FastAPI app with Uvicorn, respecting dynamic $PORT if set by cloud host
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
