# PINN Dynamics Framework
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Base image with dependencies
# ============================================
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Development image
# ============================================
FROM base as development

# Install dev dependencies
RUN pip install --no-cache-dir pytest black flake8 jupyter

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Default command for development
CMD ["bash"]

# ============================================
# Stage 3: Production image
# ============================================
FROM base as production

# Copy only necessary files
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY models/ ./models/
COPY demo.py .
COPY setup.py .

# Install package
RUN pip install .

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from scripts import QuadrotorPINN; print('OK')" || exit 1

# Default command: run API server
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 4: API-only image (smallest)
# ============================================
FROM python:3.11-slim as api

WORKDIR /app

# Install only runtime dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    fastapi \
    uvicorn

# Copy only what's needed for API
COPY scripts/pinn_base.py scripts/pinn_model.py scripts/api.py ./scripts/
COPY scripts/__init__.py ./scripts/
COPY models/*.pth ./models/
COPY models/*.pkl ./models/

EXPOSE 8000

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
