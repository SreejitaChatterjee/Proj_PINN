# Docker Configuration

Docker files for containerized deployment.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Main Docker image definition |
| `docker-compose.yaml` | Multi-container orchestration |
| `.dockerignore` | Files to exclude from build context |

## Usage

```bash
# Build image
docker build -t pinn-dynamics .

# Run with docker-compose
docker-compose up

# Run interactive shell
docker run -it pinn-dynamics bash
```

## Image Contents

- Python 3.11+
- PyTorch with CUDA support
- PINN dynamics framework
- All dependencies from requirements.txt
