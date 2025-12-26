"""
FastAPI Model Serving Endpoint

Run:
    uvicorn scripts.api:app --reload --port 8000

Endpoints:
    GET  /health          - Health check
    GET  /info            - Model info
    POST /predict         - Single-step prediction
    POST /rollout         - Multi-step rollout
    POST /physics_loss    - Compute physics loss
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import models
try:
    from .pinn_base import CartPolePINN, PendulumPINN
    from .pinn_model import QuadrotorPINN
except ImportError:
    from pinn_base import CartPolePINN, PendulumPINN
    from pinn_model import QuadrotorPINN


# ============================================
# Request/Response Models
# ============================================


class PredictRequest(BaseModel):
    """Single-step prediction request."""

    state: List[float] = Field(..., description="Current state vector")
    control: List[float] = Field(..., description="Control input vector")

    class Config:
        json_schema_extra = {
            "example": {
                "state": [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "control": [0.67, 0, 0, 0],
            }
        }


class PredictResponse(BaseModel):
    """Prediction response."""

    next_state: List[float]
    model: str
    state_names: List[str]


class RolloutRequest(BaseModel):
    """Multi-step rollout request."""

    initial_state: List[float]
    controls: List[List[float]] = Field(..., description="Sequence of control inputs")
    n_steps: Optional[int] = Field(None, description="Number of steps (defaults to len(controls))")


class RolloutResponse(BaseModel):
    """Rollout response."""

    trajectory: List[List[float]]
    n_steps: int


class PhysicsLossRequest(BaseModel):
    """Physics loss computation request."""

    inputs: List[List[float]] = Field(..., description="Batch of state+control vectors")
    outputs: List[List[float]] = Field(..., description="Batch of predicted next states")
    dt: float = Field(0.001, description="Timestep")


class PhysicsLossResponse(BaseModel):
    """Physics loss response."""

    loss: float


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    state_dim: int
    control_dim: int
    n_parameters: int
    state_names: List[str]
    control_names: List[str]
    learnable_params: dict


# ============================================
# Global Model State
# ============================================

models = {}


def load_models():
    """Load models on startup."""
    global models

    # Load quadrotor model
    model_path = Path(__file__).parent.parent / "models" / "quadrotor_pinn_diverse.pth"
    quadrotor = QuadrotorPINN()

    if model_path.exists():
        quadrotor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"Loaded quadrotor model from {model_path}")

    quadrotor.eval()
    models["quadrotor"] = quadrotor

    # Also load simple models (untrained)
    models["pendulum"] = PendulumPINN()
    models["pendulum"].eval()

    models["cartpole"] = CartPolePINN()
    models["cartpole"].eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    load_models()
    yield
    models.clear()


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="PINN Dynamics API",
    description="Physics-Informed Neural Networks for Dynamics Prediction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(models.keys())}


@app.get("/info/{model_name}", response_model=ModelInfo)
async def model_info(model_name: str = "quadrotor"):
    """Get model information."""
    if model_name not in models:
        raise HTTPException(
            404, f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )

    model = models[model_name]

    return ModelInfo(
        name=model_name,
        state_dim=model.state_dim,
        control_dim=model.control_dim,
        n_parameters=sum(p.numel() for p in model.parameters()),
        state_names=model.get_state_names(),
        control_names=model.get_control_names(),
        learnable_params={k: v.item() for k, v in model.params.items()},
    )


@app.post("/predict/{model_name}", response_model=PredictResponse)
async def predict(request: PredictRequest, model_name: str = "quadrotor"):
    """Single-step prediction."""
    if model_name not in models:
        raise HTTPException(404, f"Model '{model_name}' not found")

    model = models[model_name]

    # Validate input dimensions
    if len(request.state) != model.state_dim:
        raise HTTPException(
            400, f"State must have {model.state_dim} elements, got {len(request.state)}"
        )
    if len(request.control) != model.control_dim:
        raise HTTPException(
            400,
            f"Control must have {model.control_dim} elements, got {len(request.control)}",
        )

    # Predict
    with torch.no_grad():
        inp = torch.tensor(request.state + request.control, dtype=torch.float32).unsqueeze(0)
        out = model(inp).squeeze(0).tolist()

    return PredictResponse(
        next_state=out,
        model=model_name,
        state_names=model.get_state_names(),
    )


@app.post("/rollout/{model_name}", response_model=RolloutResponse)
async def rollout(request: RolloutRequest, model_name: str = "quadrotor"):
    """Multi-step autoregressive rollout."""
    if model_name not in models:
        raise HTTPException(404, f"Model '{model_name}' not found")

    model = models[model_name]

    # Validate
    if len(request.initial_state) != model.state_dim:
        raise HTTPException(400, f"Initial state must have {model.state_dim} elements")

    n_steps = request.n_steps or len(request.controls)
    if len(request.controls) < n_steps:
        raise HTTPException(
            400, f"Need at least {n_steps} control inputs, got {len(request.controls)}"
        )

    # Rollout
    initial_state = torch.tensor(request.initial_state, dtype=torch.float32)
    controls = torch.tensor(request.controls[:n_steps], dtype=torch.float32)

    trajectory = model.rollout(initial_state, controls)

    return RolloutResponse(
        trajectory=trajectory.tolist(),
        n_steps=n_steps,
    )


@app.post("/physics_loss/{model_name}", response_model=PhysicsLossResponse)
async def compute_physics_loss(request: PhysicsLossRequest, model_name: str = "quadrotor"):
    """Compute physics loss for given inputs/outputs."""
    if model_name not in models:
        raise HTTPException(404, f"Model '{model_name}' not found")

    model = models[model_name]

    inputs = torch.tensor(request.inputs, dtype=torch.float32)
    outputs = torch.tensor(request.outputs, dtype=torch.float32)

    loss = model.physics_loss(inputs, outputs, dt=request.dt)

    return PhysicsLossResponse(loss=loss.item())


@app.get("/")
async def root():
    """Root endpoint with API documentation links."""
    return {
        "message": "PINN Dynamics API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "models": list(models.keys()),
    }


# ============================================
# Run server
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
