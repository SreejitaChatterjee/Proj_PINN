"""
Model Export Utilities

Export trained PINN models to ONNX format for deployment.

Usage:
    python scripts/export.py --model quadrotor --output model.onnx

Or programmatically:
    from scripts.export import export_to_onnx
    export_to_onnx(model, "model.onnx")
"""

import argparse
from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = None,
    opset_version: int = 14,
    dynamic_axes: dict = None,
) -> str:
    """
    Export a PINN model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path for output .onnx file
        input_shape: Input tensor shape (default: inferred from model)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size

    Returns:
        Path to exported ONNX file
    """
    model.eval()

    # Infer input shape from model
    if input_shape is None:
        if hasattr(model, "input_dim"):
            input_shape = (1, model.input_dim)
        else:
            raise ValueError("Cannot infer input shape. Provide input_shape argument.")

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes for batch dimension
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy export for compatibility
    )

    print(f"Exported model to {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {model(dummy_input).shape}")

    return str(output_path)


def verify_onnx(onnx_path: str, test_input: torch.Tensor = None) -> bool:
    """
    Verify an exported ONNX model.

    Args:
        onnx_path: Path to ONNX file
        test_input: Optional test input tensor

    Returns:
        True if verification passes
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Install onnx and onnxruntime: pip install onnx onnxruntime")
        return False

    # Check model is valid
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"ONNX model {onnx_path} is valid")

    # Run inference
    if test_input is not None:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: test_input.numpy()})
        print(f"ONNX inference output shape: {result[0].shape}")

    return True


def export_quadrotor_model(
    model_path: str = "models/quadrotor_pinn_diverse.pth",
    output_path: str = "models/quadrotor_pinn.onnx",
) -> str:
    """Export the trained quadrotor model to ONNX."""
    from pinn_model import QuadrotorPINN

    # Load model
    model = QuadrotorPINN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Export
    return export_to_onnx(model, output_path, input_shape=(1, 16))


def main():
    parser = argparse.ArgumentParser(description="Export PINN model to ONNX")
    parser.add_argument(
        "--model", default="quadrotor", choices=["quadrotor", "pendulum", "cartpole"]
    )
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    args = parser.parse_args()

    # Select model
    if args.model == "quadrotor":
        from pinn_model import QuadrotorPINN

        model = QuadrotorPINN()
        weights = args.weights or "models/quadrotor_pinn_diverse.pth"
        output = args.output or "models/quadrotor_pinn.onnx"
    elif args.model == "pendulum":
        from pinn_base import PendulumPINN

        model = PendulumPINN()
        weights = args.weights
        output = args.output or "models/pendulum_pinn.onnx"
    elif args.model == "cartpole":
        from pinn_base import CartPolePINN

        model = CartPolePINN()
        weights = args.weights
        output = args.output or "models/cartpole_pinn.onnx"

    # Load weights if provided
    if weights and Path(weights).exists():
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        print(f"Loaded weights from {weights}")

    # Export
    onnx_path = export_to_onnx(model, output)

    # Verify
    if args.verify:
        test_input = torch.randn(1, model.input_dim)
        verify_onnx(onnx_path, test_input)


if __name__ == "__main__":
    main()
