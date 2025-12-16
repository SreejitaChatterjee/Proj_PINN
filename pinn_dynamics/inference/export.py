"""
Model export utilities for deployment.

Supports exporting PINN models to:
    - ONNX: For cross-platform deployment
    - TorchScript: For C++ inference
"""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def export_onnx(
    model,
    output_path: str,
    input_size: int = None,
    opset_version: int = 11,
    dynamic_axes: bool = True,
    verify: bool = True,
) -> str:
    """
    Export model to ONNX format.

    Args:
        model: A DynamicsPINN model
        output_path: Path for output .onnx file
        input_size: Input tensor size (default: model.input_dim)
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
        verify: Verify exported model with onnxruntime

    Returns:
        Path to exported model

    Example:
        from pinn_dynamics import QuadrotorPINN, export_onnx

        model = QuadrotorPINN()
        model.load_state_dict(torch.load('model.pth'))

        export_onnx(model, 'quadrotor.onnx')
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("ONNX export requires: pip install onnx")

    model.eval()

    # Determine input size
    if input_size is None:
        input_size = getattr(model, "input_dim", None)
        if input_size is None:
            raise ValueError("Could not determine input_size. Please specify explicitly.")

    # Create dummy input
    dummy_input = torch.randn(1, input_size)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_config,
    )

    logger.info(f"Exported ONNX model to {output_path}")

    # Verify
    if verify:
        _verify_onnx(output_path, dummy_input)

    return str(output_path)


def _verify_onnx(onnx_path: Path, dummy_input: torch.Tensor):
    """Verify ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("Skipping ONNX verification (requires onnxruntime)")
        return

    # Check model is valid
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Run inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    logger.info(f"ONNX verification passed. Output shape: {ort_outputs[0].shape}")


def export_torchscript(
    model,
    output_path: str,
    method: str = "trace",
    input_size: int = None,
) -> str:
    """
    Export model to TorchScript format.

    Args:
        model: A DynamicsPINN model
        output_path: Path for output .pt file
        method: 'trace' or 'script'
        input_size: Input tensor size (default: model.input_dim)

    Returns:
        Path to exported model

    Example:
        from pinn_dynamics import QuadrotorPINN, export_torchscript

        model = QuadrotorPINN()
        model.load_state_dict(torch.load('model.pth'))

        export_torchscript(model, 'quadrotor_scripted.pt')
    """
    model.eval()

    # Determine input size
    if input_size is None:
        input_size = getattr(model, "input_dim", None)
        if input_size is None:
            raise ValueError("Could not determine input_size. Please specify explicitly.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "trace":
        dummy_input = torch.randn(1, input_size)
        scripted_model = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

    scripted_model.save(str(output_path))
    logger.info(f"Exported TorchScript model to {output_path}")

    return str(output_path)


def load_onnx_session(onnx_path: str):
    """
    Load ONNX model for inference.

    Args:
        onnx_path: Path to .onnx file

    Returns:
        onnxruntime InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("ONNX inference requires: pip install onnxruntime")

    return ort.InferenceSession(onnx_path)


def load_torchscript(path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """
    Load TorchScript model.

    Args:
        path: Path to .pt file
        device: Device to load to

    Returns:
        TorchScript module
    """
    return torch.jit.load(path, map_location=device)
