"""
Export Model: Convert to ONNX or TorchScript for deployment.

This example shows how to:
1. Export a trained model to ONNX format
2. Export to TorchScript
3. Run inference with the exported model
"""

import torch
import numpy as np
from pathlib import Path

from pinn_dynamics import QuadrotorPINN
from pinn_dynamics.inference import export_onnx, export_torchscript, load_torchscript


def main():
    # Paths
    MODEL_PATH = Path(__file__).parent.parent / "models" / "quadrotor_pinn_diverse.pth"
    ONNX_PATH = Path(__file__).parent.parent / "models" / "quadrotor.onnx"
    TORCHSCRIPT_PATH = Path(__file__).parent.parent / "models" / "quadrotor_scripted.pt"

    # 1. Load trained model
    print("Loading model...")
    model = QuadrotorPINN()

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        print(f"  Loaded weights from {MODEL_PATH}")
    else:
        print(f"  No weights found at {MODEL_PATH}, using random initialization")

    model.eval()

    # 2. Export to ONNX
    print("\nExporting to ONNX...")
    try:
        export_onnx(model, str(ONNX_PATH), verify=True)
        print(f"  Saved to {ONNX_PATH}")
    except ImportError as e:
        print(f"  Skipping ONNX export (missing dependency): {e}")

    # 3. Export to TorchScript
    print("\nExporting to TorchScript...")
    export_torchscript(model, str(TORCHSCRIPT_PATH))
    print(f"  Saved to {TORCHSCRIPT_PATH}")

    # 4. Test TorchScript inference
    print("\nTesting TorchScript inference...")
    scripted_model = load_torchscript(str(TORCHSCRIPT_PATH))

    # Create test input
    test_input = torch.randn(1, 16)  # [batch, state + control]

    # Run inference
    with torch.no_grad():
        output_original = model(test_input)
        output_scripted = scripted_model(test_input)

    # Compare outputs
    diff = torch.abs(output_original - output_scripted).max().item()
    print(f"  Max difference between original and scripted: {diff:.2e}")

    if diff < 1e-5:
        print("  Export successful!")
    else:
        print("  Warning: Outputs differ significantly")

    # 5. Test ONNX inference (if available)
    print("\nTesting ONNX inference...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ONNX_PATH))
        onnx_input = {session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = session.run(None, onnx_input)[0]

        diff = np.abs(output_original.numpy() - onnx_output).max()
        print(f"  Max difference between original and ONNX: {diff:.2e}")

        if diff < 1e-5:
            print("  ONNX export successful!")
    except ImportError:
        print("  Skipping ONNX test (onnxruntime not installed)")
    except Exception as e:
        print(f"  ONNX test failed: {e}")

    # 6. Print export summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    print(f"Original model size: {MODEL_PATH.stat().st_size / 1024:.1f} KB" if MODEL_PATH.exists() else "Original model: N/A")
    if ONNX_PATH.exists():
        print(f"ONNX model size: {ONNX_PATH.stat().st_size / 1024:.1f} KB")
    if TORCHSCRIPT_PATH.exists():
        print(f"TorchScript model size: {TORCHSCRIPT_PATH.stat().st_size / 1024:.1f} KB")
    print("\nUse these exported models for:")
    print("  - ONNX: Cross-platform inference, TensorRT, embedded systems")
    print("  - TorchScript: C++ inference, mobile deployment")


if __name__ == "__main__":
    main()
