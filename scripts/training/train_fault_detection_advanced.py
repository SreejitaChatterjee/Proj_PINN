#!/usr/bin/env python3
"""
Advanced Motor Fault Detection Training
========================================

Comprehensive training script with all advanced features:
- Multiple model architectures (CNN, Transformer, TCN, Ensemble)
- Uncertainty quantification (MC Dropout, Conformal Prediction)
- Explainability (GradCAM, Sensor Importance)
- Deployment (ONNX export, Quantization)

Usage:
    python scripts/train_fault_detection_advanced.py --model ensemble --task multiclass

Dataset: PADRE (AeroLab, Poznan University of Technology)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fault_detection.advanced_tasks import (
    AnomalyDetector,
    PerMotorClassifier,
    SeverityRegressor,
)
from fault_detection.data import DataAugmentation, PADREDataset, create_data_loaders
from fault_detection.deployment import (
    QuantizedModel,
    StreamingInference,
    export_onnx,
    export_torchscript,
)
from fault_detection.explainability import GradCAM1D, SensorImportance
from fault_detection.models import (
    EnsembleDetector,
    MotorFaultCNN,
    MultiScaleCNN,
    TCNDetector,
    TransformerDetector,
)
from fault_detection.uncertainty import (
    ConformalPredictor,
    MCDropoutWrapper,
    OODDetector,
    TemperatureScaler,
)

# =============================================================================
# Training Utilities
# =============================================================================


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    if scheduler:
        scheduler.step()

    return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = F.softmax(outputs, dim=-1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train(model, train_loader, val_loader, config, device):
    """Full training loop."""
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    class_weights = config.get("class_weights")
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for {config['epochs']} epochs on {device}")
    print("-" * 70)

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.1f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                },
                config["save_path"],
            )
            print(f"  -> Saved best model (acc: {best_val_acc:.1f}%)")

    return history, best_val_acc


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Advanced Fault Detection Training")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PADRE_dataset/Parrot_Bebop_2/Normalized_data",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["binary", "multiclass", "per_motor"],
    )
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["cnn", "transformer", "multiscale", "tcn", "ensemble"],
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save_dir", type=str, default="models/fault_detection_advanced")

    # Advanced features
    parser.add_argument(
        "--uncertainty", action="store_true", help="Enable uncertainty quantification"
    )
    parser.add_argument("--explainability", action="store_true", help="Run explainability analysis")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Quantize model")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Load Data
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Loading PADRE Dataset")
    print("=" * 70)

    train_loader, val_loader, test_loader, data_info = create_data_loaders(
        data_dir=args.data_dir,
        task=args.task,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    n_classes = data_info["n_classes"]
    print(
        f"\nSplit sizes: Train={data_info['train_size']}, "
        f"Val={data_info['val_size']}, Test={data_info['test_size']}"
    )

    # ==========================================================================
    # Create Model
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"Creating {args.model.upper()} Model")
    print("=" * 70)

    if args.model == "cnn":
        model = MotorFaultCNN(n_input_channels=24, n_classes=n_classes, dropout=args.dropout)
    elif args.model == "transformer":
        model = TransformerDetector(n_input_channels=24, n_classes=n_classes, dropout=args.dropout)
    elif args.model == "multiscale":
        model = MultiScaleCNN(n_input_channels=24, n_classes=n_classes, dropout=args.dropout)
    elif args.model == "tcn":
        model = TCNDetector(n_input_channels=24, n_classes=n_classes, dropout=args.dropout)
    elif args.model == "ensemble":
        model = EnsembleDetector(n_input_channels=24, n_classes=n_classes, dropout=args.dropout)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ==========================================================================
    # Train
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "class_weights": data_info["class_weights"],
        "save_path": str(save_dir / f"{args.model}_{args.task}.pth"),
    }

    history, best_acc = train(model, train_loader, val_loader, config, device)

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(config["save_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2f}%")

    if args.task == "binary":
        target_names = ["Normal", "Faulty"]
    elif args.task == "multiclass":
        target_names = ["Normal", "Chipped", "Bent"]
    else:
        target_names = None

    print("\nClassification Report:")
    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=target_names,
        )
    )

    # ==========================================================================
    # Uncertainty Quantification
    # ==========================================================================
    if args.uncertainty:
        print("\n" + "=" * 70)
        print("Uncertainty Quantification")
        print("=" * 70)

        # MC Dropout
        print("\n1. MC Dropout Uncertainty:")
        mc_model = MCDropoutWrapper(model, n_samples=30)
        mc_model.model.to(device)

        # Test on a batch
        test_batch, test_labels = next(iter(test_loader))
        test_batch = test_batch.to(device)

        unc_output = mc_model(test_batch[:5])
        print(f"   Predictions: {unc_output.predictions.cpu().numpy()}")
        print(f"   Confidence: {unc_output.confidence.cpu().numpy()}")
        print(f"   Uncertainty: {unc_output.uncertainty.cpu().numpy()}")

        # Temperature Scaling
        print("\n2. Temperature Scaling Calibration:")
        temp_model = TemperatureScaler(model)
        temp_model.calibrate(val_loader, device)

        ece_before = TemperatureScaler.compute_ece(
            torch.tensor(test_metrics["probabilities"]),
            torch.tensor(test_metrics["labels"]),
        )
        print(f"   ECE before calibration: {ece_before:.4f}")

        # Conformal Prediction
        print("\n3. Conformal Prediction:")
        conformal = ConformalPredictor(model, alpha=0.1)
        conformal.calibrate(val_loader, device)

        coverage_stats = conformal.compute_coverage(test_loader, device)
        print(f"   Target coverage: {coverage_stats['target_coverage']:.1%}")
        print(f"   Empirical coverage: {coverage_stats['coverage']:.1%}")
        print(f"   Average set size: {coverage_stats['avg_set_size']:.2f}")

    # ==========================================================================
    # Explainability
    # ==========================================================================
    if args.explainability:
        print("\n" + "=" * 70)
        print("Explainability Analysis")
        print("=" * 70)

        # Get a sample
        sample_batch, sample_labels = next(iter(test_loader))
        sample = sample_batch[:1].to(device)

        # GradCAM
        print("\n1. GradCAM-1D Analysis:")
        try:
            gradcam = GradCAM1D(model)
            heatmap, pred, conf = gradcam(sample)
            print(f"   Prediction: {pred} (confidence: {conf:.1%})")
            print(f"   Important regions: timesteps {np.where(heatmap > 0.7)[0][:5]}...")

            fig = gradcam.visualize(sample, title=f"GradCAM: Class {pred}")
            fig.savefig(save_dir / "gradcam_example.png", dpi=150, bbox_inches="tight")
            print(f"   Saved: {save_dir}/gradcam_example.png")
            gradcam.remove_hooks()
        except Exception as e:
            print(f"   GradCAM failed: {e}")

        # Sensor Importance
        print("\n2. Sensor Importance Analysis:")
        sensor_imp = SensorImportance(model)
        importance = sensor_imp.gradient_importance(sample)

        top_sensors = np.argsort(importance)[-5:][::-1]
        print(f"   Top 5 sensors: {[sensor_imp.SENSOR_NAMES[i] for i in top_sensors]}")

        motor_imp = sensor_imp.get_grouped_importance(importance, "motor")
        print(f"   Motor importance: {motor_imp}")

        fig = sensor_imp.visualize(importance)
        fig.savefig(save_dir / "sensor_importance.png", dpi=150, bbox_inches="tight")
        print(f"   Saved: {save_dir}/sensor_importance.png")

    # ==========================================================================
    # Export & Deployment
    # ==========================================================================
    if args.export_onnx:
        print("\n" + "=" * 70)
        print("ONNX Export")
        print("=" * 70)

        onnx_path = export_onnx(
            model.cpu(),
            save_dir / f"{args.model}_{args.task}.onnx",
            input_shape=(1, 24, args.window_size),
        )

        # TorchScript
        ts_path = export_torchscript(
            model,
            save_dir / f"{args.model}_{args.task}.pt",
            input_shape=(1, 24, args.window_size),
        )

    if args.quantize:
        print("\n" + "=" * 70)
        print("Model Quantization")
        print("=" * 70)

        quant = QuantizedModel(model)
        quant_model = quant.quantize_dynamic()

        size_info = quant.get_size_reduction()
        print(f"Original size: {size_info['original_mb']:.2f} MB")
        print(f"Quantized size: {size_info['quantized_mb']:.2f} MB")
        print(f"Reduction: {size_info['reduction_pct']:.1f}%")

        bench = quant.benchmark()
        print(f"Original latency: {bench['original_ms']:.2f} ms")
        print(f"Quantized latency: {bench['quantized_ms']:.2f} ms")
        print(f"Speedup: {bench['speedup']:.2f}x")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    results = {
        "model": args.model,
        "task": args.task,
        "test_accuracy": test_metrics["accuracy"],
        "n_classes": n_classes,
        "n_params": n_params,
        "epochs": args.epochs,
        "best_epoch": checkpoint["epoch"],
        "config": vars(args),
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    with open(save_dir / f"results_{args.model}_{args.task}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved: {config['save_path']}")
    print(f"Results saved: {save_dir}/results_{args.model}_{args.task}.json")


if __name__ == "__main__":
    main()
