"""
Experiment Tracking

Unified interface for Weights & Biases and MLflow logging.

Usage:
    from scripts.tracking import ExperimentTracker

    tracker = ExperimentTracker(backend="wandb", project="pinn-dynamics")
    tracker.log_params({"lr": 0.001, "epochs": 100})
    tracker.log_metrics({"loss": 0.5, "val_loss": 0.6}, step=10)
    tracker.finish()
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics at a step."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: str = None):
        """Log a file artifact."""
        pass

    @abstractmethod
    def finish(self):
        """Finish the run."""
        pass


class WandbTracker(BaseTracker):
    """Weights & Biases tracker."""

    def __init__(
        self,
        project: str = "pinn-dynamics",
        entity: str = None,
        name: str = None,
        config: Dict = None,
        tags: list = None,
    ):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("Install wandb: pip install wandb")

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            reinit=True,
        )

    def log_params(self, params: Dict[str, Any]):
        self.wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str = None):
        artifact = self.wandb.Artifact(name or Path(path).stem, type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_model(self, model, name: str = "model"):
        """Log PyTorch model."""
        import torch
        path = f"/tmp/{name}.pth"
        torch.save(model.state_dict(), path)
        self.log_artifact(path, name)

    def watch(self, model, log: str = "all", log_freq: int = 100):
        """Watch model gradients and parameters."""
        self.wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        self.wandb.finish()


class MLflowTracker(BaseTracker):
    """MLflow tracker."""

    def __init__(
        self,
        experiment_name: str = "pinn-dynamics",
        tracking_uri: str = None,
        run_name: str = None,
    ):
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            raise ImportError("Install mlflow: pip install mlflow")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        self.mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, name: str = None):
        self.mlflow.log_artifact(path)

    def log_model(self, model, name: str = "model"):
        """Log PyTorch model."""
        self.mlflow.pytorch.log_model(model, name)

    def finish(self):
        self.mlflow.end_run()

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class LocalTracker(BaseTracker):
    """Simple local file-based tracker (no external dependencies)."""

    def __init__(self, log_dir: str = "runs", run_name: str = None):
        import datetime
        if run_name is None:
            run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.params = {}
        self.metrics = []

    def log_params(self, params: Dict[str, Any]):
        self.params.update(params)
        with open(self.log_dir / "params.json", 'w') as f:
            json.dump(self.params, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        entry = {"step": step, **metrics}
        self.metrics.append(entry)

        with open(self.log_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_artifact(self, path: str, name: str = None):
        import shutil
        dest = self.log_dir / (name or Path(path).name)
        shutil.copy(path, dest)

    def log_model(self, model, name: str = "model"):
        """Log PyTorch model."""
        import torch
        torch.save(model.state_dict(), self.log_dir / f"{name}.pth")

    def finish(self):
        # Save final summary
        summary = {
            "params": self.params,
            "final_metrics": self.metrics[-1] if self.metrics else {},
            "log_dir": str(self.log_dir),
        }
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Run saved to {self.log_dir}")


class ExperimentTracker:
    """
    Unified experiment tracker factory.

    Supports: wandb, mlflow, local
    """

    def __init__(
        self,
        backend: str = "local",
        project: str = "pinn-dynamics",
        run_name: str = None,
        config: Dict = None,
        **kwargs
    ):
        self.backend = backend.lower()

        if self.backend == "wandb":
            self.tracker = WandbTracker(
                project=project,
                name=run_name,
                config=config,
                **kwargs
            )
        elif self.backend == "mlflow":
            self.tracker = MLflowTracker(
                experiment_name=project,
                run_name=run_name,
                **kwargs
            )
        elif self.backend == "local":
            self.tracker = LocalTracker(
                run_name=run_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'wandb', 'mlflow', or 'local'")

        if config:
            self.log_params(config)

    def log_params(self, params: Dict[str, Any]):
        self.tracker.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        self.tracker.log_metrics(metrics, step)

    def log_artifact(self, path: str, name: str = None):
        self.tracker.log_artifact(path, name)

    def log_model(self, model, name: str = "model"):
        if hasattr(self.tracker, 'log_model'):
            self.tracker.log_model(model, name)
        else:
            import torch
            path = f"/tmp/{name}.pth"
            torch.save(model.state_dict(), path)
            self.tracker.log_artifact(path, name)

    def watch(self, model, **kwargs):
        """Watch model (wandb only)."""
        if hasattr(self.tracker, 'watch'):
            self.tracker.watch(model, **kwargs)

    def finish(self):
        self.tracker.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


# Example usage
if __name__ == "__main__":
    # Test local tracker
    with ExperimentTracker(backend="local", run_name="test_run") as tracker:
        tracker.log_params({"lr": 0.001, "epochs": 100})

        for epoch in range(5):
            tracker.log_metrics({
                "train_loss": 1.0 / (epoch + 1),
                "val_loss": 1.1 / (epoch + 1),
            }, step=epoch)

    print("Tracking test complete!")
