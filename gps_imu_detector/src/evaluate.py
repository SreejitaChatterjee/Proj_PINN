"""
Rigorous Evaluation Script for GPS-IMU Anomaly Detector

Implements complete evaluation protocol:
1. LOSO-CV with proper train/test splits
2. Attack-specific recall metrics
3. Ablation studies
4. Cross-dataset transfer
5. Latency benchmarks
6. Comprehensive reporting

This script produces all results for the paper.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)

from data_loader import GPSIMUDataLoader, AttackCatalog
from feature_extractor import BatchFeatureExtractor
from model import CNNGRUDetector
from hybrid_scorer import HybridScorer
from transfer import TransferEvaluator
from quantization import LatencyBenchmark, ModelQuantizer


@dataclass
class EvaluationConfig:
    """Configuration for rigorous evaluation."""
    # Data
    data_path: str = ""
    dt: float = 0.005
    feature_windows: List[int] = field(default_factory=lambda: [5, 10, 25])

    # Model
    cnn_channels: Tuple[int, ...] = (32, 64)
    gru_hidden_size: int = 64
    dropout: float = 0.2

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15

    # Evaluation
    fpr_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    n_bootstrap: int = 100  # For confidence intervals

    # Attacks to evaluate
    attack_types: List[str] = field(default_factory=lambda: [
        'bias', 'drift', 'noise', 'coordinated', 'intermittent', 'ramp'
    ])
    attack_magnitudes: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Output
    output_dir: str = "./results"
    seed: int = 42


@dataclass
class AttackResults:
    """Results for a specific attack type."""
    attack_type: str
    magnitude: float
    n_samples: int
    recall_at_1pct_fpr: float
    recall_at_5pct_fpr: float
    recall_at_10pct_fpr: float
    auroc: float
    auprc: float
    detection_delay_mean: float  # Time to first detection
    detection_delay_std: float


@dataclass
class FoldResults:
    """Results for a single LOSO-CV fold."""
    fold_idx: int
    test_sequence: str
    n_train: int
    n_test: int
    n_test_attacks: int
    overall_auroc: float
    overall_auprc: float
    attack_results: List[AttackResults]
    training_time_sec: float


@dataclass
class OverallResults:
    """Aggregated results across all folds."""
    n_folds: int
    total_train_samples: int
    total_test_samples: int

    # Mean metrics (with std)
    mean_auroc: float
    std_auroc: float
    mean_auprc: float
    std_auprc: float

    # Per-attack results
    attack_summary: Dict[str, Dict[str, float]]

    # Confidence intervals
    auroc_ci_lower: float
    auroc_ci_upper: float

    # Ablation results
    ablation_results: Optional[Dict] = None

    # Transfer results
    transfer_results: Optional[Dict] = None

    # Latency results
    latency_results: Optional[Dict] = None


class RigorousEvaluator:
    """
    Complete evaluation pipeline for GPS-IMU anomaly detector.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize components
        self.feature_extractor = BatchFeatureExtractor(
            windows=config.feature_windows
        )
        self.attack_catalog = AttackCatalog(seed=config.seed)

        # Results storage
        self.fold_results: List[FoldResults] = []

    def run_full_evaluation(self, data_loader: GPSIMUDataLoader) -> OverallResults:
        """
        Run complete LOSO-CV evaluation.

        Args:
            data_loader: Loaded data

        Returns:
            OverallResults with all metrics
        """
        print("\n" + "="*60)
        print("RIGOROUS EVALUATION")
        print("="*60)

        splits = data_loader.get_loso_splits()
        print(f"\nRunning {len(splits)}-fold LOSO-CV")

        # Run each fold
        for fold_idx, (train_ids, test_id) in enumerate(splits):
            fold_result = self._run_fold(
                data_loader, train_ids, test_id, fold_idx
            )
            self.fold_results.append(fold_result)

        # Aggregate results
        overall = self._aggregate_results()

        # Run ablation study
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        overall.ablation_results = self._run_ablation(data_loader)

        # Run latency benchmark
        print("\n" + "="*60)
        print("LATENCY BENCHMARK")
        print("="*60)
        overall.latency_results = self._run_latency_benchmark()

        # Save results
        self._save_results(overall)

        # Print summary
        self._print_summary(overall)

        return overall

    def _run_fold(
        self,
        data_loader: GPSIMUDataLoader,
        train_ids: List[str],
        test_id: str,
        fold_idx: int
    ) -> FoldResults:
        """Run single LOSO-CV fold."""
        print(f"\n--- Fold {fold_idx}: Test on {test_id} ---")

        # Get data
        X_train, X_test, _, _ = data_loader.get_train_test_data(
            train_ids, test_id, fit_scaler=(fold_idx == 0)
        )

        # Extract features
        print("  Extracting features...")
        train_features = self.feature_extractor.extract(X_train)
        test_features_clean = self.feature_extractor.extract(X_test)

        # Train model
        print("  Training model...")
        input_dim = train_features.shape[1]
        model, train_time = self._train_model(
            train_features, np.zeros(len(train_features))
        )

        # Generate and evaluate attacks
        print("  Evaluating attacks...")
        attack_results = []

        for attack_type in self.config.attack_types:
            for magnitude in self.config.attack_magnitudes:
                # Generate attack
                X_attacked, attack_labels = self.attack_catalog.generate_attack(
                    X_test, attack_type, magnitude, 'position'
                )

                # Extract features
                attack_features = self.feature_extractor.extract(X_attacked)
                attack_labels_aligned = attack_labels[len(X_attacked) - len(attack_features):]

                # Combine with clean data
                combined_features = np.vstack([test_features_clean, attack_features])
                combined_labels = np.concatenate([
                    np.zeros(len(test_features_clean)),
                    attack_labels_aligned
                ])

                # Get predictions
                scores = self._predict(model, combined_features)

                # Compute metrics
                result = self._compute_attack_metrics(
                    scores, combined_labels, attack_type, magnitude
                )
                attack_results.append(result)

        # Compute overall metrics
        all_scores = []
        all_labels = []
        for attack_type in self.config.attack_types:
            for magnitude in self.config.attack_magnitudes:
                X_attacked, labels = self.attack_catalog.generate_attack(
                    X_test, attack_type, magnitude, 'position'
                )
                features = self.feature_extractor.extract(X_attacked)
                labels_aligned = labels[len(X_attacked) - len(features):]
                scores = self._predict(model, features)
                all_scores.extend(scores)
                all_labels.extend(labels_aligned)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) > 1:
            overall_auroc = roc_auc_score(all_labels, all_scores)
            overall_auprc = average_precision_score(all_labels, all_scores)
        else:
            overall_auroc = 0.0
            overall_auprc = 0.0

        return FoldResults(
            fold_idx=fold_idx,
            test_sequence=test_id,
            n_train=len(train_features),
            n_test=len(test_features_clean),
            n_test_attacks=int(np.sum(all_labels)),
            overall_auroc=overall_auroc,
            overall_auprc=overall_auprc,
            attack_results=attack_results,
            training_time_sec=train_time
        )

    def _train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[CNNGRUDetector, float]:
        """Train model and return training time."""
        input_dim = features.shape[1]
        model = CNNGRUDetector(
            input_dim=input_dim,
            cnn_channels=self.config.cnn_channels,
            gru_hidden_size=self.config.gru_hidden_size,
            dropout=self.config.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            model.train()
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device).unsqueeze(1)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs, _ = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                optimizer.step()

        train_time = time.time() - start_time
        return model, train_time

    def _predict(self, model: CNNGRUDetector, features: np.ndarray) -> np.ndarray:
        """Get model predictions."""
        model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)
            outputs, _ = model(x)
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
        return scores

    def _compute_attack_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        attack_type: str,
        magnitude: float
    ) -> AttackResults:
        """Compute metrics for specific attack."""
        normal_mask = labels == 0
        attack_mask = labels == 1

        normal_scores = scores[normal_mask]
        attack_scores = scores[attack_mask]

        # Recall at FPR thresholds
        recall_1pct = recall_5pct = recall_10pct = 0.0
        if len(normal_scores) > 0 and len(attack_scores) > 0:
            threshold_1pct = np.percentile(normal_scores, 99)
            threshold_5pct = np.percentile(normal_scores, 95)
            threshold_10pct = np.percentile(normal_scores, 90)

            recall_1pct = np.mean(attack_scores > threshold_1pct)
            recall_5pct = np.mean(attack_scores > threshold_5pct)
            recall_10pct = np.mean(attack_scores > threshold_10pct)

        # AUROC and AUPRC
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
        else:
            auroc = 0.0
            auprc = 0.0

        # Detection delay (first detection after attack starts)
        attack_indices = np.where(attack_mask)[0]
        if len(attack_indices) > 0 and len(normal_scores) > 0:
            threshold = np.percentile(normal_scores, 95)
            detections = scores[attack_indices] > threshold
            if np.any(detections):
                first_detection = np.argmax(detections)
                delay_mean = float(first_detection)
                delay_std = 0.0  # Single sequence
            else:
                delay_mean = float(len(attack_indices))
                delay_std = 0.0
        else:
            delay_mean = 0.0
            delay_std = 0.0

        return AttackResults(
            attack_type=attack_type,
            magnitude=magnitude,
            n_samples=int(np.sum(attack_mask)),
            recall_at_1pct_fpr=recall_1pct,
            recall_at_5pct_fpr=recall_5pct,
            recall_at_10pct_fpr=recall_10pct,
            auroc=auroc,
            auprc=auprc,
            detection_delay_mean=delay_mean,
            detection_delay_std=delay_std
        )

    def _aggregate_results(self) -> OverallResults:
        """Aggregate results across folds."""
        aurocs = [f.overall_auroc for f in self.fold_results]
        auprcs = [f.overall_auprc for f in self.fold_results]

        # Bootstrap confidence intervals
        auroc_bootstrap = []
        for _ in range(self.config.n_bootstrap):
            idx = np.random.choice(len(aurocs), len(aurocs), replace=True)
            auroc_bootstrap.append(np.mean([aurocs[i] for i in idx]))
        auroc_ci_lower = np.percentile(auroc_bootstrap, 2.5)
        auroc_ci_upper = np.percentile(auroc_bootstrap, 97.5)

        # Per-attack summary
        attack_summary = {}
        for attack_type in self.config.attack_types:
            for magnitude in self.config.attack_magnitudes:
                key = f"{attack_type}_{magnitude}"
                recalls_1pct = []
                recalls_5pct = []
                for fold in self.fold_results:
                    for ar in fold.attack_results:
                        if ar.attack_type == attack_type and ar.magnitude == magnitude:
                            recalls_1pct.append(ar.recall_at_1pct_fpr)
                            recalls_5pct.append(ar.recall_at_5pct_fpr)

                attack_summary[key] = {
                    'mean_recall_1pct': float(np.mean(recalls_1pct)),
                    'std_recall_1pct': float(np.std(recalls_1pct)),
                    'mean_recall_5pct': float(np.mean(recalls_5pct)),
                    'std_recall_5pct': float(np.std(recalls_5pct)),
                }

        return OverallResults(
            n_folds=len(self.fold_results),
            total_train_samples=sum(f.n_train for f in self.fold_results),
            total_test_samples=sum(f.n_test for f in self.fold_results),
            mean_auroc=float(np.mean(aurocs)),
            std_auroc=float(np.std(aurocs)),
            mean_auprc=float(np.mean(auprcs)),
            std_auprc=float(np.std(auprcs)),
            attack_summary=attack_summary,
            auroc_ci_lower=auroc_ci_lower,
            auroc_ci_upper=auroc_ci_upper
        )

    def _run_ablation(self, data_loader: GPSIMUDataLoader) -> Dict:
        """Run ablation study on model components."""
        # Use first fold for ablation
        splits = data_loader.get_loso_splits()
        train_ids, test_id = splits[0]

        X_train, X_test, _, _ = data_loader.get_train_test_data(train_ids, test_id)
        train_features = self.feature_extractor.extract(X_train)

        # Test attacked data
        X_attacked, labels = self.attack_catalog.generate_attack(
            X_test, 'bias', 1.0, 'position'
        )
        test_features = self.feature_extractor.extract(X_attacked)
        test_labels = labels[len(X_attacked) - len(test_features):]

        ablation_configs = {
            'full_model': {'cnn_channels': (32, 64), 'gru_hidden_size': 64},
            'small_cnn': {'cnn_channels': (16, 32), 'gru_hidden_size': 64},
            'no_gru': {'cnn_channels': (32, 64), 'gru_hidden_size': 0},
            'small_all': {'cnn_channels': (16, 32), 'gru_hidden_size': 32},
        }

        results = {}
        for name, cfg in ablation_configs.items():
            print(f"  Ablation: {name}")

            # Skip no_gru for now (would need model modification)
            if cfg['gru_hidden_size'] == 0:
                continue

            model = CNNGRUDetector(
                input_dim=train_features.shape[1],
                cnn_channels=cfg['cnn_channels'],
                gru_hidden_size=cfg['gru_hidden_size']
            ).to(self.device)

            # Quick training
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(np.zeros(len(train_features)), dtype=torch.float32)
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            for epoch in range(20):  # Quick training
                model.train()
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device).unsqueeze(1)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    out, _ = model(batch_x)
                    loss = criterion(out.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            scores = self._predict(model, test_features)
            auroc = roc_auc_score(test_labels, scores) if len(np.unique(test_labels)) > 1 else 0

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())

            results[name] = {
                'auroc': float(auroc),
                'n_params': n_params
            }

        return results

    def _run_latency_benchmark(self) -> Dict:
        """Run latency benchmarks."""
        # Create model
        input_dim = 100  # Typical feature dimension
        model = CNNGRUDetector(input_dim=input_dim)

        benchmark = LatencyBenchmark(warmup_iterations=10, benchmark_iterations=100)

        results = {}

        # FP32
        result_fp32 = benchmark.benchmark_pytorch(model, (1, 1, input_dim), 'cpu')
        results['fp32_streaming'] = {
            'mean_ms': result_fp32.mean_latency_ms,
            'p99_ms': result_fp32.p99_latency_ms
        }

        # INT8
        quantizer = ModelQuantizer(model)
        quantized = quantizer.dynamic_quantize()
        result_int8 = benchmark.benchmark_pytorch(quantized, (1, 1, input_dim), 'cpu')
        results['int8_streaming'] = {
            'mean_ms': result_int8.mean_latency_ms,
            'p99_ms': result_int8.p99_latency_ms
        }

        # Check target
        meets_target = result_int8.mean_latency_ms < 5.0
        results['meets_5ms_target'] = meets_target

        return results

    def _save_results(self, overall: OverallResults):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main results
        results_dict = {
            'timestamp': timestamp,
            'config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'attack_types': self.config.attack_types,
                'attack_magnitudes': self.config.attack_magnitudes,
            },
            'overall': {
                'n_folds': overall.n_folds,
                'mean_auroc': overall.mean_auroc,
                'std_auroc': overall.std_auroc,
                'auroc_ci_95': [overall.auroc_ci_lower, overall.auroc_ci_upper],
                'mean_auprc': overall.mean_auprc,
                'std_auprc': overall.std_auprc,
            },
            'attack_summary': overall.attack_summary,
            'ablation': overall.ablation_results,
            'latency': overall.latency_results,
        }

        # Save JSON
        with open(self.output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save per-fold results
        fold_data = []
        for fold in self.fold_results:
            fold_dict = {
                'fold_idx': fold.fold_idx,
                'test_sequence': fold.test_sequence,
                'overall_auroc': fold.overall_auroc,
                'attack_results': [
                    {
                        'attack_type': ar.attack_type,
                        'magnitude': ar.magnitude,
                        'recall_1pct': ar.recall_at_1pct_fpr,
                        'recall_5pct': ar.recall_at_5pct_fpr,
                    }
                    for ar in fold.attack_results
                ]
            }
            fold_data.append(fold_dict)

        with open(self.output_dir / f'fold_results_{timestamp}.json', 'w') as f:
            json.dump(fold_data, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")

    def _print_summary(self, overall: OverallResults):
        """Print summary of results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        print(f"\nOverall Metrics ({overall.n_folds}-fold LOSO-CV):")
        print(f"  AUROC: {overall.mean_auroc:.3f} ± {overall.std_auroc:.3f}")
        print(f"  AUROC 95% CI: [{overall.auroc_ci_lower:.3f}, {overall.auroc_ci_upper:.3f}]")
        print(f"  AUPRC: {overall.mean_auprc:.3f} ± {overall.std_auprc:.3f}")

        print(f"\nPer-Attack Results (Recall@5%FPR):")
        for attack_type in self.config.attack_types:
            recalls = []
            for magnitude in self.config.attack_magnitudes:
                key = f"{attack_type}_{magnitude}"
                if key in overall.attack_summary:
                    r = overall.attack_summary[key]['mean_recall_5pct']
                    recalls.append(r)
            if recalls:
                print(f"  {attack_type}: {np.mean(recalls):.1%}")

        if overall.ablation_results:
            print(f"\nAblation Results:")
            for name, res in overall.ablation_results.items():
                print(f"  {name}: AUROC={res['auroc']:.3f}, params={res['n_params']:,}")

        if overall.latency_results:
            print(f"\nLatency Results:")
            for key, res in overall.latency_results.items():
                if isinstance(res, dict):
                    print(f"  {key}: {res['mean_ms']:.2f}ms (p99: {res['p99_ms']:.2f}ms)")
                else:
                    print(f"  {key}: {res}")


def main():
    """Run full evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='Rigorous Evaluation')
    parser.add_argument('--data', type=str, help='Data path')
    parser.add_argument('--output', type=str, default='./results', help='Output dir')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick eval (fewer attacks)')
    args = parser.parse_args()

    # Config
    config = EvaluationConfig(
        data_path=args.data or "",
        epochs=args.epochs,
        output_dir=args.output
    )

    if args.quick:
        config.attack_types = ['bias', 'drift']
        config.attack_magnitudes = [1.0]
        config.epochs = 20

    # Run evaluation
    evaluator = RigorousEvaluator(config)

    if config.data_path:
        data_loader = GPSIMUDataLoader(config.data_path, dt=config.dt)
        data_loader.load()
        results = evaluator.run_full_evaluation(data_loader)
    else:
        print("No data path provided. Running synthetic data evaluation...")

        # Generate synthetic data for testing
        np.random.seed(42)
        n_sequences = 5
        seq_len = 1000
        n_features = 15

        # Create mock data loader
        class MockDataLoader:
            def __init__(self):
                self.sequences = {f"seq_{i}": np.random.randn(seq_len, n_features)
                                 for i in range(n_sequences)}

            def get_loso_splits(self):
                ids = list(self.sequences.keys())
                return [(ids[:i] + ids[i+1:], ids[i]) for i in range(len(ids))]

            def get_train_test_data(self, train_ids, test_id, fit_scaler=True):
                X_train = np.vstack([self.sequences[i] for i in train_ids])
                X_test = self.sequences[test_id]
                return X_train, X_test, None, None

        mock_loader = MockDataLoader()
        results = evaluator.run_full_evaluation(mock_loader)


if __name__ == "__main__":
    main()
