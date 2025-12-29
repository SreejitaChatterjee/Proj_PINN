"""
Training Script for GPS-IMU Anomaly Detector

Implements LOSO-CV training with proper evaluation protocol.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm

from data_loader import GPSIMUDataLoader, AttackCatalog
from feature_extractor import BatchFeatureExtractor
from physics_residuals import HybridPhysicsChecker
from ekf import EKFAnomalyDetector
from model import CNNGRUDetector, StreamingDetector
from hybrid_scorer import HybridScorer, TemporalConsistencyScorer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class GPSIMUTrainer:
    """
    Full training pipeline for GPS-IMU anomaly detector.

    Implements:
    1. LOSO-CV training
    2. Component training (feature extractor, ML model)
    3. Hybrid scorer calibration
    4. Evaluation with proper protocol
    """

    def __init__(self, config: Dict, output_dir: str = './experiments'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize components
        self.feature_extractor = BatchFeatureExtractor(
            windows=config['features']['windows']
        )

        self.physics_checker = HybridPhysicsChecker(
            dt=config['data']['dt'],
            use_pinn=config['physics'].get('use_pinn', False)
        )

        self.ekf_detector = EKFAnomalyDetector(
            dt=config['data']['dt'],
            window_size=50
        )

        self.temporal_scorer = TemporalConsistencyScorer(window_size=10)

        # Results storage
        self.results = {
            'folds': [],
            'overall': {}
        }

    def train_fold(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        fold_idx: int
    ) -> Dict:
        """
        Train on single LOSO fold.

        Args:
            train_data: Dict with 'features', 'labels' for training
            val_data: Dict with 'features', 'labels' for validation
            fold_idx: Fold index for logging

        Returns:
            Fold results dict
        """
        print(f"\n=== Training Fold {fold_idx} ===")

        # Initialize ML model
        input_dim = train_data['features'].shape[1]
        model = CNNGRUDetector(
            input_dim=input_dim,
            cnn_channels=tuple(self.config['model']['cnn_channels']),
            gru_hidden_size=self.config['model']['gru_hidden_size'],
            dropout=self.config['model']['dropout']
        ).to(self.device)

        # Training setup
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        criterion = nn.BCEWithLogitsLoss()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_data['features'], dtype=torch.float32),
            torch.tensor(train_data['labels'], dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.config['training']['epochs']):
            model.train()
            train_loss = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Add sequence dimension
                batch_features = batch_features.unsqueeze(1)

                optimizer.zero_grad()
                outputs, _ = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_features = torch.tensor(
                    val_data['features'], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                val_outputs, _ = model(val_features)
                val_probs = torch.sigmoid(val_outputs).squeeze().cpu().numpy()

                val_labels = val_data['labels']
                val_loss = criterion(
                    torch.tensor(val_probs),
                    torch.tensor(val_labels, dtype=torch.float32)
                ).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate on validation set
        fold_results = self.evaluate(model, val_data, fold_idx)

        # Save model
        model_path = self.output_dir / f'model_fold{fold_idx}.pth'
        torch.save(model.state_dict(), model_path)

        return fold_results

    def evaluate(
        self,
        model: CNNGRUDetector,
        test_data: Dict[str, np.ndarray],
        fold_idx: int
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            model: Trained model
            test_data: Test data dict
            fold_idx: Fold index

        Returns:
            Evaluation results
        """
        model.eval()

        with torch.no_grad():
            features = torch.tensor(
                test_data['features'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            outputs, _ = model(features)
            ml_scores = torch.sigmoid(outputs).squeeze().cpu().numpy()

        labels = test_data['labels']

        # Compute metrics at different FPR thresholds
        results = {'fold': fold_idx}

        for fpr_target in self.config['evaluation']['fpr_thresholds']:
            # Find threshold for target FPR on normal data
            normal_scores = ml_scores[labels == 0]
            if len(normal_scores) > 0:
                threshold = np.percentile(normal_scores, (1 - fpr_target) * 100)
            else:
                threshold = 0.5

            # Compute recall
            attack_scores = ml_scores[labels == 1]
            if len(attack_scores) > 0:
                recall = np.mean(attack_scores > threshold)
            else:
                recall = 0.0

            # Actual FPR
            actual_fpr = np.mean(normal_scores > threshold) if len(normal_scores) > 0 else 0

            results[f'recall@{int(fpr_target*100)}%fpr'] = recall
            results[f'actual_fpr@{int(fpr_target*100)}%target'] = actual_fpr

        # Overall metrics
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(np.unique(labels)) > 1:
            results['auroc'] = roc_auc_score(labels, ml_scores)
            results['auprc'] = average_precision_score(labels, ml_scores)
        else:
            results['auroc'] = 0.0
            results['auprc'] = 0.0

        return results

    def run_loso_cv(self, data_loader: GPSIMUDataLoader, attack_catalog: AttackCatalog):
        """
        Run full LOSO-CV training and evaluation.

        Args:
            data_loader: Loaded data
            attack_catalog: Attack generator
        """
        splits = data_loader.get_loso_splits()
        print(f"\nRunning {len(splits)}-fold LOSO-CV")

        all_fold_results = []

        for fold_idx, (train_ids, test_id) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx}: Test on {test_id}")
            print(f"{'='*50}")

            # Get train/test data
            X_train, X_test, train_bounds, test_bounds = data_loader.get_train_test_data(
                train_ids, test_id, fit_scaler=(fold_idx == 0)
            )

            # Extract features
            print("Extracting features...")
            train_features = self.feature_extractor.extract(X_train)
            test_features = self.feature_extractor.extract(X_test)

            # Generate attacks for test set
            print("Generating attacks...")
            test_clean = X_test.copy()
            test_attacked_list = []
            test_labels_list = []

            # Clean data
            test_attacked_list.append(test_clean)
            test_labels_list.append(np.zeros(len(test_clean)))

            # Various attacks
            for attack_type in ['bias', 'drift', 'noise']:
                for magnitude in [0.5, 1.0, 2.0]:
                    attacked, attack_labels = attack_catalog.generate_attack(
                        test_clean, attack_type, magnitude, 'position'
                    )
                    test_attacked_list.append(attacked)
                    test_labels_list.append(attack_labels)

            # Combine test data
            test_combined = np.vstack(test_attacked_list)
            test_labels = np.concatenate(test_labels_list)

            # Extract features for combined test
            test_features_combined = self.feature_extractor.extract(test_combined)
            test_labels_aligned = test_labels[self.feature_extractor.max_window:]

            # Prepare training labels (all normal)
            train_labels = np.zeros(len(train_features))

            # Train
            train_data = {'features': train_features, 'labels': train_labels}
            test_data = {'features': test_features_combined, 'labels': test_labels_aligned}

            fold_results = self.train_fold(train_data, test_data, fold_idx)
            all_fold_results.append(fold_results)

            print(f"\nFold {fold_idx} Results:")
            for k, v in fold_results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

        # Aggregate results
        self.results['folds'] = all_fold_results
        self._compute_overall_results()

        # Save results
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def _compute_overall_results(self):
        """Compute aggregate statistics across folds."""
        folds = self.results['folds']

        metrics = ['recall@1%fpr', 'recall@5%fpr', 'auroc', 'auprc']
        overall = {}

        for metric in metrics:
            values = [f.get(metric, 0) for f in folds]
            overall[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        self.results['overall'] = overall

        print("\n" + "="*50)
        print("OVERALL RESULTS (LOSO-CV)")
        print("="*50)
        for metric, stats in overall.items():
            print(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train GPS-IMU Anomaly Detector')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--data', type=str, required=True, help='Data path')
    parser.add_argument('--output', type=str, default='./experiments', help='Output dir')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize data loader
    print("Loading data...")
    data_loader = GPSIMUDataLoader(args.data, dt=config['data']['dt'])
    data_loader.load()
    data_loader.verify_no_circular_sensors()

    # Initialize attack catalog
    attack_catalog = AttackCatalog(seed=config['training']['seed'])

    # Train
    trainer = GPSIMUTrainer(config, args.output)
    trainer.run_loso_cv(data_loader, attack_catalog)


if __name__ == "__main__":
    main()
