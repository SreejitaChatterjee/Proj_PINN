"""
Hardened Training Loop

Integrates hard negative mining and domain randomization
for improved worst-case robustness.

Key techniques:
1. Iterative hard negative mining
2. Domain randomization augmentation
3. Multi-task learning (detection + attribution)
4. Curriculum learning (easy -> hard attacks)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm

from hard_negatives import HardNegativeGenerator, DomainRandomizer, AdversarialAttackGenerator
from attribution import MultiTaskDetector, MultiTaskLoss
from model import CNNGRUDetector


@dataclass
class HardenedTrainingConfig:
    """Configuration for hardened training."""
    # Basic training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 15

    # Hard negative mining
    mine_hard_negatives: bool = True
    mining_frequency: int = 10  # Mine every N epochs
    n_hard_negative_attempts: int = 20
    hard_negative_ratio: float = 0.2  # Fraction of batch

    # Domain randomization
    use_domain_randomization: bool = True
    augment_probability: float = 0.5

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 4
    epochs_per_stage: int = 25

    # Multi-task
    use_attribution: bool = True
    attack_loss_weight: float = 0.3

    # Adversarial training
    use_adversarial_training: bool = False
    adversarial_epsilon: float = 0.1
    adversarial_ratio: float = 0.1


class CurriculumScheduler:
    """
    Curriculum learning scheduler.

    Gradually increases attack difficulty during training.
    """

    def __init__(
        self,
        n_stages: int = 4,
        epochs_per_stage: int = 25
    ):
        self.n_stages = n_stages
        self.epochs_per_stage = epochs_per_stage

        # Define attack difficulty progression
        self.magnitude_schedule = [0.5, 1.0, 1.5, 2.0]  # Increasing magnitude
        self.attack_types_schedule = [
            ['bias'],  # Stage 1: Simple
            ['bias', 'noise'],  # Stage 2: Add noise
            ['bias', 'noise', 'drift'],  # Stage 3: Add drift
            ['bias', 'noise', 'drift', 'coordinated', 'intermittent']  # Stage 4: All
        ]

    def get_stage(self, epoch: int) -> int:
        """Get current curriculum stage."""
        return min(epoch // self.epochs_per_stage, self.n_stages - 1)

    def get_attack_config(self, epoch: int) -> Tuple[float, List[str]]:
        """
        Get attack configuration for current epoch.

        Returns:
            magnitude: Attack magnitude
            attack_types: List of allowed attack types
        """
        stage = self.get_stage(epoch)
        return self.magnitude_schedule[stage], self.attack_types_schedule[stage]


class HardenedTrainer:
    """
    Hardened training loop with robustness techniques.
    """

    def __init__(
        self,
        config: HardenedTrainingConfig,
        output_dir: str = './experiments/hardened'
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize components
        self.hard_neg_generator = HardNegativeGenerator(seed=42)
        self.domain_randomizer = DomainRandomizer(seed=42)
        self.curriculum = CurriculumScheduler(
            n_stages=config.curriculum_stages,
            epochs_per_stage=config.epochs_per_stage
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall_1pct': [],
            'val_recall_5pct': [],
            'hard_negatives_added': [],
            'curriculum_stage': []
        }

    def create_model(self, input_dim: int) -> nn.Module:
        """Create model based on config."""
        if self.config.use_attribution:
            return MultiTaskDetector(input_dim=input_dim).to(self.device)
        else:
            return CNNGRUDetector(input_dim=input_dim).to(self.device)

    def augment_batch(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply domain randomization to batch."""
        if not self.config.use_domain_randomization:
            return features, labels

        augmented = self.domain_randomizer.augment_batch(
            features,
            augment_prob=self.config.augment_probability
        )

        return augmented, labels

    def mine_hard_negatives(
        self,
        model: nn.Module,
        clean_data: np.ndarray,
        epoch: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mine hard negative examples that evade current detector.

        Args:
            model: Current model
            clean_data: Clean data to attack
            epoch: Current epoch (for curriculum)

        Returns:
            hard_data: Features of hard negatives
            hard_labels: Labels (all 1s for attacks)
        """
        if not self.config.mine_hard_negatives:
            return np.array([]), np.array([])

        # Create detector function
        model.eval()

        def detector_fn(data):
            with torch.no_grad():
                x = torch.tensor(data, dtype=torch.float32, device=self.device)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                if self.config.use_attribution:
                    output, _, _, _ = model(x)
                else:
                    output, _ = model(x)
                return torch.sigmoid(output).squeeze().cpu().numpy()

        # Get curriculum-appropriate attacks
        magnitude, attack_types = self.curriculum.get_attack_config(epoch)

        # Find evasive attacks
        evasive = self.hard_neg_generator.find_evasive_attacks(
            clean_data,
            detector_fn,
            n_attempts=self.config.n_hard_negative_attempts
        )

        if evasive:
            # Collect hard negative data
            hard_data = []
            hard_labels = []

            for attack in evasive[:5]:  # Top 5 most evasive
                hard_data.append(attack.data)
                hard_labels.append(attack.labels)

            return np.vstack(hard_data), np.concatenate(hard_labels)

        return np.array([]), np.array([])

    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        attack_generator: Optional[Callable] = None
    ) -> nn.Module:
        """
        Run hardened training.

        Args:
            train_data: Dict with 'features', 'labels', optionally 'attack_types'
            val_data: Dict with 'features', 'labels'
            attack_generator: Optional function to generate attacks for curriculum

        Returns:
            Trained model
        """
        input_dim = train_data['features'].shape[1]
        model = self.create_model(input_dim)

        # Setup optimization
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        if self.config.use_attribution:
            criterion = MultiTaskLoss(attack_weight=self.config.attack_loss_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Hard negative pool
        hard_negative_pool = {'features': [], 'labels': []}

        for epoch in range(self.config.epochs):
            # Get curriculum stage
            stage = self.curriculum.get_stage(epoch)
            self.history['curriculum_stage'].append(stage)

            # Mine hard negatives periodically
            if self.config.mine_hard_negatives and epoch > 0 and epoch % self.config.mining_frequency == 0:
                print(f"  Mining hard negatives...")
                hard_features, hard_labels = self.mine_hard_negatives(
                    model,
                    train_data['features'][:1000],  # Subsample for efficiency
                    epoch
                )
                if len(hard_features) > 0:
                    hard_negative_pool['features'].append(hard_features)
                    hard_negative_pool['labels'].append(hard_labels)
                    n_hard = len(hard_features)
                    print(f"  Added {n_hard} hard negatives")
                    self.history['hard_negatives_added'].append(n_hard)
                else:
                    self.history['hard_negatives_added'].append(0)

            # Prepare training data
            train_features = train_data['features'].copy()
            train_labels = train_data['labels'].copy()

            # Add hard negatives
            if hard_negative_pool['features']:
                hard_feat = np.vstack(hard_negative_pool['features'])
                hard_lab = np.concatenate(hard_negative_pool['labels'])

                # Sample subset
                n_hard = int(len(train_features) * self.config.hard_negative_ratio)
                if len(hard_feat) > n_hard:
                    idx = np.random.choice(len(hard_feat), n_hard, replace=False)
                    hard_feat = hard_feat[idx]
                    hard_lab = hard_lab[idx]

                train_features = np.vstack([train_features, hard_feat])
                train_labels = np.concatenate([train_labels, hard_lab])

            # Apply domain randomization
            train_features, train_labels = self.augment_batch(train_features, train_labels)

            # Create data loader
            dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32)
            )
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            # Training epoch
            model.train()
            train_loss = 0

            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device).unsqueeze(1)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                if self.config.use_attribution:
                    anomaly_out, attack_out, sensor_out, _ = model(batch_features)
                    loss, _ = criterion(
                        anomaly_out, attack_out, sensor_out,
                        batch_labels.unsqueeze(1),
                        None  # No attack type labels in basic training
                    )
                else:
                    outputs, _ = model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(loader)
            self.history['train_loss'].append(train_loss)

            # Validation
            val_loss, val_metrics = self.evaluate(model, val_data, criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_recall_1pct'].append(val_metrics['recall_1pct'])
            self.history['val_recall_5pct'].append(val_metrics['recall_5pct'])

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch} (stage {stage}): "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"R@1%={val_metrics['recall_1pct']:.3f}, "
                      f"R@5%={val_metrics['recall_5pct']:.3f}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model and history
        self.save_model(model, 'hardened_model.pth')
        self.save_history()

        return model

    def evaluate(
        self,
        model: nn.Module,
        val_data: Dict[str, np.ndarray],
        criterion: nn.Module
    ) -> Tuple[float, Dict]:
        """Evaluate model on validation data."""
        model.eval()

        with torch.no_grad():
            features = torch.tensor(
                val_data['features'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if self.config.use_attribution:
                anomaly_out, attack_out, sensor_out, _ = model(features)
                scores = torch.sigmoid(anomaly_out).squeeze().cpu().numpy()
                val_labels_t = torch.tensor(
                    val_data['labels'], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                loss, _ = criterion(
                    anomaly_out, attack_out, sensor_out, val_labels_t, None
                )
                val_loss = loss.item()
            else:
                outputs, _ = model(features)
                scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
                val_loss = criterion(
                    torch.tensor(scores),
                    torch.tensor(val_data['labels'], dtype=torch.float32)
                ).item()

        labels = val_data['labels']

        # Compute metrics
        normal_mask = labels == 0
        attack_mask = labels == 1

        normal_scores = scores[normal_mask]
        attack_scores = scores[attack_mask]

        metrics = {}
        if len(normal_scores) > 0 and len(attack_scores) > 0:
            threshold_1pct = np.percentile(normal_scores, 99)
            threshold_5pct = np.percentile(normal_scores, 95)
            metrics['recall_1pct'] = float(np.mean(attack_scores > threshold_1pct))
            metrics['recall_5pct'] = float(np.mean(attack_scores > threshold_5pct))
        else:
            metrics['recall_1pct'] = 0.0
            metrics['recall_5pct'] = 0.0

        return val_loss, metrics

    def save_model(self, model: nn.Module, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def save_history(self):
        """Save training history."""
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def run_ablation_hardening(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    output_dir: str = './experiments/ablation_hardening'
) -> Dict:
    """
    Run ablation study on hardening techniques.

    Tests impact of:
    1. Hard negative mining
    2. Domain randomization
    3. Curriculum learning
    4. Multi-task attribution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline config
    base_config = HardenedTrainingConfig(
        epochs=50,
        mine_hard_negatives=False,
        use_domain_randomization=False,
        use_curriculum=False,
        use_attribution=False
    )

    ablation_configs = {
        'baseline': base_config,
        '+hard_negatives': HardenedTrainingConfig(
            epochs=50,
            mine_hard_negatives=True,
            use_domain_randomization=False,
            use_curriculum=False,
            use_attribution=False
        ),
        '+domain_rand': HardenedTrainingConfig(
            epochs=50,
            mine_hard_negatives=False,
            use_domain_randomization=True,
            use_curriculum=False,
            use_attribution=False
        ),
        '+curriculum': HardenedTrainingConfig(
            epochs=50,
            mine_hard_negatives=False,
            use_domain_randomization=False,
            use_curriculum=True,
            use_attribution=False
        ),
        '+attribution': HardenedTrainingConfig(
            epochs=50,
            mine_hard_negatives=False,
            use_domain_randomization=False,
            use_curriculum=False,
            use_attribution=True
        ),
        'full_hardening': HardenedTrainingConfig(
            epochs=50,
            mine_hard_negatives=True,
            use_domain_randomization=True,
            use_curriculum=True,
            use_attribution=True
        ),
    }

    results = {}

    for name, config in ablation_configs.items():
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

        trainer = HardenedTrainer(config, output_dir=str(output_dir / name))
        model = trainer.train(train_data, val_data)

        # Final evaluation
        _, final_metrics = trainer.evaluate(
            model, val_data, nn.BCEWithLogitsLoss()
        )

        results[name] = {
            'recall_1pct': final_metrics['recall_1pct'],
            'recall_5pct': final_metrics['recall_5pct'],
            'config': {
                'hard_negatives': config.mine_hard_negatives,
                'domain_rand': config.use_domain_randomization,
                'curriculum': config.use_curriculum,
                'attribution': config.use_attribution
            }
        }

        print(f"\n{name} results:")
        print(f"  Recall@1%FPR: {final_metrics['recall_1pct']:.3f}")
        print(f"  Recall@5%FPR: {final_metrics['recall_5pct']:.3f}")

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Test hardened training
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    n_train = 5000
    n_val = 1000
    d = 100

    train_features = np.random.randn(n_train, d)
    train_labels = np.zeros(n_train)  # All normal for training

    val_features = np.random.randn(n_val, d)
    val_labels = np.concatenate([
        np.zeros(n_val // 2),
        np.ones(n_val // 2)
    ])
    # Make attacks slightly different
    val_features[n_val // 2:] += 0.5

    train_data = {'features': train_features, 'labels': train_labels}
    val_data = {'features': val_features, 'labels': val_labels}

    # Quick test
    config = HardenedTrainingConfig(
        epochs=20,
        mine_hard_negatives=True,
        mining_frequency=5,
        use_domain_randomization=True,
        use_curriculum=True,
        use_attribution=False  # Faster without multi-task
    )

    trainer = HardenedTrainer(config, output_dir='./experiments/test_hardened')
    model = trainer.train(train_data, val_data)

    print("\nTraining complete!")
    print(f"Final validation metrics:")
    print(f"  Recall@1%FPR history: {trainer.history['val_recall_1pct'][-5:]}")
    print(f"  Recall@5%FPR history: {trainer.history['val_recall_5pct'][-5:]}")
