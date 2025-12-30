"""
Hybrid Attack Detector with Data-Driven Routing.

Routes samples to whichever detector (baseline or PINN) performs better,
learned from training data. This achieves best-of-both-worlds performance:
- PINN for actuator attacks (control_hijack: 22% -> 99.9%)
- Baseline for temporal/sensor attacks (keeps 30% vs PINN's 16%)

Expected overall improvement: 74% -> 78-80%
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .enhanced_detector import EnhancedAttackDetector, MultiScaleFeatureExtractor


@dataclass
class HybridDetectionResult:
    """Result from hybrid detection."""
    is_attack: bool
    probability: float
    routing_score: float  # P(PINN is better)
    detector_used: str    # 'baseline', 'pinn', or 'hybrid'
    confidence: float


class RoutingClassifier:
    """
    Predicts whether PINN detector will outperform baseline for a given sample.

    Trained on validation data where we know which detector was correct.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        self.feature_importances_ = None

    def train(
        self,
        features: np.ndarray,
        pinn_better_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the routing classifier.

        Args:
            features: [N, n_features] feature vectors
            pinn_better_labels: [N] binary labels where 1 = PINN was correct and baseline wrong

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            self.is_trained = True
            return {}

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Handle class imbalance - PINN is better on fewer samples
        class_counts = np.bincount(pinn_better_labels.astype(int))
        if len(class_counts) < 2:
            # All samples favor one detector
            self.is_trained = True
            self.default_prediction = float(pinn_better_labels[0]) if len(pinn_better_labels) > 0 else 0.0
            return {"note": "All samples favor same detector"}

        class_weight = {0: 1.0, 1: class_counts[0] / (class_counts[1] + 1e-6)}

        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        self.classifier.fit(X_scaled, pinn_better_labels)

        self.feature_importances_ = self.classifier.feature_importances_
        self.is_trained = True

        # Training metrics
        preds = self.classifier.predict(X_scaled)
        accuracy = np.mean(preds == pinn_better_labels)
        pinn_better_rate = np.mean(pinn_better_labels)

        return {
            "accuracy": float(accuracy),
            "pinn_better_rate": float(pinn_better_rate),
            "n_samples": len(features),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict probability that PINN will be better for each sample.

        Args:
            features: [N, n_features] feature vectors

        Returns:
            [N] probabilities in [0, 1]
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return np.zeros(len(features))

        if self.classifier is None:
            # All samples favor same detector
            return np.full(len(features), getattr(self, 'default_prediction', 0.0))

        X_scaled = self.scaler.transform(features)
        probs = self.classifier.predict_proba(X_scaled)

        # Handle case where classifier only saw one class
        if probs.shape[1] == 1:
            return np.zeros(len(features))

        return probs[:, 1]  # P(PINN is better)


class HybridAttackDetector:
    """
    Hybrid detector that routes samples to the best-performing detector.

    Uses a data-driven routing classifier to predict which detector
    (baseline or PINN) will perform better for each sample, then
    combines predictions with learned weights.

    Usage:
        # Create detector
        detector = HybridAttackDetector(predictor=pinn_predictor)

        # Train (automatically trains both detectors + router)
        detector.train(normal_states, attack_states, normal_controls, attack_controls)

        # Predict
        preds, probs = detector.predict_batch(states, controls)
    """

    def __init__(
        self,
        predictor: Optional[Any] = None,
        target_recall: float = 0.90,
        n_estimators: int = 200,
        routing_threshold: float = 0.5,
    ):
        """
        Initialize hybrid detector.

        Args:
            predictor: PINN predictor for physics-based features
            target_recall: Target recall for threshold tuning
            n_estimators: Number of trees for classifiers
            routing_threshold: Threshold for using PINN (if routing_score > this, use PINN)
        """
        self.predictor = predictor
        self.target_recall = target_recall
        self.n_estimators = n_estimators
        self.routing_threshold = routing_threshold

        # Create both detectors
        self.baseline_detector = EnhancedAttackDetector(
            predictor=None,  # No PINN
            target_recall=target_recall,
            n_estimators=n_estimators,
        )

        self.pinn_detector = EnhancedAttackDetector(
            predictor=predictor,  # With PINN
            target_recall=target_recall,
            n_estimators=n_estimators,
        ) if predictor is not None else None

        # Routing classifier
        self.router = RoutingClassifier()

        # Feature extractor for routing (same as baseline)
        self.feature_extractor = MultiScaleFeatureExtractor(use_control_features=True)

        # Training state
        self.is_trained = False
        self.threshold = 0.5
        self.training_metrics = {}

    def train(
        self,
        normal_states: np.ndarray,
        attack_states: np.ndarray,
        normal_controls: Optional[np.ndarray] = None,
        attack_controls: Optional[np.ndarray] = None,
        validation_split: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Train the hybrid detector with data-driven routing.

        Phase 1: Train baseline detector on all data
        Phase 2: Train PINN detector on all data
        Phase 3: Evaluate both on validation set, create routing labels
        Phase 4: Train routing classifier

        Args:
            normal_states: [N, 12] normal flight data
            attack_states: [M, 12] attack data
            normal_controls: [N, 4] controls for normal (optional)
            attack_controls: [M, 4] controls for attack (optional)
            validation_split: Fraction of data to use for routing training

        Returns:
            Training metrics including per-detector and routing metrics
        """
        print("=" * 70)
        print("HYBRID ATTACK DETECTOR TRAINING")
        print("=" * 70)

        # Split data into train and validation for routing
        n_normal = len(normal_states)
        n_attack = len(attack_states)
        n_val_normal = int(n_normal * validation_split)
        n_val_attack = int(n_attack * validation_split)

        # Training data
        train_normal = normal_states[:-n_val_normal] if n_val_normal > 0 else normal_states
        train_attack = attack_states[:-n_val_attack] if n_val_attack > 0 else attack_states
        train_normal_ctrl = normal_controls[:-n_val_normal] if normal_controls is not None and n_val_normal > 0 else normal_controls
        train_attack_ctrl = attack_controls[:-n_val_attack] if attack_controls is not None and n_val_attack > 0 else attack_controls

        # Validation data for routing
        val_normal = normal_states[-n_val_normal:] if n_val_normal > 0 else normal_states[:1000]
        val_attack = attack_states[-n_val_attack:] if n_val_attack > 0 else attack_states[:1000]
        val_normal_ctrl = normal_controls[-n_val_normal:] if normal_controls is not None and n_val_normal > 0 else None
        val_attack_ctrl = attack_controls[-n_val_attack:] if attack_controls is not None and n_val_attack > 0 else None

        # Phase 1: Train baseline detector
        print("\n[1/4] Training Baseline Detector (no PINN)...")
        baseline_metrics = self.baseline_detector.train(
            train_normal, train_attack, train_normal_ctrl, train_attack_ctrl
        )
        print(f"  Baseline - Precision: {baseline_metrics.get('precision', 0)*100:.1f}%, "
              f"Recall: {baseline_metrics.get('recall', 0)*100:.1f}%")

        # Phase 2: Train PINN detector (if available)
        pinn_metrics = {}
        if self.pinn_detector is not None:
            print("\n[2/4] Training PINN Detector...")
            pinn_metrics = self.pinn_detector.train(
                train_normal, train_attack, train_normal_ctrl, train_attack_ctrl
            )
            print(f"  PINN - Precision: {pinn_metrics.get('precision', 0)*100:.1f}%, "
                  f"Recall: {pinn_metrics.get('recall', 0)*100:.1f}%")
        else:
            print("\n[2/4] Skipping PINN Detector (no predictor provided)")

        # Phase 3: Create routing labels from validation performance
        print("\n[3/4] Creating Routing Labels from Validation Set...")
        routing_features, routing_labels = self._create_routing_labels(
            val_normal, val_attack, val_normal_ctrl, val_attack_ctrl
        )

        # Phase 4: Train routing classifier
        print("\n[4/4] Training Routing Classifier...")
        if len(routing_features) > 0:
            routing_metrics = self.router.train(routing_features, routing_labels)
            print(f"  Routing Accuracy: {routing_metrics.get('accuracy', 0)*100:.1f}%")
            print(f"  PINN Better Rate: {routing_metrics.get('pinn_better_rate', 0)*100:.1f}%")
        else:
            routing_metrics = {"note": "No routing samples available"}
            print("  No routing samples available")

        # Use baseline threshold as default
        self.threshold = self.baseline_detector.threshold
        self.is_trained = True

        self.training_metrics = {
            "baseline": baseline_metrics,
            "pinn": pinn_metrics,
            "routing": routing_metrics,
        }

        print("\n" + "=" * 70)
        print("HYBRID DETECTOR TRAINING COMPLETE")
        print("=" * 70)

        return self.training_metrics

    def _create_routing_labels(
        self,
        val_normal: np.ndarray,
        val_attack: np.ndarray,
        val_normal_ctrl: Optional[np.ndarray],
        val_attack_ctrl: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create routing labels by comparing detector performance.

        Returns:
            features: [N, n_features] feature vectors for routing training
            labels: [N] binary labels where 1 = PINN was better
        """
        if self.pinn_detector is None:
            # No PINN detector, all samples use baseline
            return np.array([]), np.array([])

        window_size = self.feature_extractor.long_window

        all_features = []
        all_labels = []

        # Process attack data (where differences matter most)
        if len(val_attack) > window_size:
            # Get predictions from both detectors
            preds_base, probs_base = self.baseline_detector.predict_batch(
                val_attack, val_attack_ctrl
            )
            preds_pinn, probs_pinn = self.pinn_detector.predict_batch(
                val_attack, val_attack_ctrl
            )

            # Ground truth: all attack data should be detected (label=1)
            labels_true = np.ones(len(preds_base))

            # Which detector was correct?
            correct_base = (preds_base == labels_true)
            correct_pinn = (preds_pinn == labels_true)

            # PINN is better when: PINN correct AND baseline wrong
            pinn_better = correct_pinn & ~correct_base

            # Extract features for routing
            for i in range(window_size, len(val_attack)):
                window = val_attack[i-window_size:i]
                ctrl_window = val_attack_ctrl[i-window_size:i] if val_attack_ctrl is not None else None

                feat = self.feature_extractor.extract(window, ctrl_window)
                all_features.append(feat)

                # Align with prediction index
                pred_idx = i - window_size
                if pred_idx < len(pinn_better):
                    all_labels.append(int(pinn_better[pred_idx]))

        # Process normal data (for FPR balancing)
        if len(val_normal) > window_size:
            preds_base_n, _ = self.baseline_detector.predict_batch(
                val_normal, val_normal_ctrl
            )
            preds_pinn_n, _ = self.pinn_detector.predict_batch(
                val_normal, val_normal_ctrl
            )

            # Ground truth: all normal data should be not detected (label=0)
            labels_true_n = np.zeros(len(preds_base_n))

            correct_base_n = (preds_base_n == labels_true_n)
            correct_pinn_n = (preds_pinn_n == labels_true_n)

            # PINN is better when: PINN correct AND baseline wrong
            pinn_better_n = correct_pinn_n & ~correct_base_n

            # Sample a subset of normal data for balance
            n_sample = min(len(preds_base_n), len(all_labels) if all_labels else 1000)
            indices = np.random.choice(len(val_normal) - window_size, size=n_sample, replace=False)

            for idx in indices:
                i = idx + window_size
                window = val_normal[i-window_size:i]
                ctrl_window = val_normal_ctrl[i-window_size:i] if val_normal_ctrl is not None else None

                feat = self.feature_extractor.extract(window, ctrl_window)
                all_features.append(feat)

                pred_idx = idx
                if pred_idx < len(pinn_better_n):
                    all_labels.append(int(pinn_better_n[pred_idx]))

        if not all_features:
            return np.array([]), np.array([])

        features = np.array(all_features)
        labels = np.array(all_labels)

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        print(f"  Created {len(features)} routing samples")
        print(f"  PINN better on {np.mean(labels)*100:.1f}% of samples")

        return features, labels

    def predict_batch(
        self,
        states: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction with data-driven routing.

        Args:
            states: [N, 12] state sequence
            controls: [N, 4] control sequence (optional)

        Returns:
            preds: [N-window] binary predictions
            probs: [N-window] attack probabilities
        """
        if not self.is_trained:
            n_out = max(0, len(states) - self.feature_extractor.long_window)
            return np.zeros(n_out), np.zeros(n_out)

        window_size = self.feature_extractor.long_window

        # Get predictions from baseline
        preds_base, probs_base = self.baseline_detector.predict_batch(states, controls)

        if self.pinn_detector is None or not self.pinn_detector.is_trained:
            # No PINN, use baseline only
            return preds_base, probs_base

        # Get predictions from PINN
        preds_pinn, probs_pinn = self.pinn_detector.predict_batch(states, controls)

        # Extract features for routing
        routing_features = []
        for i in range(window_size, len(states)):
            window = states[i-window_size:i]
            ctrl_window = controls[i-window_size:i] if controls is not None else None
            feat = self.feature_extractor.extract(window, ctrl_window)
            routing_features.append(feat)

        if not routing_features:
            return preds_base, probs_base

        routing_features = np.array(routing_features)
        routing_features = np.nan_to_num(routing_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Get routing probabilities
        pinn_preference = self.router.predict(routing_features)

        # Weighted combination
        # pinn_preference = P(PINN is better)
        # final_prob = (1 - pinn_preference) * baseline_prob + pinn_preference * pinn_prob
        final_probs = (1 - pinn_preference) * probs_base + pinn_preference * probs_pinn
        final_preds = (final_probs >= self.threshold).astype(int)

        return final_preds, final_probs

    def evaluate(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on labeled data.

        Args:
            states: [N, 12] state sequence
            labels: [N] ground truth labels
            controls: [N, 4] control sequence (optional)

        Returns:
            Evaluation metrics
        """
        preds, probs = self.predict_batch(states, controls)

        window_size = self.feature_extractor.long_window
        aligned_labels = labels[window_size:window_size + len(preds)]

        if len(preds) != len(aligned_labels):
            min_len = min(len(preds), len(aligned_labels))
            preds = preds[:min_len]
            probs = probs[:min_len]
            aligned_labels = aligned_labels[:min_len]

        tp = np.sum((preds == 1) & (aligned_labels == 1))
        fp = np.sum((preds == 1) & (aligned_labels == 0))
        fn = np.sum((preds == 0) & (aligned_labels == 1))
        tn = np.sum((preds == 0) & (aligned_labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
        }

    def save(self, path: str):
        """Save detector to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save both detectors
        self.baseline_detector.save(str(path / "baseline_detector.pkl"))
        if self.pinn_detector is not None:
            self.pinn_detector.save(str(path / "pinn_detector.pkl"))

        # Save router
        with open(path / "router.pkl", "wb") as f:
            pickle.dump({
                "router": self.router,
                "threshold": self.threshold,
                "routing_threshold": self.routing_threshold,
                "training_metrics": self.training_metrics,
            }, f)

    def load(self, path: str):
        """Load detector from disk."""
        path = Path(path)

        # Load both detectors
        self.baseline_detector.load(str(path / "baseline_detector.pkl"))
        if (path / "pinn_detector.pkl").exists():
            if self.pinn_detector is None:
                self.pinn_detector = EnhancedAttackDetector(predictor=self.predictor)
            self.pinn_detector.load(str(path / "pinn_detector.pkl"))

        # Load router
        with open(path / "router.pkl", "rb") as f:
            data = pickle.load(f)
            self.router = data["router"]
            self.threshold = data["threshold"]
            self.routing_threshold = data.get("routing_threshold", 0.5)
            self.training_metrics = data.get("training_metrics", {})

        self.is_trained = True
