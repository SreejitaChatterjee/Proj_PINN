"""
Validate Attack Detection on PADRE Real Sensor Data.

This script demonstrates VALID attack detection using cross-sensor
validation on PADRE's redundant sensors.

Key insight: No circular dependency because:
- 4 independent physical sensors for each modality
- Majority voting establishes ground truth
- Attacked sensor deviates from consensus
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_dynamics.data.padre_loader import (
    PADREDataLoader, CrossSensorValidator, inject_sensor_attack
)


def evaluate_detection(
    loader: PADREDataLoader,
    validator: CrossSensorValidator,
    n_files: int = 5
) -> dict:
    """
    Comprehensive evaluation of cross-sensor attack detection.
    """
    files = loader.list_files("3DR_Solo")[:n_files]
    if not files:
        return {"error": "No files found"}

    results = {
        'attack_types': ['bias', 'drift', 'noise', 'stuck'],
        'modalities': ['accel', 'gyro', 'baro'],
        'sensors': ['A', 'B', 'C', 'D'],
        'evaluations': []
    }

    for filepath in files:
        data = loader.load_file(filepath)
        filename = Path(filepath).name

        for attack_type in ['bias', 'drift', 'stuck']:
            for modality in ['accel', 'gyro']:
                for target_sensor in ['B']:  # Attack one sensor

                    # Inject attack
                    attacked_data, labels = inject_sensor_attack(
                        data,
                        sensor=target_sensor,
                        modality=modality,
                        attack_type=attack_type,
                        start_idx=data.n_samples // 4,
                        duration=data.n_samples // 2,
                        magnitude=0.3
                    )

                    # Validate
                    val_results = validator.validate_all(attacked_data)

                    # Get detection rates for the attacked modality
                    if modality == 'accel':
                        rates = val_results['accelerometer']['sensors']
                    elif modality == 'gyro':
                        rates = val_results['gyroscope']['sensors']
                    else:
                        rates = val_results['barometer']['sensors']

                    # Check detection success
                    attacked_rate = rates[target_sensor]['anomaly_ratio']
                    other_rates = [rates[s]['anomaly_ratio']
                                   for s in ['A', 'C', 'D']]
                    max_other = max(other_rates)

                    # Detection metrics
                    detected = attacked_rate > max_other
                    margin = attacked_rate - max_other

                    results['evaluations'].append({
                        'file': filename,
                        'attack_type': attack_type,
                        'modality': modality,
                        'target_sensor': target_sensor,
                        'attacked_rate': attacked_rate,
                        'max_other_rate': max_other,
                        'margin': margin,
                        'detected': detected
                    })

    return results


def compute_metrics(results: dict) -> dict:
    """Compute detection metrics from evaluation results."""
    evals = results['evaluations']

    # Overall detection rate
    n_detected = sum(1 for e in evals if e['detected'])
    n_total = len(evals)

    # By attack type
    by_attack = {}
    for attack_type in results['attack_types']:
        subset = [e for e in evals if e['attack_type'] == attack_type]
        if subset:
            by_attack[attack_type] = sum(1 for e in subset if e['detected']) / len(subset)

    # By modality
    by_modality = {}
    for modality in ['accel', 'gyro']:
        subset = [e for e in evals if e['modality'] == modality]
        if subset:
            by_modality[modality] = sum(1 for e in subset if e['detected']) / len(subset)

    # Average margin when detected
    detected_margins = [e['margin'] for e in evals if e['detected']]
    avg_margin = np.mean(detected_margins) if detected_margins else 0

    return {
        'overall_detection_rate': n_detected / n_total if n_total > 0 else 0,
        'n_detected': n_detected,
        'n_total': n_total,
        'by_attack_type': by_attack,
        'by_modality': by_modality,
        'avg_detection_margin': avg_margin
    }


def main():
    print("="*70)
    print("PADRE Cross-Sensor Attack Detection Validation")
    print("="*70)
    print("\nThis demonstrates VALID attack detection WITHOUT circular dependencies:")
    print("- 4 independent physical sensors per modality")
    print("- Attack injected into ONE sensor")
    print("- Detection via cross-sensor consensus")
    print()

    loader = PADREDataLoader()
    validator = CrossSensorValidator(threshold_sigma=2.5)

    print("Running evaluation...")
    results = evaluate_detection(loader, validator, n_files=5)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    metrics = compute_metrics(results)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nOverall Detection Rate: {metrics['overall_detection_rate']*100:.1f}%")
    print(f"  ({metrics['n_detected']}/{metrics['n_total']} attacks detected)")

    print(f"\nAverage Detection Margin: {metrics['avg_detection_margin']:.3f}")
    print("  (How much higher attacked sensor rate is vs others)")

    print("\nBy Attack Type:")
    for attack_type, rate in metrics['by_attack_type'].items():
        print(f"  {attack_type:10s}: {rate*100:.1f}%")

    print("\nBy Modality:")
    for modality, rate in metrics['by_modality'].items():
        print(f"  {modality:10s}: {rate*100:.1f}%")

    print("\n" + "="*70)
    print("FEASIBILITY ASSESSMENT")
    print("="*70)

    overall = metrics['overall_detection_rate']
    if overall >= 0.8:
        print(f"\nFEASIBILITY: HIGH ({overall*100:.0f}%)")
        print("Cross-sensor validation successfully detects attacks on real data.")
    elif overall >= 0.5:
        print(f"\nFEASIBILITY: MODERATE ({overall*100:.0f}%)")
        print("Cross-sensor validation works but needs tuning.")
    else:
        print(f"\nFEASIBILITY: LOW ({overall*100:.0f}%)")
        print("Detection approach needs significant improvement.")

    print("\nKEY ADVANTAGE: No circular dependency!")
    print("- All 4 sensors are independent physical devices")
    print("- No emulation from ground truth")
    print("- Detection based on real sensor disagreement")

    return metrics


if __name__ == "__main__":
    metrics = main()
