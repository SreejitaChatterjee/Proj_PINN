"""
GPS-IMU Anomaly Detector - Validation Run
Generates actual validated results for documentation.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Config
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print('='*60)
print('GPS-IMU ANOMALY DETECTOR - VALIDATION RUN')
print('='*60)

# Hardware info
hw = {
    'platform': platform.platform(),
    'processor': platform.processor(),
    'python': platform.python_version(),
    'pytorch': torch.__version__,
    'seed': SEED,
    'timestamp': datetime.now().isoformat()
}
print(f'Platform: {hw["platform"]}')
print(f'Processor: {hw["processor"]}')
print(f'Seed: {SEED}')

# Load data
print('\nLoading data...')
df = pd.read_csv('../data/euroc/all_sequences.csv')
print(f'Total samples: {len(df):,}')

# Use 3 sequences for train, 2 for test
seqs = df['sequence'].unique()
train_seq, test_seq = list(seqs[:3]), list(seqs[3:])
print(f'Train sequences: {train_seq}')
print(f'Test sequences: {test_seq}')

train_mask = df['sequence'].isin(train_seq)
test_mask = df['sequence'].isin(test_seq)

feature_cols = ['x','y','z','vx','vy','vz','roll','pitch','yaw','p','q','r','ax','ay','az']
X_train = df.loc[train_mask, feature_cols].values
X_test = df.loc[test_mask, feature_cols].values
print(f'Train: {len(X_train):,}, Test: {len(X_test):,}')

# Normalize (train-only scaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create sequences
SEQ_LEN = 25
def make_seq(X, L):
    return np.array([X[i:i+L] for i in range(len(X)-L+1)])

X_train_seq = make_seq(X_train, SEQ_LEN)
X_test_seq = make_seq(X_test, SEQ_LEN)
y_train = np.zeros(len(X_train_seq))
y_test_normal = np.zeros(len(X_test_seq))
print(f'Train sequences: {len(X_train_seq):,}')
print(f'Test sequences: {len(X_test_seq):,}')

# Simple CNN-GRU detector
class Detector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, 32, 3, padding=1)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.conv(x.transpose(1,2))).transpose(1,2)
        x, _ = self.gru(x)
        return torch.sigmoid(self.fc(x[:,-1,:]))

# Train
print('\nTraining detector (10 epochs)...')
model = Detector(15)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train)),
    batch_size=256, shuffle=True
)

for ep in range(10):
    total_loss = 0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if ep % 3 == 0:
        print(f'  Epoch {ep+1}: loss = {total_loss/len(loader):.4f}')

model.eval()
print('Training complete.')

# Generate attacks
print('\nGenerating attack data...')
attacks = {}
n_test = len(X_test)

# Bias attack
X_bias = X_test.copy()
X_bias[:, :3] += 0.5
attacks['bias'] = X_bias

# Drift attack
X_drift = X_test.copy()
drift = np.zeros((n_test, 3))
for i in range(1, n_test):
    drift[i] = 0.995 * drift[i-1] + np.random.randn(3) * 0.01
X_drift[:, :3] += drift
attacks['drift'] = X_drift

# Noise attack
X_noise = X_test + np.random.randn(n_test, 15) * 0.3
attacks['noise'] = X_noise

# Coordinated attack
X_coord = X_test.copy()
X_coord[:, :3] += 0.3
X_coord[:, 3:6] += 0.1
attacks['coordinated'] = X_coord

# Intermittent attack
X_int = X_test.copy()
mask = (np.arange(n_test) % 100) < 30
X_int[mask, :3] += 1.0
attacks['intermittent'] = X_int

def recall_at_fpr(y_true, scores, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr)-1)])

# Evaluate each attack
print('\nEvaluating per-attack detection...')
results = {}

for atk_name, X_atk in attacks.items():
    X_atk_seq = make_seq(X_atk, SEQ_LEN)
    y_atk = np.ones(len(X_atk_seq))

    X_eval = np.vstack([X_test_seq, X_atk_seq])
    y_eval = np.concatenate([y_test_normal, y_atk])

    with torch.no_grad():
        scores = model(torch.FloatTensor(X_eval)).squeeze().numpy()

    auroc = roc_auc_score(y_eval, scores)
    r1 = recall_at_fpr(y_eval, scores, 0.01)
    r5 = recall_at_fpr(y_eval, scores, 0.05)
    r10 = recall_at_fpr(y_eval, scores, 0.10)

    results[atk_name] = {
        'auroc': float(auroc),
        'recall_at_1pct_fpr': float(r1),
        'recall_at_5pct_fpr': float(r5),
        'recall_at_10pct_fpr': float(r10)
    }

# Print results
print('\n' + '='*70)
print('DETECTION RESULTS')
print('='*70)
print(f'{"Attack":<15} {"AUROC":>10} {"R@1%FPR":>10} {"R@5%FPR":>10} {"R@10%FPR":>10}')
print('-'*55)
for atk, res in results.items():
    print(f'{atk:<15} {res["auroc"]:>10.3f} {res["recall_at_1pct_fpr"]:>10.3f} {res["recall_at_5pct_fpr"]:>10.3f} {res["recall_at_10pct_fpr"]:>10.3f}')

mean_auroc = np.mean([r['auroc'] for r in results.values()])
worst_r5 = min(r['recall_at_5pct_fpr'] for r in results.values())
worst_atk = min(results.keys(), key=lambda a: results[a]['recall_at_5pct_fpr'])
print('-'*55)
print(f'Mean AUROC: {mean_auroc:.3f}')
print(f'Worst-case Recall@5%FPR: {worst_r5:.3f} ({worst_atk})')

# Latency benchmark
print('\n' + '='*70)
print('LATENCY BENCHMARK')
print('='*70)
x_test = torch.randn(1, SEQ_LEN, 15)
for _ in range(100): model(x_test)

times = []
for _ in range(1000):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x_test)
    times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
lat = {
    'mean_ms': float(np.mean(times)),
    'p50_ms': float(np.percentile(times, 50)),
    'p95_ms': float(np.percentile(times, 95)),
    'p99_ms': float(np.percentile(times, 99))
}
print(f'Mean:  {lat["mean_ms"]:.3f} ms')
print(f'P50:   {lat["p50_ms"]:.3f} ms')
print(f'P95:   {lat["p95_ms"]:.3f} ms')
print(f'P99:   {lat["p99_ms"]:.3f} ms')

target_5ms = lat['p99_ms'] < 5.0
print(f'Target <5ms: {"PASS" if target_5ms else "FAIL"}')

# Model size
params = sum(p.numel() for p in model.parameters())
size_mb = params * 4 / 1024 / 1024
print(f'\nModel: {params:,} params ({size_mb:.3f} MB)')
print(f'Target <1MB: {"PASS" if size_mb < 1.0 else "FAIL"}')

# Save
output = {
    'hardware': hw,
    'config': {
        'seed': SEED,
        'epochs': 10,
        'seq_len': SEQ_LEN,
        'batch_size': 256,
        'train_sequences': train_seq,
        'test_sequences': test_seq
    },
    'data': {
        'total_samples': len(df),
        'train_samples': int(train_mask.sum()),
        'test_samples': int(test_mask.sum())
    },
    'per_attack': results,
    'overall': {
        'mean_auroc': float(mean_auroc),
        'worst_recall_at_5pct_fpr': float(worst_r5),
        'worst_attack': worst_atk
    },
    'latency': lat,
    'model': {
        'params': params,
        'size_mb': float(size_mb),
        'architecture': 'CNN(32)-GRU(32)-FC(1)'
    },
    'targets': {
        'latency_5ms': target_5ms,
        'model_1mb': size_mb < 1.0
    }
}

Path('results').mkdir(exist_ok=True)
with open('results/validated_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Save model
torch.save(model.state_dict(), 'models/validated_detector.pth')
print('\n' + '='*70)
print('VALIDATED RESULTS SAVED')
print('='*70)
print('Results: results/validated_results.json')
print('Model: models/validated_detector.pth')
print('\nDONE!')
