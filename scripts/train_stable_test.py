"""Quick test of stable PINN training - 20 epochs"""
import sys
sys.path.append('.')
from train_stable import train_stable_pinn

if __name__ == "__main__":
    # Quick 20-epoch test
    train_stable_pinn(
        epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        use_fourier=False,
        hidden_size=128,
        num_residual_blocks=2
    )
