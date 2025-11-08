"""Train stable PINN with corrected hyperparameters - 100 epochs"""
import sys
sys.path.append('.')
from train_stable import train_stable_pinn

if __name__ == "__main__":
    # 100-epoch training with balanced hyperparameters
    train_stable_pinn(
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        use_fourier=False,
        hidden_size=128,
        num_residual_blocks=2
    )
