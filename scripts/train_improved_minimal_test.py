"""Quick 50-epoch test"""
import sys
sys.path.append('.')
from train_improved_minimal import train_minimal_improved

if __name__ == "__main__":
    train_minimal_improved(epochs=50, batch_size=128, lr=0.001)
