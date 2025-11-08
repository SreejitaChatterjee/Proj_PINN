"""Quick 50-epoch test of residual model"""
import sys
sys.path.append('.')

from train_residual import *

if __name__ == "__main__":
    # Quick test
    data = pd.read_csv('../data/quadrotor_training_data.csv')

    state_cols = ['z', 'roll', 'pitch', 'yaw', 'p', 'q', 'r', 'vz']
    control_cols = ['thrust', 'torque_x', 'torque_y', 'torque_z']

    data_shifted = data[state_cols].shift(-1)
    data_shifted.columns = [c + '_next' for c in state_cols]

    data_combined = pd.concat([data[state_cols + control_cols], data_shifted], axis=1)
    data_combined = data_combined.dropna()

    X = data_combined[state_cols + control_cols].values
    y = data_combined[[c + '_next' for c in state_cols]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = QuadrotorPINN(hidden_size=256)
    trainer = Trainer(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    trainer.train(train_loader, val_loader, epochs=50)

    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, '../models/scalers_residual.pkl')
