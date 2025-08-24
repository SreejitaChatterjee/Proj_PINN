#!/usr/bin/env python3
"""
Systematic optimization of physics weights for optimal parameter learning
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from enhanced_pinn_model import EnhancedQuadrotorPINN, EnhancedTrainer
from quadrotor_pinn_model import QuadrotorDataProcessor
import time

class OptimizedTrainer(EnhancedTrainer):
    """Optimized trainer with customizable weights"""
    
    def __init__(self, model, device='cpu'):
        super().__init__(model, device)
        
    def train_epoch(self, train_loader, physics_weight=5.0, reg_weight=2.0, direct_id_weight=10.0):
        """Training epoch with customizable weights"""
        self.model.train()
        total_loss = 0
        total_physics_loss = 0
        total_reg_loss = 0
        total_id_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            # Individual losses
            data_loss = self.criterion(output, target)
            physics_loss = self.model.enhanced_physics_loss(data, output, target)
            direct_id_loss = self.model.direct_parameter_identification_loss(data, target)
            reg_loss = self.model.parameter_regularization_loss()
            
            # Combined loss with custom weights
            loss = (data_loss + 
                   physics_weight * physics_loss + 
                   reg_weight * reg_loss +
                   direct_id_weight * direct_id_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.constrain_parameters()
            
            total_loss += loss.item()
            total_physics_loss += physics_loss.item()
            total_reg_loss += reg_loss.item()
            total_id_loss += direct_id_loss.item()
            
        return (total_loss / len(train_loader), 
                total_physics_loss / len(train_loader),
                total_reg_loss / len(train_loader),
                total_id_loss / len(train_loader))
    
    def train_with_weights(self, train_loader, val_loader, epochs=50, 
                          physics_weight=5.0, reg_weight=2.0, direct_id_weight=10.0):
        """Quick training with specified weights"""
        
        for epoch in range(epochs):
            train_loss, physics_loss, reg_loss, id_loss = self.train_epoch(
                train_loader, physics_weight, reg_weight, direct_id_weight)
            
            if epoch % 20 == 0:
                val_loss = self.validate(val_loader)
                print(f'  Epoch {epoch:03d}: Train: {train_loss:.2f}, Val: {val_loss:.6f}, '
                      f'Physics: {physics_loss:.2f}, Reg: {reg_loss:.2f}, ID: {id_loss:.2f}')

def evaluate_parameter_accuracy(model):
    """Calculate parameter accuracy for a trained model"""
    
    true_values = {
        'Mass': 0.068,
        'Jxx': 6.86e-5,
        'Jyy': 9.2e-5,
        'Jzz': 1.366e-4,
        'Gravity': 9.81
    }
    
    learned_values = {
        'Mass': model.m.item(),
        'Jxx': model.Jxx.item(),
        'Jyy': model.Jyy.item(),
        'Jzz': model.Jzz.item(),
        'Gravity': model.g.item()
    }
    
    accuracies = {}
    for param in true_values:
        error_pct = abs(learned_values[param] - true_values[param]) / true_values[param] * 100
        accuracies[param] = 100 - error_pct
    
    avg_accuracy = np.mean(list(accuracies.values()))
    return avg_accuracy, accuracies

def test_weight_combinations():
    """Systematically test different weight combinations"""
    
    print("SYSTEMATIC PHYSICS WEIGHT OPTIMIZATION")
    print("=" * 60)
    
    # Load and prepare data
    print("Loading training data...")
    df = pd.read_csv('quadrotor_training_data.csv')
    df = df[df['trajectory_id'] < 3].copy()
    
    processor = QuadrotorDataProcessor()
    X, y = processor.prepare_sequences(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different weight combinations
    weight_combinations = [
        # (physics_weight, reg_weight, direct_id_weight, description)
        (1.0, 0.5, 5.0, "Low physics emphasis"),
        (5.0, 2.0, 10.0, "Current best"),
        (10.0, 3.0, 15.0, "High physics emphasis"),
        (15.0, 5.0, 20.0, "Very high physics"),
        (20.0, 5.0, 25.0, "Maximum physics"),
        (10.0, 10.0, 30.0, "Strong regularization"),
        (25.0, 2.0, 50.0, "Physics + Direct ID focus"),
        (30.0, 1.0, 40.0, "Maximum physics, low reg")
    ]
    
    results = []
    
    print(f"\nTesting {len(weight_combinations)} weight combinations...")
    print("-" * 80)
    print(f"{'Config':<25} {'Physics':<8} {'Reg':<6} {'DirectID':<8} {'Accuracy':<10} {'Time':<8}")
    print("-" * 80)
    
    for i, (phys_w, reg_w, id_w, desc) in enumerate(weight_combinations):
        print(f"\n[{i+1}/{len(weight_combinations)}] Testing: {desc}")
        print(f"Weights - Physics: {phys_w}, Reg: {reg_w}, Direct ID: {id_w}")
        
        start_time = time.time()
        
        # Create fresh model for each test
        model = EnhancedQuadrotorPINN(input_size=12, hidden_size=128, output_size=16, num_layers=4)
        trainer = OptimizedTrainer(model, device)
        
        # Quick training (fewer epochs for screening)
        trainer.train_with_weights(train_loader, val_loader, epochs=60, 
                                 physics_weight=phys_w, reg_weight=reg_w, direct_id_weight=id_w)
        
        # Evaluate accuracy
        avg_accuracy, individual_acc = evaluate_parameter_accuracy(model)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        results.append({
            'config': desc,
            'physics_weight': phys_w,
            'reg_weight': reg_w,
            'direct_id_weight': id_w,
            'avg_accuracy': avg_accuracy,
            'individual_accuracy': individual_acc,
            'training_time': training_time
        })
        
        print(f"{desc:<25} {phys_w:<8} {reg_w:<6} {id_w:<8} {avg_accuracy:<9.1f}% {training_time:<7.1f}s")
        print(f"  Individual: Mass={individual_acc['Mass']:.1f}%, Jxx={individual_acc['Jxx']:.1f}%, "
              f"Jyy={individual_acc['Jyy']:.1f}%, Jzz={individual_acc['Jzz']:.1f}%")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['avg_accuracy'])
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best Configuration: {best_result['config']}")
    print(f"Optimal Weights:")
    print(f"  Physics Weight: {best_result['physics_weight']}")
    print(f"  Regularization Weight: {best_result['reg_weight']}")
    print(f"  Direct ID Weight: {best_result['direct_id_weight']}")
    print(f"Best Average Accuracy: {best_result['avg_accuracy']:.1f}%")
    print(f"Individual Accuracies:")
    for param, acc in best_result['individual_accuracy'].items():
        print(f"  {param}: {acc:.1f}%")
    
    # Sort by accuracy and show top 3
    sorted_results = sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)
    
    print(f"\nTop 3 Configurations:")
    print("-" * 60)
    for i, result in enumerate(sorted_results[:3]):
        print(f"{i+1}. {result['config']} - {result['avg_accuracy']:.1f}% accuracy")
        print(f"   Weights: P={result['physics_weight']}, R={result['reg_weight']}, D={result['direct_id_weight']}")
    
    return best_result

if __name__ == "__main__":
    best_config = test_weight_combinations()
    
    print(f"\nOptimal weights found!")
    print(f"Use these weights for final training:")
    print(f"physics_weight={best_config['physics_weight']}")
    print(f"reg_weight={best_config['reg_weight']}")  
    print(f"direct_id_weight={best_config['direct_id_weight']}")