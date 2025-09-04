# model_comparison.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Load breast cancer dataset
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Different Model Architectures


class SimpleNN(nn.Module):
    """Simple feedforward network"""

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


class DeepNN(nn.Module):
    """Deeper network with dropout"""

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


class WideNN(nn.Module):
    """Wider network"""

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# Training and Evaluation


def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")

    start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_preds = torch.argmax(train_outputs, dim=1)
        train_acc = accuracy_score(y_train.numpy(), train_preds.numpy())

        test_outputs = model(X_test)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    return {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'num_parameters': num_params
    }


# Main Comparison
if __name__ == "__main__":
    print("Neural Network Architecture Comparison")
    print("=" * 50)

    input_size = X_train_tensor.shape[1]
    models = {
        'SimpleNN': SimpleNN(input_size),
        'DeepNN': DeepNN(input_size),
        'WideNN': WideNN(input_size)
    }

    results = []

    for name, model in models.items():
        result = train_and_evaluate(model, name, X_train_tensor, y_train_tensor,
                                    X_test_tensor, y_test_tensor)
        results.append(result)

    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Model':<15} {'Train Acc':<10} {'Test Acc':<10} {'Time (s)':<10} {'Params':<10}")
    print("-" * 60)

    for result in results:
        print(f"{result['model_name']:<15} {result['train_accuracy']:.4f}     "
              f"{result['test_accuracy']:.4f}     {result['training_time']:.4f}     "
              f"{result['num_parameters']:,}")

    print("\nModel comparison completed!")
