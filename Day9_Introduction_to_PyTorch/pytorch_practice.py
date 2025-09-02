# pytorch_practice.py
import torch
import torch.nn as nn
import torch.optim as optim


def practice_1_different_architectures():
    """Practice with different network architectures"""
    print("Practice 1: Different Architectures")

    # 1. Deeper network
    class DeepNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 8)
            self.layer2 = nn.Linear(8, 4)
            self.layer3 = nn.Linear(4, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.sigmoid(self.layer3(x))
            return x

    # 2. Different activation functions
    class ReLUNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.output = nn.Linear(4, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.hidden(x))
            x = self.sigmoid(self.output(x))
            return x

    # Test both architectures
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("Testing deep network...")
    deep_model = DeepNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(deep_model.parameters(), lr=0.01)

    # Quick training
    for epoch in range(2000):
        predictions = deep_model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Deep network final loss: {loss.item():.6f}")

    print("Testing ReLU network...")
    relu_model = ReLUNetwork()
    optimizer = optim.Adam(relu_model.parameters(), lr=0.01)

    for epoch in range(2000):
        predictions = relu_model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"ReLU network final loss: {loss.item():.6f}")


def practice_2_different_optimizers():
    """Practice with different optimization algorithms"""
    print("\nPractice 2: Different Optimizers")

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
    }

    results = {}

    for opt_name, opt_class in optimizers.items():
        model = nn.Sequential(
            nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid()
        )

        criterion = nn.MSELoss()
        optimizer = opt_class(model.parameters(), lr=0.1)

        # Train
        for epoch in range(1000):
            predictions = model(X)
            loss = criterion(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        results[opt_name] = loss.item()
        print(f"{opt_name}: {loss.item():.6f}")

    return results


if __name__ == "__main__":
    print("Additional PyTorch Practice")
    print("=" * 50)

    practice_1_different_architectures()
    results = practice_2_different_optimizers()

    print("\nOptimizer Comparison:")
    for opt_name, loss in results.items():
        print(f"{opt_name}: {loss:.6f}")
