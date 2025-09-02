# pytorch_neural_network.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ======================
# 1. Create the same XOR dataset
# ======================
def create_xor_dataset():
    """Create the same XOR dataset we used yesterday"""
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y


# ======================
# 2. Define Neural Network using PyTorch
# ======================
class XORNeuralNetwork(nn.Module):
    """Same neural network as yesterday, but using PyTorch"""

    def __init__(self, input_size, hidden_size, output_size):
        super(XORNeuralNetwork, self).__init__()

        # Define layers
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Define activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass - PyTorch automatically handles backward pass!"""
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# ======================
# 3. Training Function
# ======================
def train_model(model, X, y, epochs, learning_rate):
    """Train the neural network"""
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate
    )  # Stochastic Gradient Descent

    losses = []

    print("Training PyTorch Neural Network...")
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass and optimize
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients (automatic!)
        optimizer.step()  # Update weights

        losses.append(loss.item())

        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return losses


# ======================
# 4. Main Execution
# ======================
if __name__ == "__main__":
    print("Building Neural Network with PyTorch")
    print("=" * 50)

    # Create dataset
    X, y = create_xor_dataset()
    print("Dataset:")
    print("Features:\n", X)
    print("Labels:\n", y)

    # Create model
    model = XORNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    print("\n Model Architecture:")
    print(model)

    print("\n Model Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # Train model
    losses = train_model(model, X, y, epochs=10000, learning_rate=0.1)

    # Make predictions
    print("\nFinal Predictions:")
    with torch.no_grad():  # No gradient calculation needed for inference
        predictions = model(X)
        print("Input | True | Predicted")
        print("-----------------------")
        for i in range(len(X)):
            print(f"{X[i].numpy()} | {y[i].item()} | {predictions[i].item():.6f}")

    print(f"\nFinal Loss: {losses[-1]:.6f}")

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("PyTorch Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.grid(True)
    plt.savefig("pytorch_training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nPyTorch neural network trained successfully!")
