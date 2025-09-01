# neural_network_scratch.py
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetworkFromScratch:
    """
    A simple neural network built from scratch with NumPy
    Architecture: Input layer -> Hidden layer -> Output layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))

        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))

        # Store losses for plotting
        self.losses = []

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward(self, X):
        """Forward pass through the network"""
        # Input layer to hidden layer
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden layer to output layer
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.prediction = self.sigmoid(self.output_input)

        return self.prediction

    def compute_loss(self, y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y, learning_rate):
        """Backward pass (backpropagation)"""
        m = X.shape[0]  # Number of samples

        # Calculate gradients for output layer
        d_loss = 2 * (self.prediction - y) / m  # Derivative of MSE
        d_sigmoid2 = self.sigmoid_derivative(self.prediction)
        d_output = d_loss * d_sigmoid2

        # Calculate gradients for hidden layer
        d_hidden = np.dot(d_output, self.weights2.T)
        d_sigmoid1 = self.sigmoid_derivative(self.hidden_output)
        d_hidden = d_hidden * d_sigmoid1

        # Update weights and biases
        self.weights2 -= learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)

        self.weights1 -= learning_rate * np.dot(X.T, d_hidden)
        self.bias1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, predictions)
            self.losses.append(loss)

            # Backward pass (weight updates)
            self.backward(X, y, learning_rate)

            # Print progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

    def plot_loss(self):
        """Plot training loss over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title("Training Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error Loss")
        plt.grid(True)
        plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
        plt.show()


# Create a simple dataset for testing
def create_toy_dataset():
    """Create a simple non-linear dataset"""
    # XOR-like problem (non-linear)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR output

    return X, y


# Main execution
if __name__ == "__main__":
    print("Building Neural Network from Scratch...")

    # Create dataset
    X, y = create_toy_dataset()
    print("Dataset:")
    print("Features:\n", X)
    print("Labels:\n", y)

    # Create and train neural network
    nn = NeuralNetworkFromScratch(input_size=2, hidden_size=4, output_size=1)

    print("\nInitial Weights:")
    print("Weights1 (input -> hidden):\n", nn.weights1)
    print("Weights2 (hidden -> output):\n", nn.weights2)

    print("\nTraining Neural Network...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    print("\nFinal Predictions:")
    predictions = nn.predict(X)
    print("Input | True | Predicted")
    print("-----------------------")
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]} | {predictions[i][0]:.6f}")

    print(f"\nFinal Loss: {nn.losses[-1]:.6f}")

    # Plot training progress
    nn.plot_loss()

    print("\nNeural Network trained successfully!")
    print("Check 'training_loss.png' to see the learning curve.")
