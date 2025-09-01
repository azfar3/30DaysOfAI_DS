# additional_challenges.py
import numpy as np
from nerual_network_scratch import NeuralNetworkFromScratch


def challenge_1_circle_dataset():
    """Challenge 1: Create a circular dataset"""
    # Generate circular data (non-linear)
    np.random.seed(42)
    n_samples = 1000

    # Create two concentric circles
    radius_inner = 5
    radius_outer = 8

    # Inner circle (class 0)
    theta = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r = radius_inner + np.random.normal(0, 0.5, n_samples // 2)
    x1_inner = r * np.cos(theta)
    x2_inner = r * np.sin(theta)
    y_inner = np.zeros(n_samples // 2)

    # Outer circle (class 1)
    theta = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r = radius_outer + np.random.normal(0, 0.5, n_samples // 2)
    x1_outer = r * np.cos(theta)
    x2_outer = r * np.sin(theta)
    y_outer = np.ones(n_samples // 2)

    # Combine
    X = np.vstack(
        [np.column_stack([x1_inner, x2_inner]), np.column_stack([x1_outer, x2_outer])]
    )
    y = np.concatenate([y_inner, y_outer]).reshape(-1, 1)

    return X, y


def challenge_2_different_activation():
    """Challenge 2: Implement ReLU activation"""

    class NeuralNetworkWithReLU(NeuralNetworkFromScratch):
        def relu(self, x):
            """ReLU activation function"""
            return np.maximum(0, x)

        def relu_derivative(self, x):
            """Derivative of ReLU"""
            return (x > 0).astype(float)

        def forward(self, X):
            # Using ReLU instead of sigmoid
            self.hidden_input = np.dot(X, self.weights1) + self.bias1
            self.hidden_output = self.relu(self.hidden_input)

            self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
            self.prediction = self.sigmoid(self.output_input)  # Keep sigmoid for output

            return self.prediction

        def backward(self, X, y, learning_rate):
            # Modified backprop for ReLU
            m = X.shape[0]

            # Output layer (same as before)
            d_loss = 2 * (self.prediction - y) / m
            d_sigmoid2 = self.sigmoid_derivative(self.prediction)
            d_output = d_loss * d_sigmoid2

            # Hidden layer with ReLU
            d_hidden = np.dot(d_output, self.weights2.T)
            d_relu = self.relu_derivative(self.hidden_output)
            d_hidden = d_hidden * d_relu

            # Update weights (same as before)
            self.weights2 -= learning_rate * np.dot(self.hidden_output.T, d_output)
            self.bias2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)

            self.weights1 -= learning_rate * np.dot(X.T, d_hidden)
            self.bias1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # Test with ReLU
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR output
    nn_relu = NeuralNetworkWithReLU(input_size=2, hidden_size=4, output_size=1)
    nn_relu.train(X, y, epochs=10000, learning_rate=0.01)

    print("ReLU Network Predictions:")
    predictions = nn_relu.predict(X)
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]} | {predictions[i][0]:.6f}")


if __name__ == "__main__":
    print("Additional Challenges")

    # Run challenges
    X_circle, y_circle = challenge_1_circle_dataset()
    print("Circle dataset created with", len(X_circle), "samples")

    challenge_2_different_activation()
