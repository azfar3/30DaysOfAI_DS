# Neural Network from Scratch - Explanation

## Architecture
```
Input Layer (2 neurons) → Hidden Layer (4 neurons) → Output Layer (1 neuron)
```

## Components

### 1. Initialization
- **Weights**: Random small values (using `np.random.randn() * 0.1`)
- **Biases**: Initialized to zeros
- Why small weights? Prevents saturation of sigmoid function

### 2. Forward Pass
1. `hidden_input = X • weights1 + bias1` (Matrix multiplication)
2. `hidden_output = sigmoid(hidden_input)` (Activation function)
3. `output_input = hidden_output • weights2 + bias2`
4. `prediction = sigmoid(output_input)`

### 3. Activation Function: Sigmoid
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
- Squashes values between 0 and 1
- Useful for binary classification
- **Derivative**: `sigmoid_derivative(x) = x * (1 - x)`

### 4. Loss Function: Mean Squared Error (MSE)
```python
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```
- Measures average squared difference between predictions and true values
- Easy to differentiate

### 5. Backpropagation (The Learning Part)

#### Output Layer Gradients:
1. `d_loss = 2 * (prediction - y) / m` (Derivative of MSE)
2. `d_sigmoid2 = sigmoid_derivative(prediction)`
3. `d_output = d_loss * d_sigmoid2` (Chain rule)

#### Hidden Layer Gradients:
1. `d_hidden = d_output • weights2.T` (Error from next layer)
2. `d_sigmoid1 = sigmoid_derivative(hidden_output)`
3. `d_hidden = d_hidden * d_sigmoid1` (Chain rule)

#### Weight Updates:
```python
weights -= learning_rate * (input.T • gradient)
biases -= learning_rate * sum(gradient)
```

## Why XOR Problem?
- XOR is not linearly separable
- Requires a hidden layer to learn
- Perfect for testing neural networks

## Hyperparameters
- **Hidden size**: 4 neurons (enough capacity for XOR)
- **Learning rate**: 0.1 (not too fast, not too slow)
- **Epochs**: 10,000 (enough to converge)

## What to Expect
- Loss should decrease over time
- Final predictions should be close to true XOR values
- Network learns the non-linear pattern