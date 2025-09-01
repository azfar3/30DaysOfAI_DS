# Day 8: Neural Networks from Scratch

## Objective
Build a complete neural network from scratch using only NumPy to understand the fundamental mechanics of deep learning.

## What Was Built
- A fully functional neural network with forward and backward propagation
- Sigmoid activation functions and their derivatives
- Mean Squared Error loss function
- Gradient descent optimization from first principles

## Key Concepts Learned
1. **Neural Network Architecture**: Input layer → Hidden layer → Output layer
2. **Forward Propagation**: How data flows through the network
3. **Backpropagation**: How networks learn from errors (chain rule)
4. **Activation Functions**: Sigmoid and its importance for non-linearity
5. **Gradient Descent**: How weights are updated to minimize loss

## Files Created
- `neural_network_scratch.py`: Complete neural network implementation
- `neural_network_explanation.md`: Detailed explanation of the code
- `additional_challenges.py`: Extra exercises for practice
- `training_loss.png`: Loss curve visualization

## Technical Details
- **Network Architecture**: 2-4-1 (input-hidden-output)
- **Activation Function**: Sigmoid
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Gradient Descent
- **Dataset**: XOR problem (classic non-linear problem)

## How to Run
```bash
# Run the main neural network
python neural_network_scratch.py

# Run additional challenges
python additional_challenges.py
```

## Results
The neural network successfully learns the XOR function:
- Input: [0, 0] → Output: ~0 (correct)
- Input: [0, 1] → Output: ~1 (correct) 
- Input: [1, 0] → Output: ~1 (correct)
- Input: [1, 1] → Output: ~0 (correct)

## Insights Gained
- Understanding the mathematics behind neural networks
- Appreciation for automatic differentiation in frameworks
- intuition about learning rates and network architecture
- Foundation for understanding more complex deep learning concepts

## Next Steps
- Day 9: Introduction to PyTorch framework
- Day 10: Building neural networks with PyTorch
- Day 11: Training on larger datasets