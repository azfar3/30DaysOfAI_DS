# pytorch_fundamentals.py
import torch
import numpy as np

print("PyTorch Fundamentals")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # GPU support

# ======================
# 1. Tensor Basics
# ======================
print("\n" + "=" * 50)
print("1. TENSOR BASICS")
print("=" * 50)

# Creating tensors
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2], [3, 4]])
random_tensor = torch.randn(2, 3)  # Random normal distribution

print(f"Scalar: {scalar}, shape: {scalar.shape}")
print(f"Vector: {vector}, shape: {vector.shape}")
print(f"Matrix:\n{matrix}, shape: {matrix.shape}")
print(f"Random tensor:\n{random_tensor}, shape: {random_tensor.shape}")

# ======================
# 2. Tensor Operations
# ======================
print("\n" + "=" * 50)
print("2. TENSOR OPERATIONS")
print("=" * 50)

# Basic operations
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

print(f"a + b = {a + b}")  # Element-wise addition
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"Dot product: {torch.dot(a, b)}")  # Dot product
print(f"Matrix multiplication:\n{torch.matmul(matrix, matrix)}")

# ======================
# 3. Tensor Properties
# ======================
print("\n" + "=" * 50)
print("3. TENSOR PROPERTIES")
print("=" * 50)

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"Tensor:\n{x}")
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")
print(f"Requires gradient: {x.requires_grad}")

# ======================
# 4. Gradient Calculation
# ======================
print("\n" + "=" * 50)
print("4. AUTOGRAD - AUTOMATIC DIFFERENTIATION")
print("=" * 50)

# Create tensors with requires_grad=True for gradient tracking
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Simple computation: y = w*x + b
y = w * x + b

# Compute gradients
y.backward()

print(f"x.grad: {x.grad}")  # dy/dx = w = 3
print(f"w.grad: {w.grad}")  # dy/dw = x = 2
print(f"b.grad: {b.grad}")  # dy/db = 1

# ======================
# 5. NumPy Interoperability
# ======================
print("\n" + "=" * 50)
print("5. NUMPY INTEROPERABILITY")
print("=" * 50)

# Convert NumPy array to PyTorch tensor
np_array = np.array([[1, 2], [3, 4]])
torch_tensor = torch.from_numpy(np_array)
print(f"NumPy array:\n{np_array}")
print(f"PyTorch tensor:\n{torch_tensor}")

# Convert back to NumPy
torch_to_numpy = torch_tensor.numpy()
print(f"Back to NumPy:\n{torch_to_numpy}")

# ======================
# 6. GPU Support (if available)
# ======================
print("\n" + "=" * 50)
print("6. GPU SUPPORT")
print("=" * 50)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    print(f"Tensor moved to GPU: {x_gpu.device}")
else:
    print("GPU not available, using CPU")

print("\nPyTorch fundamentals completed!")
