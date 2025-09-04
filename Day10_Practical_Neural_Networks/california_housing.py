# california_housing.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and Prepare Data
print("Loading California Housing Dataset...")

# Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {california.feature_names}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"Training set: {X_train_tensor.shape}")
print(f"Test set: {X_test_tensor.shape}")

# 2. Create Neural Network


class HousingPredictor(nn.Module):
    """Neural network for California housing price prediction"""

    def __init__(self, input_size):
        super(HousingPredictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# 3. Training Function with Validation


def train_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size=32):
    """Train model with validation and early stopping"""

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    print("Training Model...")

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train)
        train_loss = criterion(predictions, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)

        # Learning rate scheduling
        scheduler.step(val_loss)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return train_losses, val_losses

# 4. Evaluation Function


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = nn.MSELoss()(predictions, y_test)
        test_rmse = torch.sqrt(test_loss)

        # Calculate R-squared
        ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
        ss_residual = torch.sum((y_test - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

    return test_loss.item(), test_rmse.item(), r_squared.item()


# 5. Main Execution
if __name__ == "__main__":
    print("California Housing Price Prediction")
    print("=" * 50)

    # Create model
    input_size = X_train_tensor.shape[1]
    model = HousingPredictor(input_size)

    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Split training data for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
    )

    # Train model
    train_losses, val_losses = train_model(
        model, X_train_final, y_train_final, X_val, y_val,
        epochs=1000, learning_rate=0.001, batch_size=64
    )

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    test_loss, test_rmse, r_squared = evaluate_model(model, X_test_tensor, y_test_tensor)

    print(f"\nFinal Results:")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"R² Score: {r_squared:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        plt.scatter(y_test_tensor.numpy(), predictions.numpy(), alpha=0.5)
        plt.plot([y_test_tensor.min(), y_test_tensor.max()],
                 [y_test_tensor.min(), y_test_tensor.max()], 'r--')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Actual Prices ($100,000)')
        plt.ylabel('Predicted Prices ($100,000)')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('housing_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nModel training and evaluation completed!")
