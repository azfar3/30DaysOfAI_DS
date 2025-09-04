# mnist_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and Prepare MNIST Data
print("Loading MNIST Dataset...")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Create data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")

# 2. Create CNN Model for MNIST


class MNISTCNN(nn.Module):
    """CNN for MNIST digit classification"""

    def __init__(self):
        super(MNISTCNN, self).__init__()

        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# 3. Training Function


def train_cnn(model, train_loader, test_loader, epochs, learning_rate):
    """Train CNN model"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Training CNN Model...")

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return train_losses, train_accuracies, test_accuracies


def evaluate(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total

# 4. Visualization Functions


def plot_results(train_losses, train_accuracies, test_accuracies):
    """Plot training results"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # Show some test examples with predictions
    model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        outputs = model(images)
        _, predicted = outputs.max(1)

        # Plot first 12 images
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f'True: {labels[i]}, Pred: {predicted[i]}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. Main Execution
if __name__ == "__main__":
    print("MNIST Digit Classification with CNN")
    print("=" * 50)

    # Create model
    model = MNISTCNN()
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    train_losses, train_accuracies, test_accuracies = train_cnn(
        model, train_loader, test_loader, epochs=10, learning_rate=0.001
    )

    # Final evaluation
    final_test_acc = evaluate(model, test_loader)
    print(f"\nFinal Test Accuracy: {final_test_acc:.2f}%")

    # Plot results
    plot_results(train_losses, train_accuracies, test_accuracies)

    print("\nMNIST classification completed!")
