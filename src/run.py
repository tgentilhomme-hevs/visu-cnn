
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

from .visu import visualize_feature_maps

# -----------------------------
# 1. Hyperparameters
# -----------------------------

batch_size = 128
num_epochs = 30
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running on device:", device)

os.makedirs("fig_feature_maps", exist_ok=True)

# Choose which channels to visualize per layer
# (Make sure indices are < 8, since we use 8 channels per conv)
channels_per_layer = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7]
]
sample_idx = 0  # index of sample in batch to visualize

# -----------------------------
# 2. Dataset & DataLoaders
# -----------------------------

# Transform: ToTensor + Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
])


# Load MNIST dataset: building train and test sets
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 3. CNN Model (3 conv layers, small)
# -----------------------------

# Define a small CNN with 3 conv layers
# Each conv layer has 8 channels, ReLU activation, and max-pooling
# Then global average pooling and a final FC layer for 10 classes (use, .mean() over H,W)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # input: (N, 1, 28, 28)
        # First conv layer: 1 -> 8 channels, kernel 3x3, padding accrodingly
        # Second conv layer: 8 -> 8 channels, kernel 3x3, padding accordingly
        # Third conv layer: 8 -> 8 channels, kernel 3x3, padding accordingly
        # Each conv layer followed by ReLU and MaxPool2d(2)
        # YOUR SOLUTION HERE
        assert False, "Not implemented yet!"

        # After conv/pool:
        # conv1 -> pool -> 14x14, 8 channels
        # conv2 -> pool -> 7x7, 8 channels
        # conv3 (no pool) -> 7x7, 8 channels
        # Use global average pooling → 8 features
        self.feature_dim = 8
        # Final FC layer, from 8 features to num_classes
        # YOUR SOLUTION HERE
        assert False, "Not implemented yet!"

    def forward(self, x):
        # YOUR SOLUTION HERE
        assert False, "Not implemented yet!"
        return out

    def forward_feature_maps(self, x):
        """
        Forward pass that returns intermediate feature maps:

        - f1: after conv1 + ReLU (before pooling)
        - f2: after conv2 + ReLU (before pooling)
        - f3: after conv3 + ReLU
        """
        f1 = self.relu(self.conv1(x))   # (N,8,28,28)
        p1 = self.pool(f1)              # (N,8,14,14)

        f2 = self.relu(self.conv2(p1))  # (N,8,14,14)
        p2 = self.pool(f2)              # (N,8,7,7)

        f3 = self.relu(self.conv3(p2))  # (N,8,7,7)

        return f1, f2, f3


# Instantiate model
model = SmallCNN()

# Move model to device
# YOUR SOLUTION HERE
assert False, "Not implemented yet!"

# -----------------------------
# 4. Loss & Optimizer
# -----------------------------

# Define loss function and optimizer
# Use CrossEntropyLoss and Adam optimizer
# criterion = ...
# optimizer = ...
# YOUR SOLUTION HERE
assert False, "Not implemented yet!"


# -----------------------------
# 5. Evaluation helper
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # Forward pass + loss
            # YOUR SOLUTION HERE
            assert False, "Not implemented yet!"
            loss_sum += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total

# -----------------------------
# 6. Training loop
# -----------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    
    if epoch % 3 == 0 or epoch == 1:
        visualize_feature_maps(
            model, test_loader, device, epoch, channels_per_layer
        )

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # Forward + backward + optimize
        # Don't forget to zero gradients, backward and optimizer step
        # YOUR SOLUTION HERE
        assert False, "Not implemented yet!"

        running_loss += loss.item() * x.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(
        f"Epoch {epoch:02d}/{num_epochs} "
        f"- train loss: {train_loss:.4f} "
        f"- test loss: {test_loss:.4f} "
        f"- test acc: {test_acc:.4f}"
    )



