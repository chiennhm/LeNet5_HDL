import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import LeNet5, count_parameters

# TODO: Add random state
def get_data_loaders(data_dir="./data", batch_size=64, val_split=0.1, seed=42):
    """Create MNIST train, val and test data loaders (90/10 split)."""
    if seed:
        print("Using seed: ", seed)
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(root=data_dir, train=True,
                                        download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=transform)

    # Split train into train (90%) and val (10%)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train the model with validation each epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                correct += (output.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4f}")


def evaluate(model, test_loader, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LeNet5().to(device)
    print(model)
    print("\nModel parameters:")
    count_parameters(model)

    train_loader, val_loader, test_loader = get_data_loaders()

    print("\n--- Training ---")
    train(model, train_loader, val_loader, device, epochs=50)

    print("\n--- Test ---")
    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {acc:.4f}")

    # Save weights
    save_path = "lenet_mnist.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
