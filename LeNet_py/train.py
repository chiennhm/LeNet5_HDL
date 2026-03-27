import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5, count_parameters


def get_data_loaders(data_dir="./data", batch_size_train=64, batch_size_test=1000):
    """Create MNIST train and test data loaders."""
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=data_dir, train=False,
                                   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size_test)

    return train_loader, test_loader, test_dataset


def train(model, train_loader, device, epochs=10, lr=1e-3):
    """Train the model and print loss per epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


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

    train_loader, test_loader, _ = get_data_loaders()

    print("\n--- Training ---")
    train(model, train_loader, device, epochs=10)

    print("\n--- Evaluation ---")
    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {acc:.4f}")

    # Save weights
    save_path = "lenet_mnist.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
