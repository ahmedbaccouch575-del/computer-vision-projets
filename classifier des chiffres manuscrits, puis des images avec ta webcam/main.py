"""Classification de chiffres manuscrits (MNIST) avec un CNN en PyTorch.

Lancer : python main.py
Sortie : model.pth (poids du meilleur modele)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CNN(nn.Module):
    """2 couches de convolution + 2 couches denses."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 1x28x28 -> 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x14x14 -> 64x14x14
        self.pool = nn.MaxPool2d(2)                               # divise la taille par 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)                             # 10 classes (0 a 9)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = x.flatten(1)                      # -> 3136
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)

    # 55 000 images pour entrainer, 5 000 pour valider
    train_data, val_data = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000)
    test_loader = DataLoader(test_data, batch_size=1000)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main():
    train_loader, val_loader, test_loader = get_loaders()
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, val_loader)
        print(f"Epoque {epoch}/{EPOCHS} - perte : {loss:.4f} - precision validation : {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "model.pth")

    # Le test est utilise une seule fois, a la fin
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    print(f"Precision sur le test : {evaluate(model, test_loader):.4f}")


if __name__ == "__main__":
    main()