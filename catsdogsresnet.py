
# Experiment-4 : CNN Implementation
# Cats vs Dogs & CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Transforms
cifar_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dogs_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Datasets
cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
cifar_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar_transform)

cifar_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_test_loader  = DataLoader(cifar_test, batch_size=64, shuffle=False)


dogs_train = datasets.ImageFolder("./cats_vs_dogs/train", transform=dogs_transform)
dogs_test  = datasets.ImageFolder("./cats_vs_dogs/test", transform=dogs_transform)

dogs_train_loader = DataLoader(dogs_train, batch_size=32, shuffle=True)
dogs_test_loader  = DataLoader(dogs_test, batch_size=32, shuffle=False)
# CNN Model
class CNN(nn.Module):
    def __init__(self, activation_fn, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            activation_fn(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation_fn(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            activation_fn(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            activation_fn(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
# Weight Initialization
def init_weights(model, init_type):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init_type == "random":
                nn.init.normal_(m.weight, mean=0, std=0.02)
# Train & Evaluate
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100 * correct / total
# Experiment Configurations
activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU
}

optimizers = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop
}

initializations = ["xavier", "kaiming", "random"]
# Run CNN Experiments
def run_experiments(train_loader, test_loader, num_classes, save_name):
    best_acc = 0

    for act_name, act_fn in activations.items():
        for init_type in initializations:
            for opt_name, opt_fn in optimizers.items():

                print(f"\nActivation={act_name}, Init={init_type}, Optimizer={opt_name}")

                model = CNN(act_fn, num_classes).to(device)
                init_weights(model, init_type)

                optimizer = opt_fn(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(5):
                    loss = train_one_epoch(model, train_loader, optimizer, criterion)
                    acc = evaluate(model, test_loader)
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), save_name)

    print(f"Best Accuracy Saved ({save_name}): {best_acc:.2f}%")
    return best_acc
# Transfer Learning: ResNet-18
def train_resnet(train_loader, test_loader, num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"ResNet Epoch {epoch+1}: Acc={acc:.2f}%")

    return acc
# Main
if __name__ == "__main__":

    print("\n===== CIFAR-10 CNN Experiments =====")
    best_cifar = run_experiments(
        cifar_train_loader,
        cifar_test_loader,
        num_classes=10,
        save_name="best_cifar_cnn.pth"
    )

    print("\n===== Cats vs Dogs CNN Experiments =====")
    best_dogs = run_experiments(
        dogs_train_loader,
        dogs_test_loader,
        num_classes=2,
        save_name="best_dogs_cnn.pth"
    )

    print("\n===== ResNet-18 CIFAR-10 =====")
    resnet_cifar = train_resnet(cifar_train_loader, cifar_test_loader, 10)

    print("\n===== ResNet-18 Cats vs Dogs =====")
    resnet_dogs = train_resnet(dogs_train_loader, dogs_test_loader, 2)

    print("\n===== Final Comparison =====")
    print(f"CNN CIFAR-10 Best Accuracy : {best_cifar:.2f}%")
    print(f"ResNet CIFAR-10 Accuracy  : {resnet_cifar:.2f}%")
    print(f"CNN Dogs Best Accuracy   : {best_dogs:.2f}%")
    print(f"ResNet Dogs Accuracy     : {resnet_dogs:.2f}%")
