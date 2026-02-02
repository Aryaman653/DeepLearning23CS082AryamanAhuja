import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x) ** 2

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, layers, activation):
        self.weights = []
        self.biases = []
        self.activation = activation
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def act(self, x):
        if self.activation == "relu": return relu(x)
        if self.activation == "sigmoid": return sigmoid(x)
        return tanh(x)

    def act_deriv(self, x):
        if self.activation == "relu": return relu_deriv(x)
        if self.activation == "sigmoid": return sigmoid_deriv(x)
        return tanh_deriv(x)

    def forward(self, x):
        self.a = [x]
        self.z = []
        for i in range(len(self.weights) - 1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            self.a.append(self.act(z))
        z = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.z.append(z)
        self.a.append(softmax(z))
        return self.a[-1]

    def compute_loss(self, y, y_hat):
        return -np.mean(np.sum(y * np.log(y_hat + 1e-8), axis=1))

    def backward(self, y):
        gw, gb = [], []
        delta = self.a[-1] - y
        for i in reversed(range(len(self.weights))):
            gw.insert(0, self.a[i].T @ delta / y.shape[0])
            gb.insert(0, np.mean(delta, axis=0, keepdims=True))
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_deriv(self.z[i-1])
        return gw, gb

    def update(self, gw, gb, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gw[i]
            self.biases[i] -= lr * gb[i]

    def evaluate(self, loader):
        correct, total, loss = 0, 0, 0
        for images, labels in loader:
            x = images.cpu().numpy().reshape(images.shape[0], -1)
            y = one_hot(labels.cpu().numpy())
            preds = self.forward(x)
            loss += self.compute_loss(y, preds)
            correct += np.sum(np.argmax(preds, axis=1) == labels.numpy())
            total += labels.shape[0]
        return loss / len(loader), correct / total

experiments = [
    ([784, 64, 10], "relu"),
    ([784, 128, 10], "relu"),
    ([784, 128, 64, 10], "relu"),
    ([784, 128, 64, 10], "sigmoid"),
    ([784, 128, 64, 10], "tanh")
]

epochs = 5
lr = 0.01
results = []

for layers, act in experiments:
    net = NeuralNetwork(layers, act)
    for _ in range(epochs):
        for images, labels in train_loader:
            x = images.cpu().numpy().reshape(images.shape[0], -1)
            y = one_hot(labels.cpu().numpy())
            preds = net.forward(x)
            gw, gb = net.backward(y)
            net.update(gw, gb, lr)
    tr_loss, tr_acc = net.evaluate(train_loader)
    val_loss, val_acc = net.evaluate(val_loader)
    results.append([layers, act, tr_loss, tr_acc, val_loss, val_acc])

print("\nEXPERIMENT RESULTS\n")
print("{:<25} {:<10} {:<12} {:<12} {:<12} {:<12}".format(
    "Layers", "Activation", "Train Loss", "Train Acc", "Val Loss", "Val Acc"
))

for r in results:
    print("{:<25} {:<10} {:.4f}       {:.4f}       {:.4f}       {:.4f}".format(
        str(r[0]), r[1], r[2], r[3], r[4], r[5]
))

labels = [f"{r[0]}\n{r[1]}" for r in results]
train_acc = [r[3] for r in results]
val_acc = [r[5] for r in results]
val_loss = [r[4] for r in results]

plt.figure()
plt.bar(labels, val_acc)
plt.title("Validation Accuracy for Different Experiments")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("validation_accuracy.png")
plt.show()

plt.figure()
plt.bar(labels, val_loss)
plt.title("Validation Loss for Different Experiments")
plt.ylabel("Loss")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("validation_loss.png")
plt.show()

x = np.arange(len(labels))
plt.figure()
plt.bar(x - 0.2, train_acc, 0.4, label="Train Acc")
plt.bar(x + 0.2, val_acc, 0.4, label="Val Acc")
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.title("Train vs Validation Accuracy")
plt.tight_layout()
plt.savefig("train_vs_val_accuracy.png")
plt.show()


# EXPERIMENT RESULTS

# Layers                    Activation Train Loss   Train Acc    Val Loss     Val Acc     
# [784, 64, 10]             relu       0.2871       0.9209       0.2754       0.9234
# [784, 128, 10]            relu       0.2876       0.9188       0.2758       0.9219
# [784, 128, 64, 10]        relu       0.2169       0.9384       0.2147       0.9390
# [784, 128, 64, 10]        sigmoid    0.9605       0.7919       0.9483       0.7997
# [784, 128, 64, 10]        tanh       0.2609       0.9252       0.2530       0.9283
# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % 

# THE RESULTS-FIGURES ARE IN THE RESULTS FOLDER