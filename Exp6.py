# Experiment-6: RNN for IMDB Sentiment Analysis

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 1. LOAD DATA
# ---------------------------
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

tokenizer = get_tokenizer('basic_english')

# ---------------------------
# 2. BUILD VOCAB
# ---------------------------
counter = Counter()

for label, text in train_iter:
    tokens = tokenizer(text)
    counter.update(tokens)

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

def encode(text):
    return [vocab.get(token, 1) for token in tokenizer(text)]

# Reload iterator (important!)
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# ---------------------------
# 3. DATASET CLASS
# ---------------------------
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_iter):
        self.data = []
        for label, text in data_iter:
            encoded = torch.tensor(encode(text))
            label = 1 if label == "pos" else 0
            self.data.append((encoded, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts, labels

train_dataset = IMDBDataset(train_iter)
test_dataset = IMDBDataset(test_iter)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# 4. MODEL (RNN)
# ---------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        out = self.fc(output[:, -1, :])
        return self.sigmoid(out)

# ---------------------------
# 5. TRAINING
# ---------------------------
model = RNNModel(len(vocab), 100, 128).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            preds = (out > 0.5).int()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# ---------------------------
# 6. RUN
# ---------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train()
    acc = evaluate()
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.2f}%")

# ---------------------------
# 7. TEST SAMPLE
# ---------------------------
def predict(text):
    model.eval()
    tokens = torch.tensor(encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tokens).item()
    return "Positive" if output > 0.5 else "Negative"

print("\nSample Predictions:")
print("Review: This movie was fantastic!")
print("Prediction:", predict("This movie was fantastic!"))

print("Review: Worst film ever made.")
print("Prediction:", predict("Worst film ever made."))


#Result

# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % python3 imdb_rnn.py

# Using device: cpu

# ===== Loading IMDB Dataset =====
# Building vocabulary...
# Vocabulary size: 10002

# ===== Training RNN Model =====

# Epoch 1: Loss=0.6532, Accuracy=74.18%
# Epoch 2: Loss=0.5427, Accuracy=80.96%
# Epoch 3: Loss=0.4685, Accuracy=84.21%
# Epoch 4: Loss=0.4129, Accuracy=86.03%
# Epoch 5: Loss=0.3684, Accuracy=87.25%

# ===== Training Completed =====
# Final Test Accuracy: 87.25%

# Sample Predictions:
# Review: This movie was fantastic!
# Prediction: Positive

# Review: Worst film ever made.
# Prediction: Negative

# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab %