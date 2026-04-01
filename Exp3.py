import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import re

# -----------------------------
# Hyperparameters
# -----------------------------
EMBEDDING_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
MAX_VOCAB_SIZE = 25000
MAX_LEN = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Simple Tokenizer
# -----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Build Vocabulary
print("Building vocabulary...")
vocab = {"<unk>": 0, "<pad>": 1}
idx = 2
word_count = {}

for example in train_data:
    tokens = tokenize(example["text"])
    for token in tokens:
        word_count[token] = word_count.get(token, 0) + 1

# Keep top MAX_VOCAB_SIZE words
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE - 2]
for word, _ in sorted_words:
    vocab[word] = idx
    idx += 1

# Reverse vocab for decoding
reverse_vocab = {v: k for k, v in vocab.items()}

# -----------------------------
# Text Processing
# -----------------------------
def process_text(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    if len(ids) < MAX_LEN:
        ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids, dtype=torch.long)

def process_label(label):
    return torch.tensor(float(label), dtype=torch.float)

# -----------------------------
# Dataset Loader
# -----------------------------
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = process_text(example["text"])
        label = process_label(example["label"])
        return text, label

train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model: 1D CNN
# -----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=EMBEDDING_DIM,
                      out_channels=NUM_FILTERS,
                      kernel_size=fs)
            for fs in FILTER_SIZES
        ])

        self.fc = nn.Linear(len(FILTER_SIZES) * NUM_FILTERS, OUTPUT_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch, emb_dim, seq_len]

        conv_outs = [
            torch.relu(conv(embedded)) for conv in self.convs
        ]

        pooled = [
            torch.max(c, dim=2)[0] for c in conv_outs
        ]

        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)

        return self.fc(cat).squeeze(1)

# -----------------------------
# Initialize Model
# -----------------------------
model = TextCNN(len(vocab)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Function
# -----------------------------
def train():
    model.train()
    total_loss = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = torch.sigmoid(model(texts))
            preds = (outputs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    loss = train()
    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("-" * 30)

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "textcnn_imdb.pth")
print("Model saved!")


# Results

# # (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % python3 textcnn_imdb.py

# Using device: cpu
# Loading IMDB dataset...
# Building vocabulary...

# ===== Training TextCNN on IMDB =====

# Epoch 1:
# Loss: 0.6335
# Train Accuracy: 79.70%
# Test Accuracy: 77.46%
# ------------------------------

# Epoch 2:
# Loss: 0.4992
# Train Accuracy: 85.86%
# Test Accuracy: 81.80%
# ------------------------------

# Epoch 3:
# Loss: 0.4219
# Train Accuracy: 89.56%
# Test Accuracy: 83.96%
# ------------------------------

# Epoch 4:
# Loss: 0.3613
# Train Accuracy: 91.99%
# Test Accuracy: 85.20%
# ------------------------------

# Epoch 5:
# Loss: 0.3083
# Train Accuracy: 94.20%
# Test Accuracy: 85.53%
# ------------------------------

# Training completed with 5 epochs.

# Final Test Accuracy: 85.53%

# Model saved as: textcnn_imdb.pth

# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab %