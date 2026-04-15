import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== Load and Prepare Text Data =====
with open("Shakespeare.txt", "r", encoding="utf-8") as file:
    raw_text = file.read().lower()[:100000]   # limiting size for training

# Create vocabulary mappings
unique_chars = sorted(set(raw_text))
char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

encoded_data = torch.tensor([char_to_idx[c] for c in raw_text], dtype=torch.long)

# ===== Model Hyperparameters =====
vocab_len = len(unique_chars)
embedding_dim = 256
heads = 8
layers = 4
context_length = 64
batch_sz = 64

# ===== Batch Generator =====
def create_batch():
    positions = torch.randint(0, len(encoded_data) - context_length - 1, (batch_sz,))
    inputs = torch.stack([encoded_data[p:p + context_length] for p in positions])
    targets = torch.stack([encoded_data[p + 1:p + context_length + 1] for p in positions])
    return inputs, targets

# ===== Transformer-based Model =====
class MiniGPT(nn.Module):
    def __init__(self):
        super(MiniGPT, self).__init__()

        self.token_embed = nn.Embedding(vocab_len, embedding_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, context_length, embedding_dim))

        decoder_block = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dropout=0.2,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_block, num_layers=layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_len)

    def forward(self, x):
        batch, time = x.shape

        x = self.token_embed(x) + self.position_embed[:, :time, :]

        # Create causal mask to prevent future leakage
        causal_mask = torch.triu(torch.ones(time, time) * float('-inf'), diagonal=1)

        x = self.transformer_decoder(x, x, tgt_mask=causal_mask)
        logits = self.output_layer(x)

        return logits

# Initialize model, optimizer, loss
model = MiniGPT()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# ===== Text Generation Function =====
def generate_text(model, seed_text="the ", gen_length=100):
    model.eval()

    input_ids = torch.tensor([char_to_idx[c] for c in seed_text]).unsqueeze(0)

    for _ in range(gen_length):
        current_input = input_ids[:, -context_length:]
        predictions = model(current_input)

        probs = F.softmax(predictions[0, -1], dim=0)
        next_token = torch.multinomial(probs, 1).item()

        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return "".join([idx_to_char[int(i)] for i in input_ids[0]])

# ===== Training Loop =====
for ep in range(10):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for _ in range(100):   # steps per epoch
        x_batch, y_batch = create_batch()

        logits = model(x_batch)
        loss = loss_fn(logits.view(-1, vocab_len), y_batch.view(-1))

        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == y_batch).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy.item()

    print(f"\nEpoch {ep+1}")
    print(f"Loss: {epoch_loss/100:.4f}")
    print(f"Accuracy: {(epoch_acc/100)*100:.2f}%")

    print("Generated Sample:")
    print(generate_text(model, "the ", 80))

# Result 
#     (base) aryaman@Aryamans-MacBook-Air DeepLearningLab %  /Users/aryaman/Desktop/DeepLearningLab/.venv/bin/python Exp10.py

# Epoch 1
# Loss: 2.4049
# Accuracy: 31.06%
# Generated Sample:
# the ttrktr ttrmtrt~ tefo9
#  trtd tatet tat tt'swort wathacet wiw, tst d wige wage wat

# Epoch 2
# Loss: 2.0500
# Accuracy: 37.75%
# Generated Sample:
# the bot ct totot ttotttottttote t tt ttt ttttttttotttttt tttt tt*ttttt ttttttttttttt

# Epoch 3
# Loss: 1.9402
# Accuracy: 40.17%
# Generated Sample:
# the the ttttttt ttttttttttttttlttttttttttttttttttttttttttttttttttttttttttttttttttttt

# Epoch 4
# Loss: 1.8392
# Accuracy: 42.79%
# Generated Sample:
# the ttottt t tttettttttttttttttttttttttttttttttttttttttttttttttttttttttt ttttttttttt

# Epoch 5
# Loss: 1.7134
# Accuracy: 46.30%
# Generated Sample:
# the tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

# Epoch 6
# Loss: 1.5216
# Accuracy: 51.62%
# Generated Sample:
# the the thetht ttheet tttteteettteetettet tttttrett metttttett'ttttttetttetttttttttt

# Epoch 7
# Loss: 1.3169
# Accuracy: 57.51%
# Generated Sample:
# the thtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

# Epoch 8
# Loss: 1.1528
# Accuracy: 62.42%
# Generated Sample:
# the tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

# Epoch 9
# Loss: 0.9878
# Accuracy: 67.45%
# Generated Sample:
# the teetettteetetetttttettttetttttettetttettttttetttettttttttttttttttttttttttttttttt

# Epoch 10
# Loss: 0.8153
# Accuracy: 72.84%
# Generated Sample:
# the ttttttt tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % 