import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD AND PREPROCESS DATA
df = pd.read_csv("poems-100.csv")
text = " ".join(df["text"].astype(str).tolist()).lower()

# 2. CREATE VOCABULARY
words = text.split()
vocabulary = sorted(set(words))
w2i = {word: idx for idx, word in enumerate(vocabulary)}
i2w = {idx: word for word, idx in w2i.items()}
vocab_len = len(vocabulary)

# 3. CREATE SEQUENCES
sequence_length = 5
X_data, y_data = [], []
for i in range(len(words) - sequence_length):
    sequence = words[i:i+sequence_length]
    next_word = words[i+sequence_length]
    X_data.append([w2i[w] for w in sequence])
    y_data.append(w2i[next_word])

X_data = torch.tensor(X_data)
y_data = torch.tensor(y_data)

# 4. ONE-HOT ENCODER
def to_onehot(tensor, vocab_len):
    return torch.eye(vocab_len)[tensor]

# 5. MODEL CLASS
class PoemGenerator(nn.Module):
    def __init__(self, vocab_len, emb_dim, hidden_dim, use_emb, architecture="RNN"):
        super().__init__()
        self.use_emb = use_emb
        
        if use_emb:
            self.embedding = nn.Embedding(vocab_len, emb_dim)
            inp_size = emb_dim
        else:
            inp_size = vocab_len
        
        if architecture == "LSTM":
            self.cell = nn.LSTM(inp_size, hidden_dim, batch_first=True)
        else:
            self.cell = nn.RNN(inp_size, hidden_dim, batch_first=True)
        
        self.out_layer = nn.Linear(hidden_dim, vocab_len)
    
    def forward(self, x):
        if self.use_emb:
            x = self.embedding(x)
        output, _ = self.cell(x)
        output = self.out_layer(output[:, -1, :])
        return output

# 6. TRAINING FUNCTION
def train(use_emb=False, architecture="RNN", num_epochs=20):
    model = PoemGenerator(vocab_len, 50, 128, use_emb, architecture).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.003)
    
    X = X_data.to(device)
    y = y_data.to(device)
    
    if not use_emb:
        X = to_onehot(X, vocab_len).to(device)
    
    for epoch in range(num_epochs):
        opt.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        loss.backward()
        opt.step()
        print(f"{architecture} | Emb={use_emb} | Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

# 7. TEXT GENERATION
def generate(model, seed, gen_length=20, use_emb=False):
    model.eval()
    output_words = seed.lower().split()
    
    for _ in range(gen_length):
        seq = [w2i.get(w, 0) for w in output_words[-sequence_length:]]
        seq_tensor = torch.tensor([seq]).to(device)
        
        if not use_emb:
            seq_tensor = to_onehot(seq_tensor, vocab_len).to(device)
        
        with torch.no_grad():
            prediction = model(seq_tensor)
            next_idx = torch.argmax(prediction).item()
            next_word = i2w[next_idx]
        
        output_words.append(next_word)
    
    return " ".join(output_words)

# 8. TRAIN ALL MODELS
print("\n=== RNN + One-Hot ===")
model1 = train(use_emb=False, architecture="RNN")

print("\n=== LSTM + One-Hot ===")
model2 = train(use_emb=False, architecture="LSTM")

print("\n=== RNN + Embeddings ===")
model3 = train(use_emb=True, architecture="RNN")

print("\n=== LSTM + Embeddings ===")
model4 = train(use_emb=True, architecture="LSTM")

# 9. GENERATE TEXT
starter = "love is like"
print("\n--- Generated Text ---")
print(f"RNN+OneHot: {generate(model1, starter, use_emb=False)}")
print(f"LSTM+OneHot: {generate(model2, starter, use_emb=False)}")
print(f"RNN+Embedding: {generate(model3, starter, use_emb=True)}")
print(f"LSTM+Embedding: {generate(model4, starter, use_emb=True)}")





# RESULTS

# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab %  python RNNLSTM.py

# === RNN + One-Hot ===
# RNN | Emb=False | Epoch 1, Loss: 8.8678
# RNN | Emb=False | Epoch 2, Loss: 8.8018
# RNN | Emb=False | Epoch 3, Loss: 8.7217
# RNN | Emb=False | Epoch 4, Loss: 8.5821
# RNN | Emb=False | Epoch 5, Loss: 8.3370
# RNN | Emb=False | Epoch 6, Loss: 7.9871
# RNN | Emb=False | Epoch 7, Loss: 7.6356
# RNN | Emb=False | Epoch 8, Loss: 7.3943
# RNN | Emb=False | Epoch 9, Loss: 7.2396
# RNN | Emb=False | Epoch 10, Loss: 7.1206
# RNN | Emb=False | Epoch 11, Loss: 7.0303
# RNN | Emb=False | Epoch 12, Loss: 6.9716
# RNN | Emb=False | Epoch 13, Loss: 6.9395
# RNN | Emb=False | Epoch 14, Loss: 6.9248
# RNN | Emb=False | Epoch 15, Loss: 6.9218
# RNN | Emb=False | Epoch 16, Loss: 6.9265
# RNN | Emb=False | Epoch 17, Loss: 6.9341
# RNN | Emb=False | Epoch 18, Loss: 6.9373
# RNN | Emb=False | Epoch 19, Loss: 6.9314
# RNN | Emb=False | Epoch 20, Loss: 6.9174

# === LSTM + One-Hot ===
# LSTM | Emb=False | Epoch 1, Loss: 8.8511
# LSTM | Emb=False | Epoch 2, Loss: 8.8247
# LSTM | Emb=False | Epoch 3, Loss: 8.7955
# LSTM | Emb=False | Epoch 4, Loss: 8.7578
# LSTM | Emb=False | Epoch 5, Loss: 8.7054
# LSTM | Emb=False | Epoch 6, Loss: 8.6295
# LSTM | Emb=False | Epoch 7, Loss: 8.5178
# LSTM | Emb=False | Epoch 8, Loss: 8.3544
# LSTM | Emb=False | Epoch 9, Loss: 8.1243
# LSTM | Emb=False | Epoch 10, Loss: 7.8323
# LSTM | Emb=False | Epoch 11, Loss: 7.5428
# LSTM | Emb=False | Epoch 12, Loss: 7.3800
# LSTM | Emb=False | Epoch 13, Loss: 7.3531
# LSTM | Emb=False | Epoch 14, Loss: 7.3188
# LSTM | Emb=False | Epoch 15, Loss: 7.2343
# LSTM | Emb=False | Epoch 16, Loss: 7.1223
# LSTM | Emb=False | Epoch 17, Loss: 7.0105
# LSTM | Emb=False | Epoch 18, Loss: 6.9186
# LSTM | Emb=False | Epoch 19, Loss: 6.8583
# LSTM | Emb=False | Epoch 20, Loss: 6.8339

# === RNN + Embeddings ===
# RNN | Emb=True | Epoch 1, Loss: 8.8901
# RNN | Emb=True | Epoch 2, Loss: 8.7980
# RNN | Emb=True | Epoch 3, Loss: 8.7069
# RNN | Emb=True | Epoch 4, Loss: 8.6003
# RNN | Emb=True | Epoch 5, Loss: 8.4568
# RNN | Emb=True | Epoch 6, Loss: 8.2502
# RNN | Emb=True | Epoch 7, Loss: 7.9722
# RNN | Emb=True | Epoch 8, Loss: 7.6611
# RNN | Emb=True | Epoch 9, Loss: 7.3814
# RNN | Emb=True | Epoch 10, Loss: 7.1669
# RNN | Emb=True | Epoch 11, Loss: 7.0133
# RNN | Emb=True | Epoch 12, Loss: 6.9069
# RNN | Emb=True | Epoch 13, Loss: 6.8431
# RNN | Emb=True | Epoch 14, Loss: 6.8119
# RNN | Emb=True | Epoch 15, Loss: 6.7964
# RNN | Emb=True | Epoch 16, Loss: 6.7797
# RNN | Emb=True | Epoch 17, Loss: 6.7521
# RNN | Emb=True | Epoch 18, Loss: 6.7148
# RNN | Emb=True | Epoch 19, Loss: 6.6746
# RNN | Emb=True | Epoch 20, Loss: 6.6371

# === LSTM + Embeddings ===
# LSTM | Emb=True | Epoch 1, Loss: 8.8529
# LSTM | Emb=True | Epoch 2, Loss: 8.8195
# LSTM | Emb=True | Epoch 3, Loss: 8.7835
# LSTM | Emb=True | Epoch 4, Loss: 8.7385
# LSTM | Emb=True | Epoch 5, Loss: 8.6770
# LSTM | Emb=True | Epoch 6, Loss: 8.5890
# LSTM | Emb=True | Epoch 7, Loss: 8.4619
# LSTM | Emb=True | Epoch 8, Loss: 8.2811
# LSTM | Emb=True | Epoch 9, Loss: 8.0387
# LSTM | Emb=True | Epoch 10, Loss: 7.7479
# LSTM | Emb=True | Epoch 11, Loss: 7.4537
# LSTM | Emb=True | Epoch 12, Loss: 7.2173
# LSTM | Emb=True | Epoch 13, Loss: 7.0783
# LSTM | Emb=True | Epoch 14, Loss: 7.0283
# LSTM | Emb=True | Epoch 15, Loss: 7.0273
# LSTM | Emb=True | Epoch 16, Loss: 7.0393
# LSTM | Emb=True | Epoch 17, Loss: 7.0363
# LSTM | Emb=True | Epoch 18, Loss: 7.0045
# LSTM | Emb=True | Epoch 19, Loss: 6.9489
# LSTM | Emb=True | Epoch 20, Loss: 6.8859

# --- Generated Text ---
# RNN+OneHot: love is like the the the the the the the the the the the the the the the the the the the the
# LSTM+OneHot: love is like the the the the the the the the the the the the the the the the the the the the
# RNN+Embedding: love is like the the the the the the the the the the the the the the the the the the the the
# LSTM+Embedding: love is like and the the the the the the the the the the the the the the the the the the the