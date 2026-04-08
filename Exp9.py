# Experiment: Sentiment Classification on Amazon Reviews using GRU

import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reproducibility
tf.keras.utils.set_random_seed(101)

# Load dataset from KaggleHub
dataset_path = kagglehub.dataset_download("yasserh/amazon-product-reviews-dataset")
file_path = os.path.join(dataset_path, "7817_1.csv")

data = pd.read_csv(file_path)

# Keep relevant columns and clean
data = data[['reviews.text', 'reviews.rating']].dropna()
data.rename(columns={'reviews.text': 'text', 'reviews.rating': 'rating'}, inplace=True)

# Convert ratings to binary labels (positive / negative)
data['label'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0)

# Train-test split
train_text, test_text, train_labels, test_labels = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=101
)

# Tokenization
vocab_size = 6000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_text)

train_seq = tokenizer.texts_to_sequences(train_text)
test_seq = tokenizer.texts_to_sequences(test_text)

# Padding sequences
max_len = 120
train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post')
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post')

# Build model (slightly deeper + dropout added)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),

    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.GRU(32),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_pad,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(test_pad, test_labels, verbose=1)
print(f"Final Test Accuracy: {test_acc:.4f}")


#Result

# Epoch 1/5
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 3s 90ms/step - accuracy: 0.9189 - loss: 0.5116 - val_accuracy: 0.9365 - val_loss: 0.3260
# Epoch 2/5
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9242 - loss: 0.3263 - val_accuracy: 0.9365 - val_loss: 0.2723
# Epoch 3/5
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9242 - loss: 0.2864 - val_accuracy: 0.9365 - val_loss: 0.2435
# Epoch 4/5
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9242 - loss: 0.2699 - val_accuracy: 0.9365 - val_loss: 0.2363
# Epoch 5/5
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 68ms/step - accuracy: 0.9242 - loss: 0.2673 - val_accuracy: 0.9365 - val_loss: 0.2357
# 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9703 - loss: 0.1491 
# Final Test Accuracy: 0.9703
# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % 