# Experiment: Sentiment Analysis using BiLSTM on IMDB dataset

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Set seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Hyperparameters
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 200
EMBED_DIM = 128
BATCH = 64
NUM_EPOCHS = 10

# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_FEATURES)

# Padding sequences to fixed length
train_data = pad_sequences(train_data, maxlen=SEQUENCE_LENGTH, padding='post')
test_data = pad_sequences(test_data, maxlen=SEQUENCE_LENGTH, padding='post')

# Define the model
def build_model():
    model = Sequential()

    model.add(Embedding(input_dim=MAX_FEATURES, output_dim=EMBED_DIM, input_length=SEQUENCE_LENGTH))

    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.4))

    model.add(Bidirectional(LSTM(16)))
    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    return model

model = build_model()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Training
history = model.fit(
    train_data,
    train_labels,
    validation_split=0.2,
    epochs=NUM_EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")


# Result

# Epoch 1/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 128ms/step - accuracy: 0.7477 - loss: 0.5128 - val_accuracy: 0.8658 - val_loss: 0.3337
# Epoch 2/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 46s 145ms/step - accuracy: 0.8923 - loss: 0.2955 - val_accuracy: 0.8768 - val_loss: 0.3359
# Epoch 3/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 46s 148ms/step - accuracy: 0.9252 - loss: 0.2165 - val_accuracy: 0.8686 - val_loss: 0.3692
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 19s 24ms/step - accuracy: 0.8575 - loss: 0.3416

# Final Test Accuracy: 0.8575
# (base) aryaman@Aryamans-MacBook-Air DeepLearningLab % 