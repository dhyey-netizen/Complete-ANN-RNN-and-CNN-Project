"""Project 2 (RNN): IMDB movie review sentiment analysis.

Reads a movie review one word at a time and predicts positive or negative.
Trains a SimpleRNN first, then an LSTM, and compares them.

Run:  python imdb_rnn.py
"""
import matplotlib
matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

VOCAB = 10000   # keep the 10,000 most common words
MAXLEN = 200    # every review becomes 200 tokens
EPOCHS = 5

# 1. Load (each review is a list of word indices)
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB)
print("Train / test reviews:", len(x_train), len(x_test))
print("First review (first 10 ids):", x_train[0][:10], "label:", y_train[0])

# 2. Pad / truncate to equal length
x_train = keras.utils.pad_sequences(x_train, maxlen=MAXLEN)
x_test = keras.utils.pad_sequences(x_test, maxlen=MAXLEN)


def build_model(cell):
    """Embedding -> recurrent layer -> sigmoid output."""
    return keras.Sequential([
        layers.Input(shape=(MAXLEN,)),
        layers.Embedding(VOCAB, 32),
        cell,
        layers.Dense(1, activation="sigmoid"),
    ])


def train_and_eval(name, cell):
    model = build_model(cell)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(f"\n=== {name} ===")
    model.summary()
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=64,
        verbose=2,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} test accuracy: {acc:.3f}")
    return model, history, acc


rnn_model, rnn_hist, rnn_acc = train_and_eval("SimpleRNN", layers.SimpleRNN(32))
lstm_model, lstm_hist, lstm_acc = train_and_eval("LSTM", layers.LSTM(32))

print(f"\nSimpleRNN: {rnn_acc:.3f}   LSTM: {lstm_acc:.3f}")

# Compare validation accuracy
plt.plot(rnn_hist.history["val_accuracy"], label="SimpleRNN")
plt.plot(lstm_hist.history["val_accuracy"], label="LSTM")
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_vs_lstm.png")
plt.close()

lstm_model.save("imdb_lstm.keras")

# 7. Predict your own reviews
word_index = keras.datasets.imdb.get_word_index()


def encode(text):
    ids = [1]  # 1 = start-of-review token
    for w in text.lower().split():
        i = word_index.get(w, -1) + 3  # Keras shifts indices by 3
        ids.append(i if 3 <= i < VOCAB else 2)  # 2 = unknown word
    return keras.utils.pad_sequences([ids], maxlen=MAXLEN)


for review in ["this movie was fantastic and moving",
               "terrible plot and awful acting"]:
    p = float(lstm_model.predict(encode(review), verbose=0)[0][0])
    print(f"{review!r} -> {'positive' if p > 0.5 else 'negative'} ({p:.3f})")
