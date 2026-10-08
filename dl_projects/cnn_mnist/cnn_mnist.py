"""Project 3 (CNN): MNIST handwritten digit classification.

Classifies 28x28 grayscale images of handwritten digits (0-9) with a small
convolutional neural network.

Run:  python cnn_mnist.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

# 1. Load data: 60,000 training and 10,000 test images, each 28x28 pixels
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, "Test:", x_test.shape)
print("Pixel range:", x_train.min(), "to", x_train.max())

# Look at a few examples
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, img, label in zip(axes.ravel(), x_train, y_train):
    ax.imshow(img, cmap="gray")
    ax.set_title(int(label))
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_digits.png")
plt.close()

# 2. Preprocess: scale pixels to 0-1 and add a channel dimension (28, 28, 1)
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

# 3. Build the CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),   # find small patterns
    layers.MaxPooling2D(pool_size=2),                      # shrink the image
    layers.Conv2D(64, kernel_size=3, activation="relu"),   # combine into bigger patterns
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),                                      # 5x5x64 -> 1600 numbers
    layers.Dropout(0.3),                                   # reduce overfitting
    layers.Dense(10, activation="softmax"),                # one output per digit
])

# 4. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 5. Train (early stopping keeps the best epoch)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2,
)

# 6. Evaluate on the untouched test set
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print("\n", classification_report(y_test, y_pred, digits=3))

# Confusion matrix heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted digit")
plt.ylabel("true digit")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Learning curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history.history["loss"], label="train loss")
ax[0].plot(history.history["val_loss"], label="validation loss")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train accuracy")
ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
plt.tight_layout()
plt.savefig("learning_curves.png")
plt.close()

# Show some mistakes the model made
wrong = np.where(y_pred != y_test)[0][:8]
if len(wrong):
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.6 * len(wrong), 2.2))
    for ax, i in zip(np.atleast_1d(axes), wrong):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"true {y_test[i]}\npred {y_pred[i]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("mistakes.png")
    plt.close()

# 7. Save the model and predict one image
model.save("mnist_cnn.keras")
probs = model.predict(x_test[:1], verbose=0)[0]
print("First test image: true", y_test[0], "-> predicted", int(np.argmax(probs)),
      f"({probs.max():.3f})")
