"""Project 1 (ANN): Iris flower classification.

Predicts the species of an iris flower (setosa, versicolor, virginica)
from 4 measurements, using a small fully connected neural network.

Run:  python iris_ann.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

# 1. Load data (target: 0 = setosa, 1 = versicolor, 2 = virginica)
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print("Shape:", X.shape)
print("Class counts:\n", y.value_counts())
print(X.describe())

# Quick EDA plot: correlation between features
plt.figure(figsize=(5, 4))
sns.heatmap(X.corr(), annot=True, cmap="Blues")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# 2. Train/test split (stratified keeps class ratios equal)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 3. Scale: fit on train only, then apply the same scaler to test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the ANN (3 output neurons, one per species)
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

# 5. Compile (labels are integers 0/1/2, so use the sparse loss)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 6. Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    verbose=0,
)

# 7. Evaluate on the untouched test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.3f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=iris.target_names))

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

# 8. Save the model and predict a new flower
model.save("iris_ann.keras")
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
probs = model.predict(scaler.transform(new_flower), verbose=0)[0]
print("Prediction for [5.1, 3.5, 1.4, 0.2]:",
      iris.target_names[np.argmax(probs)], probs.round(3))
