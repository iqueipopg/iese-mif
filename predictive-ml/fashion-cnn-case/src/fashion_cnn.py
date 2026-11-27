"""Fashion image classification with convolutional networks (Keras).

Answers Q1-Q11 of the case in order: a dense Keras baseline, what re-running fit does, the case's CNN, its parameter
count, its mistakes and a deeper CNN for the >90% challenge. CPU only; seeds fixed, but TensorFlow on CPU is not
bit-for-bit deterministic across machines. Every figure goes to output/figures and every table to output/tables.

    uv run python src/fashion_cnn.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.datasets import fashion_mnist

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
keras.utils.set_random_seed(SEED)

# Q1
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(f"Q1 train {x_train.shape}, test {x_test.shape}, pixels {x_train.min()}-{x_train.max()}")
fig, ax = plt.subplots(2, 5, figsize=(6.4, 2.8))
for c, a in enumerate(ax.flat):
    a.imshow(x_train[np.argmax(y_train == c)], cmap="gray_r")
    a.set_title(CLASSES[c], fontsize=8)
    a.axis("off")
fig.tight_layout()
fig.savefig(FIG / "q1_one_per_class.png")
plt.close(fig)
x_train, x_test = x_train / 255.0, x_test / 255.0
results = []


def compile_(m):
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def record(name, m, secs, epochs):
    tr = m.evaluate(x_train[..., None] if m.input_shape[-1] == 1 else x_train, y_train, verbose=0)[1]
    te = m.evaluate(x_test[..., None] if m.input_shape[-1] == 1 else x_test, y_test, verbose=0)[1]
    row = {"model": name, "params": m.count_params(), "epochs": epochs, "train_acc": tr, "test_acc": te,
           "cpu_seconds": round(secs, 1)}
    results.append(row)
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


# Q2: dense baseline 784 -> 32 -> 32 -> 10, 10 epochs
dense = compile_(keras.Sequential([keras.Input((28, 28)), layers.Flatten(), layers.Dense(32, activation="relu"),
                                   layers.Dense(32, activation="relu"), layers.Dense(10, activation="softmax")]))
print("Q2 / Q3 dense baseline, three consecutive fit() calls of 10 epochs")
reruns = []
for run in range(1, 4):
    t0 = time.perf_counter()
    dense.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    secs = time.perf_counter() - t0
    tr = dense.evaluate(x_train, y_train, verbose=0)[1]
    te = dense.evaluate(x_test, y_test, verbose=0)[1]
    reruns.append({"run": run, "epochs_so_far": 10 * run, "train_acc": tr, "test_acc": te})
    print(f"   run {run}: train {tr:.4f}, test {te:.4f}")
    if run == 1:
        results.append({"model": "Dense 784-32-32-10", "params": dense.count_params(), "epochs": 10,
                        "train_acc": tr, "test_acc": te, "cpu_seconds": round(secs, 1)})
pd.DataFrame(reruns).round(4).to_csv(TAB / "q3_rerun_fit.csv", index=False)

# Q4-Q5, Q8: the case's CNN
cnn = compile_(keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
    layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax"),
]))
summary = pd.DataFrame([{"layer": l.__class__.__name__, "output": str(l.output.shape[1:]), "params": l.count_params()}
                        for l in cnn.layers])
summary.to_csv(TAB / "q8_cnn_summary.csv", index=False)
print("Q8\n", summary.to_string(index=False), "\n   total", cnn.count_params(), "vs dense", dense.count_params())

# Q9: 5 epochs
t0 = time.perf_counter()
hist = cnn.fit(x_train[..., None], y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
secs = time.perf_counter() - t0
pd.DataFrame(hist.history).round(4).rename_axis("epoch").to_csv(TAB / "q9_cnn_history.csv")
print("Q9")
record("CNN (case), 5 epochs", cnn, secs, 5)

# Q10: what it gets wrong
pred = cnn.predict(x_test[..., None], verbose=0).argmax(axis=1)
wrong = np.flatnonzero(pred != y_test)
per_class = pd.Series([(pred[y_test == c] == c).mean() for c in range(10)], index=CLASSES, name="accuracy")
per_class.round(3).sort_values().to_csv(TAB / "q10_per_class_accuracy.csv")
print(f"Q10 {len(wrong)} errors\n", per_class.round(3).sort_values().to_string())
fig, ax = plt.subplots(2, 6, figsize=(7.0, 2.8))
for a, i in zip(ax.flat, np.random.default_rng(SEED).choice(wrong, 12, replace=False)):
    a.imshow(x_test[i], cmap="gray_r")
    a.set_title(f"{CLASSES[y_test[i]]}\n→ {CLASSES[pred[i]]}", fontsize=6)
    a.axis("off")
fig.tight_layout()
fig.savefig(FIG / "q10_misclassified.png")
plt.close(fig)

# Q11: a deeper CNN for >90%
def block(f):
    return [layers.Conv2D(f, 3, padding="same", activation="relu"), layers.BatchNormalization(),
            layers.Conv2D(f, 3, padding="same", activation="relu"), layers.BatchNormalization(),
            layers.MaxPooling2D(2), layers.Dropout(0.25)]


deep = compile_(keras.Sequential([keras.Input((28, 28, 1)), *block(32), *block(64), layers.Flatten(),
                                  layers.Dense(128, activation="relu"), layers.Dropout(0.4),
                                  layers.Dense(10, activation="softmax")]))
print("Q11")
t0 = time.perf_counter()
deep.fit(x_train[..., None], y_train, epochs=12, batch_size=64, validation_split=0.1, verbose=0)
record("Deeper CNN (2 conv blocks + BN + dropout), 12 epochs", deep, time.perf_counter() - t0, 12)

pd.DataFrame(results).round(4).to_csv(TAB / "q11_comparison.csv", index=False)
print(pd.DataFrame(results).round(4).to_string(index=False))
