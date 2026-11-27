"""Fashion image classification with multi-layer perceptrons (scikit-learn).

Answers Q1-Q9 of the case in order: the normalisation experiment, architectures, parameter count, the confusion
matrix and a binarised (toy quantisation) input. Every figure goes to output/figures and every table to
output/tables.

    uv run python src/fashion_mlp.py
"""

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
MAX_ITER = 60
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Q1
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(f"Q1 train {x_train.shape}, test {x_test.shape}, pixels {x_train.min()}-{x_train.max()}, "
      f"images per class {np.bincount(y_train)[0]}")

# Q2: one image per class
fig, ax = plt.subplots(2, 5, figsize=(6.4, 2.8))
for c, a in enumerate(ax.flat):
    a.imshow(x_train[np.argmax(y_train == c)], cmap="gray_r")
    a.set_title(CLASSES[c], fontsize=8)
    a.axis("off")
fig.tight_layout()
fig.savefig(FIG / "q2_one_per_class.png")
plt.close(fig)

flat_train = x_train.reshape(len(x_train), -1).astype("float32")
flat_test = x_test.reshape(len(x_test), -1).astype("float32")


def n_params(layers, n_in=784, n_out=10):
    sizes = [n_in, *layers, n_out]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def fit(layers, Xtr, Xte, label):
    t0 = time.perf_counter()
    m = MLPClassifier(hidden_layer_sizes=layers, max_iter=MAX_ITER, random_state=SEED).fit(Xtr, y_train)
    secs = time.perf_counter() - t0
    row = {"model": label, "hidden_layers": str(layers), "params": n_params(layers),
           "train_acc": m.score(Xtr, y_train), "test_acc": m.score(Xte, y_test), "cpu_seconds": round(secs, 1),
           "epochs": m.n_iter_}
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return m, row


# Q3: the normalisation experiment
print("Q3 / Q5 / Q9 fits")
rows = []
_, r = fit((2, 2), flat_train, flat_test, "(2, 2) raw pixels 0-255")
rows.append(r)
X_tr, X_te = flat_train / 255.0, flat_test / 255.0
for layers in [(2, 2), (16,), (16, 2), (32, 32)]:
    m, r = fit(layers, X_tr, X_te, f"{layers} pixels / 255")
    rows.append(r)
    if layers == (32, 32):
        best = m

# Q4: how big are the starting weights?
# scikit-learn draws ReLU weights uniformly in ±sqrt(6 / (fan_in + fan_out)) (Glorot)
bound = np.sqrt(6 / (784 + 32))
w0 = np.random.default_rng(SEED).uniform(-bound, bound, size=(784, 32))
print(f"Q4 initial first-layer weights within ±{bound:.3f}; a neuron's sum on raw pixels "
      f"averages {np.abs(flat_train[:1000] @ w0).mean():.1f} vs {np.abs(X_tr[:1000] @ w0).mean():.2f} on /255")

# Q6: parameter count of (32, 32)
q6 = pd.DataFrame([{"layer": "784 -> 32", "weights": 784 * 32, "biases": 32},
                   {"layer": "32 -> 32", "weights": 32 * 32, "biases": 32},
                   {"layer": "32 -> 10", "weights": 32 * 10, "biases": 10}])
q6["total"] = q6.weights + q6.biases
q6.to_csv(TAB / "q6_params_32_32.csv", index=False)
print("Q6\n", q6.to_string(index=False), "\n   total", q6.total.sum(),
      "check", sum(c.size for c in best.coefs_) + sum(b.size for b in best.intercepts_))

# Q7: confusion matrix
cm = confusion_matrix(y_test, best.predict(X_te))
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(TAB / "q7_confusion_matrix.csv")
off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j]
pairs = pd.DataFrame(sorted(off, reverse=True)[:6], columns=["count", "true", "predicted"])
pairs.to_csv(TAB / "q7_most_confused.csv", index=False)
per_class = pd.Series(cm.diagonal() / cm.sum(axis=1), index=CLASSES, name="accuracy").sort_values()
per_class.round(3).to_csv(TAB / "q7_per_class_accuracy.csv")
print("Q7\n", pairs.to_string(index=False), "\n", per_class.round(3).to_string())
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color="white" if cm[i, j] > 500 else "black")
ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=6)
ax.set_yticks(range(10), CLASSES, fontsize=6)
ax.set(xlabel="predicted", ylabel="true", title="MLP (32, 32), test set")
fig.tight_layout()
fig.savefig(FIG / "q7_confusion_matrix.png")
plt.close(fig)

# Q9: binarised input
_, r = fit((32, 32), (X_tr > 0.3).astype("float32"), (X_te > 0.3).astype("float32"), "(32, 32) binary (> 0.3)")
rows.append(r)

# Q8: comparison table
table = pd.DataFrame(rows)
table.round(4).to_csv(TAB / "q8_comparison.csv", index=False)
print("Q8\n", table.round(4).to_string(index=False))
