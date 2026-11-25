"""Predicting credit card churn with K-Nearest Neighbours (Kaggle Credit Card Customers).

Answers Q1-Q8 of the case in order. Every figure goes to output/figures and every table to
output/tables, so the report can be rebuilt from this script alone.

    uv run python src/churn_knn.py
"""

from pathlib import Path

import kagglehub
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#2b5c8a", "#c0392b", "#9a9a9a"
K_GRID = [3, 5, 7, 11, 15, 21, 31, 51]


def rates(y_true, y_pred):
    """Accuracy, recall and precision on churners, plus the raw confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


# ---------------------------------------------------------------- Q1: load, drop leakage, encode
path = Path(kagglehub.dataset_download("sakshigoyal7/credit-card-customers"))
raw = pd.read_csv(path / "BankChurners.csv")
print(f"Q1. Raw file: {raw.shape[0]} rows x {raw.shape[1]} columns")
leak = [c for c in raw.columns if c.startswith("Naive_Bayes_Classifier")]
df = raw.drop(columns=leak + ["CLIENTNUM"])
df["churn"] = (df.pop("Attrition_Flag") == "Attrited Customer").astype(int)
CAT = df.select_dtypes(include=["object", "string"]).columns.tolist()
NUM = [c for c in df.select_dtypes("number").columns if c != "churn"]
print(f"   Dropped {len(leak)} leakage columns and CLIENTNUM -> {len(NUM)} numeric + {len(CAT)} categorical features")
print(f"   Churn rate: {df['churn'].mean():.2%} ({df['churn'].sum()} of {len(df)})")
pd.DataFrame([(c, "numeric") for c in NUM] + [(c, "categorical") for c in CAT],
             columns=["variable", "type"]).to_csv(TAB / "q1_variable_types.csv", index=False)

X_raw = df.drop(columns="churn")
y = df["churn"]
X = pd.get_dummies(X_raw, columns=CAT, drop_first=False, dtype=float)
print(f"   Feature matrix after get_dummies: {X.shape[1]} columns")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xr_tr, Xr_te = X_raw.loc[X_tr.index], X_raw.loc[X_te.index]
print(f"   Train {len(X_tr)} / test {len(X_te)}, churn {y_tr.mean():.1%} / {y_te.mean():.1%}")

# ---------------------------------------------------------------- Q2: KNN k=5 on raw data
results = {"Always \"stays\"": rates(y_te, np.zeros(len(y_te), dtype=int))}
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
results["KNN k=5, unscaled"] = rates(y_te, knn_raw.predict(X_te))
print(f"Q2. KNN k=5 unscaled: accuracy {results['KNN k=5, unscaled']['accuracy']:.1%}, "
      f"recall {results['KNN k=5, unscaled']['recall']:.1%} (baseline {results['Always \"stays\"']['accuracy']:.1%})")

# ---------------------------------------------------------------- Q3: ranges
rng = pd.DataFrame({c: [X[c].min(), X[c].max(), X[c].max() - X[c].min()] for c in NUM},
                   index=["min", "max", "range"]).T.sort_values("range", ascending=False)
rng.to_csv(TAB / "q3_ranges.csv")
ratio = rng.loc["Credit_Limit", "range"] / rng.loc["Avg_Utilization_Ratio", "range"]
print(f"Q3. Credit_Limit range {rng.loc['Credit_Limit', 'range']:,.0f} vs Avg_Utilization_Ratio "
      f"{rng.loc['Avg_Utilization_Ratio', 'range']:.3f}: {ratio:,.0f} times larger")

# How much of the squared distance does each column take, unscaled? Mean squared gap between random pairs.
rs = np.random.default_rng(SEED)
i, j = rs.integers(0, len(X_tr), 20000), rs.integers(0, len(X_tr), 20000)
sq = ((X_tr.values[i] - X_tr.values[j]) ** 2).mean(axis=0)
share = pd.Series(sq / sq.sum(), index=X.columns).sort_values(ascending=False)
share.rename("share_of_squared_distance").to_csv(TAB / "q3_distance_share_unscaled.csv")
print("   Share of the unscaled distance:", ", ".join(f"{k} {v:.1%}" for k, v in share.head(3).items()))

fig, ax = plt.subplots(figsize=(4.2, 2.6))
top = share.head(6)[::-1]
ax.barh(top.index, top.values, color=BLUE)
ax.set_xlabel("Share of squared Euclidean distance (unscaled)")
ax.set_title("Without scaling, three dollar columns decide who is a neighbour", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q3_distance_share.png")
plt.close(fig)

# ---------------------------------------------------------------- Q4: scale everything, then k=5
for name, scaler in [("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler())]:
    s = scaler.fit(X_tr)
    m = KNeighborsClassifier(n_neighbors=5).fit(s.transform(X_tr), y_tr)
    results[f"KNN k=5, {name} on all columns"] = rates(y_te, m.predict(s.transform(X_te)))
    r = results[f"KNN k=5, {name} on all columns"]
    print(f"Q4. {name} on every column incl. dummies: accuracy {r['accuracy']:.1%}, recall {r['recall']:.1%}")
# Why StandardScaler on dummies hurts: a rare dummy becomes a huge number when it is 1.
z = StandardScaler().fit(X_tr)
dummy_z = pd.Series((1 - z.mean_) / z.scale_, index=X.columns)[[c for c in X.columns if c not in NUM]]
dummy_z.sort_values(ascending=False).rename("z_value_when_1").to_csv(TAB / "q4_dummy_z_when_one.csv")
print(f"   Largest dummy after StandardScaler: {dummy_z.idxmax()} = {dummy_z.max():.1f} s.d. when it is 1 "
      f"({int(X[dummy_z.idxmax()].sum())} customers in the file)")

# ---------------------------------------------------------------- Q8 pipeline (built here, used for Q5-Q6)
pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pipe = Pipeline([("pre", pre), ("knn", KNeighborsClassifier())])

# ---------------------------------------------------------------- Q5: grid search over k
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(pipe, {"knn__n_neighbors": K_GRID}, cv=cv, scoring="accuracy", return_train_score=True)
grid.fit(Xr_tr, y_tr)
cvres = pd.DataFrame({"k": K_GRID,
                      "train_accuracy": grid.cv_results_["mean_train_score"],
                      "validation_accuracy": grid.cv_results_["mean_test_score"],
                      "validation_sd": grid.cv_results_["std_test_score"]})
cvres.to_csv(TAB / "q5_grid_k.csv", index=False)
best_k = grid.best_params_["knn__n_neighbors"]
print("Q5. k  train  validation")
for _, r in cvres.iterrows():
    print(f"   {int(r.k):>2}  {r.train_accuracy:.1%}  {r.validation_accuracy:.1%}")
print(f"   Best k = {best_k}")

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(cvres.k, cvres.train_accuracy, "o-", color=GREY, label="train (CV folds)")
ax.plot(cvres.k, cvres.validation_accuracy, "o-", color=BLUE, label="validation (CV folds)")
ax.axhline(1 - y_tr.mean(), ls=":", color=RED, lw=1, label='always "stays"')
ax.axvline(best_k, ls="--", color=GREY, lw=0.8)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xticks(K_GRID, [str(k) for k in K_GRID])
ax.set_xlabel("k (number of neighbours, log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Small k overfits, large k underfits", fontsize=8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q5_accuracy_vs_k.png")
plt.close(fig)

# ---------------------------------------------------------------- Q6: confusion matrix of the best model
best = grid.best_estimator_
p_te = best.predict_proba(Xr_te)[:, 1]
results[f"Pipeline, k={best_k} (Q8)"] = rates(y_te, (p_te >= 0.5).astype(int))
results[f"Pipeline, k={best_k}, alarm at 20%"] = rates(y_te, (p_te >= 0.2).astype(int))
auc_best = roc_auc_score(y_te, p_te)
r = results[f"Pipeline, k={best_k} (Q8)"]
print(f"Q6. Best pipeline on test: accuracy {r['accuracy']:.1%}, precision {r['precision']:.1%}, "
      f"recall {r['recall']:.1%}, AUC {auc_best:.3f}")
r2 = results[f"Pipeline, k={best_k}, alarm at 20%"]
print(f"   Alarm at 20%: recall {r2['recall']:.1%}, precision {r2['precision']:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
for ax, (title, key) in zip(axes, [("cut-off 50%", f"Pipeline, k={best_k} (Q8)"),
                                   ("cut-off 20%", f"Pipeline, k={best_k}, alarm at 20%")]):
    m = results[key]
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    ax.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["stays", "churns"])
    ax.set_yticks([0, 1], ["stays", "churns"])
    ax.set_xlabel("Predicted")
    ax.set_title(f"k={best_k}, {title}", fontsize=8)
axes[0].set_ylabel("Actual")
fig.tight_layout()
fig.savefig(FIG / "q6_confusion.png")
plt.close(fig)

# ---------------------------------------------------------------- Q7: curse of dimensionality
# Ratio of nearest to farthest distance from a random query point among n uniform points, as dimension grows.
dims = [2, 5, 10, 20, 37, 100]
rows = []
for d in dims:
    pts = rs.random((2000, d))
    q = rs.random((200, d))
    dist = np.sqrt(((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    rows.append((d, (dist.min(1) / dist.max(1)).mean()))
curse = pd.DataFrame(rows, columns=["dimensions", "nearest_over_farthest"])
curse.to_csv(TAB / "q7_curse_of_dimensionality.csv", index=False)
print("Q7. nearest/farthest distance ratio:", ", ".join(f"d={d}: {v:.2f}" for d, v in rows))

# Numeric columns only (drops the 23 dummy columns).
num_pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())])
g2 = GridSearchCV(num_pipe, {"knn__n_neighbors": K_GRID}, cv=cv, scoring="accuracy").fit(Xr_tr[NUM], y_tr)
results[f"Numeric columns only, k={g2.best_params_['knn__n_neighbors']}"] = rates(y_te, g2.predict(Xr_te[NUM]))
r = results[f"Numeric columns only, k={g2.best_params_['knn__n_neighbors']}"]
print(f"   Numeric columns only (k={g2.best_params_['knn__n_neighbors']}): accuracy {r['accuracy']:.1%}, recall {r['recall']:.1%}")

# ---------------------------------------------------------------- Q8: pipeline refit on raw data
manual = ColumnTransformer([("num", StandardScaler(), NUM),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]).fit(Xr_tr)
m = KNeighborsClassifier(n_neighbors=best_k).fit(manual.transform(Xr_tr), y_tr)
same = np.array_equal(m.predict(manual.transform(Xr_te)), best.predict(Xr_te))
print(f"Q8. Manual steps and Pipeline give identical test predictions: {same}")

summary = pd.DataFrame(results).T[["accuracy", "recall", "precision", "TP", "FN", "FP", "TN"]]
summary.to_csv(TAB / "q8_model_comparison.csv")
print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
