"""Predicting employee attrition with decision trees (IBM HR Analytics data).

Answers Q1-Q8 of the case in order. Every figure goes to output/figures and every table to
output/tables, so the report can be rebuilt from this script alone.

    uv run python src/attrition.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#2b5c8a", "#c0392b", "#9a9a9a"


def rates(y_true, y_pred):
    """Accuracy, TP rate (recall on leavers) and FP rate (stayers wrongly flagged)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp_rate": tp / (tp + fn),
        "fp_rate": fp / (fp + tn),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


# ---------------------------------------------------------------- Q1: load and explore
path = Path(kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset"))
df = pd.read_csv(path / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
print(f"Q1. {len(df)} employees, {df.shape[1]} columns, attrition rate {df['Attrition'].mean():.1%}")

# Columns that carry no information: constant across all rows, or a row identifier.
constant = [c for c in df.columns if df[c].nunique() == 1]
dropped = constant + ["EmployeeNumber"]
print("   Dropped (constant or ID):", dropped)

ORDINAL = ["Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction",
           "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]
NOMINAL = [c for c in df.select_dtypes(include=["object", "string"]).columns if c not in dropped]
CONTINUOUS = [c for c in df.select_dtypes("number").columns
              if c not in ORDINAL + dropped + ["Attrition"]]
types = pd.DataFrame(
    [(c, "nominal (text)") for c in NOMINAL]
    + [(c, "ordinal (integer code)") for c in ORDINAL]
    + [(c, "numeric (count or amount)") for c in CONTINUOUS]
    + [(c, "drop (constant or ID)") for c in dropped],
    columns=["variable", "type"],
)
types.to_csv(TAB / "q1_q2_variable_types.csv", index=False)

# Attrition rate by level of each categorical variable: the gap between levels is the "visual relationship".
cat_vars = NOMINAL + ORDINAL
gaps = []
for c in cat_vars:
    r = df.groupby(c)["Attrition"].agg(["mean", "size"])
    r = r[r["size"] >= 30]  # ignore tiny levels so one small group does not dominate
    gaps.append((c, r["mean"].max() - r["mean"].min(), r["mean"].idxmax(), r["mean"].max()))
gaps = pd.DataFrame(gaps, columns=["variable", "gap", "riskiest_level", "riskiest_rate"]).sort_values("gap", ascending=False)
gaps.to_csv(TAB / "q1_categorical_gaps.csv", index=False)

# Numeric variables: standardized difference in means between leavers and stayers.
num_gap = []
for c in CONTINUOUS:
    m1, m0 = df.loc[df.Attrition == 1, c].mean(), df.loc[df.Attrition == 0, c].mean()
    num_gap.append((c, m1, m0, (m1 - m0) / df[c].std()))
num_gap = pd.DataFrame(num_gap, columns=["variable", "mean_leavers", "mean_stayers", "std_diff"])
num_gap = num_gap.reindex(num_gap["std_diff"].abs().sort_values(ascending=False).index)
num_gap.to_csv(TAB / "q1_numeric_gaps.csv", index=False)
print(gaps.head(6).to_string(index=False))
print(num_gap.head(6).to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), gridspec_kw={"width_ratios": [2, 3, 3, 4]})
for ax, c in zip(axes, ["OverTime", "MaritalStatus", "BusinessTravel", "JobLevel"]):
    r = df.groupby(c)["Attrition"].mean()
    ax.bar(r.index.astype(str), r.values, color=[RED if v == r.max() else BLUE for v in r.values])
    ax.axhline(df.Attrition.mean(), color=GREY, ls="--", lw=0.8)
    ax.set_title(c)
    ax.set_ylim(0, 0.45)
    ax.tick_params(axis="x", labelsize=7)
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=7)
axes[0].set_ylabel("Attrition rate")
fig.tight_layout()
fig.savefig(FIG / "q1_attrition_by_group.png")
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
for ax, c in zip(axes, ["MonthlyIncome", "TotalWorkingYears", "Age", "YearsAtCompany"]):
    ax.boxplot([df.loc[df.Attrition == 0, c], df.loc[df.Attrition == 1, c]], showfliers=False)
    ax.set_xticks([1, 2], ["Stayed", "Left"])
    ax.set_title(c)
fig.tight_layout()
fig.savefig(FIG / "q1_numeric_boxplots.png")
plt.close(fig)

# ---------------------------------------------------------------- Q2: features, encoding, split
FEATURES = [
    # work pattern and pay
    "OverTime", "MonthlyIncome", "StockOptionLevel", "JobLevel", "BusinessTravel",
    # career stage
    "Age", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
    "YearsSinceLastPromotion", "NumCompaniesWorked",
    # satisfaction scores (ordinal 1-4)
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction",
    # life outside work
    "MaritalStatus", "DistanceFromHome",
    # where they sit
    "Department",
]
X = df[FEATURES].copy()
X["OverTime"] = (X["OverTime"] == "Yes").astype(int)  # binary: one 0/1 column is enough
# BusinessTravel is ordered (none < rarely < frequently), so an integer code keeps its meaning.
X["BusinessTravel"] = X["BusinessTravel"].map({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2})
# MaritalStatus and Department have no order: one-hot, so no model reads "Single > Married".
X = pd.get_dummies(X, columns=["MaritalStatus", "Department"], drop_first=False, dtype=int)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nQ2. {X.shape[1]} model columns. Train {len(X_train)} ({y_train.mean():.1%} leave), "
      f"test {len(X_test)} ({y_test.mean():.1%} leave)")

# ---------------------------------------------------------------- Q3: accuracy vs depth
DEPTHS = [2, 3, 4, 5, 7, 10, None]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for d in DEPTHS:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_train, y_train)
    cv_acc = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), X_train, y_train, cv=cv)
    rows.append({"max_depth": "None" if d is None else d, "train_acc": t.score(X_train, y_train),
                 "test_acc": t.score(X_test, y_test), "cv_train_acc": cv_acc.mean(),
                 "actual_depth": t.get_depth(), "leaves": t.get_n_leaves()})
depth_tab = pd.DataFrame(rows)
depth_tab.to_csv(TAB / "q3_depth.csv", index=False)
print("\nQ3.\n", depth_tab.round(3).to_string(index=False))

# How fragile is "the optimal depth"? Redo the 80/20 split with 50 different seeds and count which depth wins.
wins = []
for s in range(50):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
    accs = [DecisionTreeClassifier(max_depth=dd, random_state=SEED).fit(a, c).score(b, d) for dd in DEPTHS]
    wins.append(str(DEPTHS[int(np.argmax(accs))]))
wins = pd.Series(wins).value_counts().reindex([str(d) for d in DEPTHS], fill_value=0)
wins.rename("times_best_of_50_splits").to_csv(TAB / "q3_depth_wins_50_splits.csv")
print("   Depth that wins the test set across 50 random splits:", wins.to_dict())

best_row = depth_tab.loc[depth_tab["test_acc"].idxmax()]
BEST = int(best_row["max_depth"])
print(f"   Best depth on the test set: {BEST}")

fig, ax = plt.subplots(figsize=(5, 3))
xs = range(len(DEPTHS))
ax.plot(xs, depth_tab.train_acc, "o-", color=BLUE, label="Train")
ax.plot(xs, depth_tab.test_acc, "o-", color=RED, label="Test")
ax.plot(xs, depth_tab.cv_train_acc, "o--", color=GREY, label="5-fold CV (train set)")
ax.axhline(1 - y_test.mean(), color="black", ls=":", lw=0.8)
ax.text(len(DEPTHS) - 1, 1 - y_test.mean() + 0.005, "always 'Stay'", ha="right", fontsize=7)
ax.set_xticks(list(xs), [str(d) for d in depth_tab.max_depth])
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q3_accuracy_vs_depth.png")
plt.close(fig)

# ---------------------------------------------------------------- Q4: cross-validation of the best depth
cv_full = cross_val_score(DecisionTreeClassifier(max_depth=BEST, random_state=SEED), X, y, cv=cv)
cv_tab = pd.DataFrame({"fold": range(1, 6), "accuracy": cv_full})
cv_tab.to_csv(TAB / "q4_cv_folds.csv", index=False)
print(f"\nQ4. depth={BEST}: folds {np.round(cv_full, 3)}, mean {cv_full.mean():.3f}, sd {cv_full.std():.3f}; "
      f"single test {best_row['test_acc']:.3f}")

# ---------------------------------------------------------------- Q5: confusion matrix
tree = DecisionTreeClassifier(max_depth=BEST, random_state=SEED).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
r_tree = rates(y_test, pred_tree)
print(f"\nQ5. Tree depth {BEST}: {r_tree}")

fig, ax = plt.subplots(figsize=(2.8, 2.5))
cm = confusion_matrix(y_test, pred_tree)
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Stay", "Leave"])
ax.set_yticks([0, 1], ["Stay", "Leave"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "q5_confusion_tree.png")
plt.close(fig)

# ---------------------------------------------------------------- Q6: feature importances
imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.rename("importance").to_csv(TAB / "q6_importances.csv")
print("\nQ6.\n", imp[imp > 0].round(3).to_string())

top = imp[imp > 0].head(10)[::-1]
fig, ax = plt.subplots(figsize=(4.2, 3))
ax.barh(top.index, top.values, color=BLUE)
ax.set_xlabel("Gini importance")
fig.tight_layout()
fig.savefig(FIG / "q6_importances.png")
plt.close(fig)

# ---------------------------------------------------------------- Q7: depth-3 tree as business rules
tree3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)
rules = export_text(tree3, feature_names=list(X.columns), show_weights=True)
(TAB / "q7_tree_depth3.txt").write_text(rules)
print("\nQ7.\n", rules)

# Leaf table: path, size and attrition rate of each leaf, in the training data.
t = tree3.tree_
leaves = []


def walk(node, path):
    if t.children_left[node] == -1:
        n = t.n_node_samples[node]
        rate = t.value[node][0][1]  # sklearn stores class fractions per node
        leaves.append({"rule": " AND ".join(path), "n_train": n, "attrition_rate": rate,
                       "prediction": "Leave" if rate > 0.5 else "Stay"})
        return
    f, thr = X.columns[t.feature[node]], t.threshold[node]
    walk(t.children_left[node], path + [f"{f} <= {thr:.1f}"])
    walk(t.children_right[node], path + [f"{f} > {thr:.1f}"])


walk(0, [])
leaf_tab = pd.DataFrame(leaves).sort_values("attrition_rate", ascending=False)
leaf_tab.to_csv(TAB / "q7_leaves.csv", index=False)
print(leaf_tab.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 5))
plot_tree(tree3, feature_names=list(X.columns), class_names=["Stay", "Leave"], filled=True,
          impurity=False, proportion=True, rounded=True, fontsize=7, ax=ax)
fig.tight_layout()
fig.savefig(FIG / "q7_tree_depth3.png")
plt.close(fig)

# ---------------------------------------------------------------- Q8: logistic regression on the same features
# Scaling matters for the regularised logit (it penalises coefficients, so units must be comparable);
# it would not change the tree at all.
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
pred_logit = logit.predict(X_test)
r_logit = rates(y_test, pred_logit)
cv_logit = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv)

# Both models output probabilities; AUC measures how well they rank leavers above stayers at any threshold.
auc_tree = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
auc_logit = roc_auc_score(y_test, logit.predict_proba(X_test)[:, 1])

# Same models with the alarm threshold lowered from 50% to 25%: what HR would actually do with a 16% base rate.
r_tree_25 = rates(y_test, (tree.predict_proba(X_test)[:, 1] >= 0.25).astype(int))
r_logit_25 = rates(y_test, (logit.predict_proba(X_test)[:, 1] >= 0.25).astype(int))
r_naive = rates(y_test, np.zeros_like(y_test))

comp = pd.DataFrame(
    {f"Tree (depth {BEST})": r_tree, "Logistic regression": r_logit, "Always 'Stay'": r_naive,
     f"Tree, threshold 25%": r_tree_25, "Logit, threshold 25%": r_logit_25}
).T
comp["auc"] = [auc_tree, auc_logit, 0.5, auc_tree, auc_logit]
comp["cv_accuracy"] = [cv_full.mean(), cv_logit.mean(), 1 - y.mean(), np.nan, np.nan]
comp.to_csv(TAB / "q8_model_comparison.csv")
print("\nQ8.\n", comp.round(3).to_string())

coef = pd.Series(logit[-1].coef_[0], index=X.columns).sort_values()
coef.rename("std_coef").to_csv(TAB / "q8_logit_coefficients.csv")
print("\n   Logit coefficients (per 1 sd):\n", coef.round(2).to_string())
