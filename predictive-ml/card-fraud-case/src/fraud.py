"""Catching card fraud without labels: Isolation Forest against Random Forest and XGBoost.

Answers Q1-Q7 of the case in order. Every model is fitted on day 1 and judged on day 2. The decision is the fraud
team's: review the N transactions with the highest score each day. A review costs EUR 10, a missed fraud EUR 500.
Every figure goes to output/figures and every table to output/tables.

    uv run python src/fraud.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
N = 100
REVIEW, MISS = 10, 500
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#2b5c8a", "#c0392b", "#9a9a9a", "#3a7d44"
GRID = np.unique(np.r_[np.arange(10, 1000, 10), np.arange(1000, 5001, 10)])


def caught_curve(y, score):
    """Frauds among the top-n scores, for every n from 1 to len(y)."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(np.asarray(y)[order])


def review(y, score, n):
    """Confusion matrix of reviewing the top n."""
    tp = int(caught_curve(y, score)[n - 1])
    fraud = int(np.sum(y))
    return {"N": n, "TP": tp, "FP": n - tp, "FN": fraud - tp, "TN": len(y) - n - (fraud - tp),
            "precision": tp / n, "recall": tp / fraud}


def best_n(y, score, cost):
    """The N on the grid that minimises review cost + missed-fraud cost."""
    c = caught_curve(y, score)
    total = cost * GRID + MISS * (np.sum(y) - c[GRID - 1])
    return int(GRID[np.argmin(total)])


def day_cost(y, score, n, cost):
    caught = int(caught_curve(y, score)[n - 1])
    return caught, cost * n + MISS * (int(np.sum(y)) - caught)


# Q1: the data
path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
df = pd.read_csv(path / "creditcard.csv")
df["hour"] = (df["Time"] // 3600) % 24
df["day"] = np.where(df["Time"] < 86_400, 1, 2)
df["log_amount"] = np.log1p(df["Amount"])
feat = [f"V{i}" for i in range(1, 29)] + ["log_amount"]
d1, d2 = df[df.day == 1], df[df.day == 2]
print(f"Q1 {len(df):,} transactions, {df.Class.sum()} frauds ({df.Class.mean():.3%}); "
      f"'all legit' accuracy {1 - df.Class.mean():.2%}")
print(f"   day 1: {len(d1):,} / {d1.Class.sum()} frauds; day 2: {len(d2):,} / {d2.Class.sum()} frauds")
amounts = df.groupby("Class")["Amount"].describe()[["count", "mean", "50%", "max"]]
amounts.round(2).to_csv(TAB / "q1_amounts.csv")
print(amounts.round(2))
by_hour = df.groupby("hour")["Class"].agg(["mean", "sum", "count"]).rename(columns={"mean": "fraud_rate"})
by_hour.to_csv(TAB / "q1_fraud_rate_by_hour.csv")
print("   top hours by fraud rate\n", by_hour.sort_values("fraud_rate", ascending=False).head(4))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
bins = np.linspace(0, np.log1p(df.Amount.max()), 40)
for c, col, lab in [(0, GREY, "legitimate"), (1, RED, "fraud")]:
    ax[0].hist(df.loc[df.Class == c, "log_amount"], bins=bins, density=True, alpha=0.6, color=col, label=lab)
ax[0].set(title="log(1 + amount)", yticks=[])
ax[0].legend(frameon=False)
ax[1].bar(by_hour.index, by_hour.fraud_rate * 100, color=BLUE)
ax[1].set(title="Fraud rate by hour (%)", xlabel="hour of day")
fig.tight_layout()
fig.savefig(FIG / "q1_amount_hour.png")
plt.close(fig)

scaler = StandardScaler().fit(d1[feat])
X1, X2 = scaler.transform(d1[feat]), scaler.transform(d2[feat])
y1, y2 = d1.Class.values, d2.Class.values

# Q2: Isolation Forest on day 1, no labels
iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X1)
s_iso1, s_iso2 = -iso.score_samples(X1), -iso.score_samples(X2)  # higher = more anomalous


def mean_depth(model, X):
    """Average number of splits to isolate each row, over the trees (path length incl. the leaf correction)."""
    from sklearn.ensemble._iforest import _average_path_length

    depths = np.zeros(len(X))
    for tree, feats in zip(model.estimators_, model.estimators_features_):
        leaves = tree.apply(X[:, feats])
        node_depth = tree.tree_.compute_node_depths() - 1
        depths += node_depth[leaves] + _average_path_length(tree.tree_.n_node_samples[leaves])
    return depths / len(model.estimators_)


depth = mean_depth(iso, X2)
print(f"Q2 mean splits to isolate on day 2: fraud {depth[y2 == 1].mean():.1f}, legit {depth[y2 == 0].mean():.1f}")
pd.DataFrame({"class": ["fraud", "legit"], "mean_splits": [depth[y2 == 1].mean(), depth[y2 == 0].mean()],
              "mean_score": [s_iso2[y2 == 1].mean(), s_iso2[y2 == 0].mean()]}).round(3).to_csv(
    TAB / "q2_depth_score.csv", index=False)
fig, ax = plt.subplots(figsize=(3.6, 2.3))
b = np.linspace(s_iso2.min(), s_iso2.max(), 50)
ax.hist(s_iso2[y2 == 0], bins=b, density=True, alpha=0.6, color=GREY, label="legitimate")
ax.hist(s_iso2[y2 == 1], bins=b, density=True, alpha=0.6, color=RED, label="fraud")
ax.set(title="Anomaly score, day 2", yticks=[])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "q2_score_distribution.png")
plt.close(fig)

# Q3: top 100
r = review(y2, s_iso2, N)
print(f"Q3 top {N}: {r}; random list expects {N * y2.mean():.2f} frauds")
pd.DataFrame([r]).round(4).to_csv(TAB / "q3_top100_isolation_forest.csv", index=False)

# Q4: precision and recall as N grows
q4 = pd.DataFrame([review(y2, s_iso2, n) for n in [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]])
q4.round(4).to_csv(TAB / "q4_precision_recall_by_n.csv", index=False)
print("Q4\n", q4[["N", "TP", "precision", "recall"]].round(3).to_string(index=False))

# Q5: cost-optimal N chosen on day 1, applied on day 2
rows = []
for cost in [2, 5, 10, 20, 50]:
    n = best_n(y1, s_iso1, cost)
    caught, total = day_cost(y2, s_iso2, n, cost)
    rows.append({"review_cost": cost, "N_from_day1": n, "frauds_caught_day2": caught, "cost_day2": total,
                 "do_nothing_day2": MISS * int(y2.sum())})
q5 = pd.DataFrame(rows)
q5["saving"] = q5.do_nothing_day2 - q5.cost_day2
q5.to_csv(TAB / "q5_cost_by_review_cost.csv", index=False)
print("Q5\n", q5.to_string(index=False))

# Q6: what the top alerts look like
top = np.argsort(-s_iso2)[:N]
alerts = d2.iloc[top]
z = pd.DataFrame(X2[top], columns=feat)
far = z.drop(columns="log_amount").abs().mean().sort_values(ascending=False).head(5)
found = z[alerts.Class.values == 1].drop(columns="log_amount").mean().sort_values().head(5)
q6 = {
    "median_amount_top100": alerts.Amount.median(), "median_amount_all_day2": d2.Amount.median(),
    "median_amount_frauds_found": alerts.loc[alerts.Class == 1, "Amount"].median(),
    "frauds_in_top10": int(alerts.Class.values[:10].sum()),
    "hours_top100_mode": int(alerts.hour.mode()[0]),
}
pd.Series(q6).to_csv(TAB / "q6_top_alerts.csv")
pd.DataFrame({"furthest_from_normal_top100 (|z|)": far.round(2)}).to_csv(TAB / "q6_top_alert_features.csv")
pd.DataFrame({"frauds_found_mean_z": found.round(2)}).to_csv(TAB / "q6_found_fraud_features.csv")
print("Q6", q6, "\n   furthest", far.round(2).to_dict(), "\n   frauds found", found.round(2).to_dict())

# Q7: the labels arrive
w = (y1 == 0).sum() / (y1 == 1).sum()
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
                            random_state=SEED, n_jobs=-1).fit(X1, y1)
xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=w,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                       eval_metric="aucpr").fit(X1, y1)
scores = {"Isolation Forest": (s_iso1, s_iso2),
          "Random Forest": (rf.predict_proba(X1)[:, 1], rf.predict_proba(X2)[:, 1]),
          "XGBoost": (xg.predict_proba(X1)[:, 1], xg.predict_proba(X2)[:, 1])}
# the supervised models fit day 1 almost perfectly, so N is chosen on out-of-fold day-1 scores
from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scores["Random Forest"] = (cross_val_predict(rf, X1, y1, cv=cv, method="predict_proba")[:, 1],
                           scores["Random Forest"][1])
scores["XGBoost"] = (cross_val_predict(xg, X1, y1, cv=cv, method="predict_proba")[:, 1], scores["XGBoost"][1])

rows = []
for name, (s1, s2) in scores.items():
    top100 = review(y2, s2, N)
    n = best_n(y1, s1, REVIEW)
    caught, total = day_cost(y2, s2, n, REVIEW)
    rows.append({"model": name, "TP_top100": top100["TP"], "FP_top100": top100["FP"],
                 "precision_top100": top100["precision"], "recall_top100": top100["recall"],
                 "N_cost_optimal": n, "caught_at_N": caught, "cost_day2": total})
q7 = pd.DataFrame(rows)
q7.round(3).to_csv(TAB / "q7_supervised_vs_anomaly.csv", index=False)
print("Q7\n", q7.round(3).to_string(index=False))
print(f"   doing nothing on day 2: EUR {MISS * y2.sum():,}")

fig, ax = plt.subplots(figsize=(4.0, 2.5))
ns = np.arange(1, 5001)
for (name, (_, s2)), col in zip(scores.items(), [GREY, BLUE, RED]):
    ax.plot(ns, caught_curve(y2, s2)[:5000], color=col, label=name)
ax.axhline(y2.sum(), color="black", lw=0.6, ls=":")
ax.set(xscale="log", xlabel="alerts reviewed on day 2", ylabel="frauds caught", title="Frauds caught vs alerts")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q7_frauds_caught.png")
plt.close(fig)
