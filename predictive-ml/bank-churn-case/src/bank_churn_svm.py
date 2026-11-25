"""Who is about to leave the bank? Support Vector Machines for a retention call list (Kaggle Churn Modelling).

Answers Q1-Q7 of the case in order. The retention team calls 10% of the customers: every model is judged on how
many leavers its call list reaches. Every figure goes to output/figures and every table to output/tables.

    uv run python src/bank_churn_svm.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 0  # the case fixes random_state=0
CALL = 0.10  # the team calls 10% of the customers
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#2b5c8a", "#c0392b", "#9a9a9a", "#3a7d44"


def call_list(score, share=CALL):
    """Boolean mask: the `share` of customers with the highest score."""
    n = int(round(share * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-np.asarray(score), kind="stable")[:n]] = True
    return mask


def list_stats(y, score):
    y = np.asarray(y)
    c = call_list(score)
    tp, fp = int((c & (y == 1)).sum()), int((c & (y == 0)).sum())
    fn, tn = int((~c & (y == 1)).sum()), int((~c & (y == 0)).sum())
    return {"called": int(c.sum()), "leavers_reached (TP)": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / c.sum(), "recall": tp / (y == 1).sum()}


# ---------------------------------------------------------------- Q1: explore
path = Path(kagglehub.dataset_download("shrutimechlearn/churn-modelling"))
df = pd.read_csv(path / "Churn_Modelling.csv").drop(columns=["RowNumber", "CustomerId", "Surname"])
print(f"Q1. {len(df)} customers, {df['Exited'].mean():.1%} left")
groups = {
    "Age": pd.cut(df["Age"], [17, 30, 40, 50, 60, 70, 100], labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71+"]),
    "NumOfProducts": df["NumOfProducts"],
    "Geography": df["Geography"],
    "IsActiveMember": df["IsActiveMember"].map({0: "inactive", 1: "active"}),
}
rows = []
for var, g in groups.items():
    r = df.groupby(g, observed=True)["Exited"].agg(["mean", "size"])
    for lvl, v in r.iterrows():
        rows.append((var, str(lvl), v["mean"], int(v["size"])))
explore = pd.DataFrame(rows, columns=["variable", "group", "share_left", "customers"])
explore.to_csv(TAB / "q1_share_left_by_group.csv", index=False)
for var in groups:
    e = explore[explore.variable == var]
    print(f"   {var}: " + ", ".join(f"{g} {s:.1%} (n={n})" for g, s, n in e[["group", "share_left", "customers"]].values))

fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3), sharey=True)
for ax, var in zip(axes, groups):
    e = explore[explore.variable == var]
    ax.bar(e.group, e.share_left, color=BLUE)
    ax.axhline(df["Exited"].mean(), color=RED, ls=":", lw=1)
    ax.set_title(var, fontsize=8)
    ax.tick_params(axis="x", labelsize=6, rotation=30)
axes[0].set_ylabel("Share who left")
fig.tight_layout()
fig.savefig(FIG / "q1_share_left.png")
plt.close(fig)

# ---------------------------------------------------------------- split and features
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT = ["Geography", "Gender"]
X, y = df[NUM + CAT], df["Exited"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
print(f"   Train {len(X_tr)} / test {len(X_te)}; a random call list of {int(CALL * len(X_te))} reaches "
      f"{CALL * len(X_te) * y_te.mean():.0f} leavers on average")


def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(drop="first"), CAT)])


# ---------------------------------------------------------------- Q2: two features
TWO = ["Age", "NumOfProducts"]
two = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
       "SVM, RBF kernel": make_pipeline(StandardScaler(), SVC(kernel="rbf"))}
q2 = {}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
aa, pp = np.meshgrid(np.linspace(18, 92, 300), np.linspace(0.5, 4.5, 200))
grid = pd.DataFrame({"Age": aa.ravel(), "NumOfProducts": pp.ravel()})
rs = np.random.default_rng(SEED)
jit = rs.uniform(-0.25, 0.25, len(X_te))
for ax, (name, m) in zip(axes, two.items()):
    m.fit(X_tr[TWO], y_tr)
    s_te = m.decision_function(X_te[TWO])
    q2[name] = list_stats(y_te, s_te)
    cut = np.sort(s_te)[::-1][int(round(CALL * len(s_te))) - 1]  # lowest score that makes the list
    zone = (m.decision_function(grid) >= cut).reshape(aa.shape)
    ax.contourf(aa, pp, zone, levels=[0.5, 1.5], colors=[RED], alpha=0.15)
    for v, c, lab in [(0, GREY, "stayed"), (1, RED, "left")]:
        k = y_te == v
        ax.scatter(X_te["Age"][k], X_te["NumOfProducts"][k] + jit[k], s=2, color=c, alpha=0.4, label=lab)
    ax.set_title(f"{name}: list reaches {q2[name]['leavers_reached (TP)']} leavers", fontsize=8)
    ax.set_xlabel("Age")
    ax.set_yticks([1, 2, 3, 4])
axes[0].set_ylabel("Number of products (jittered)")
axes[0].legend(frameon=False, fontsize=7, markerscale=4, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "q2_call_zones_two_features.png")
plt.close(fig)
pd.DataFrame(q2).T.to_csv(TAB / "q2_two_feature_lists.csv")
print("Q2. Age + NumOfProducts only, test call list:",
      ", ".join(f"{k} {v['leavers_reached (TP)']}" for k, v in q2.items()))

# ---------------------------------------------------------------- Q3: all features, four models
models = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "SVM, linear": SVC(kernel="linear"),
    "SVM, polynomial degree 2": SVC(kernel="poly", degree=2),
    "SVM, RBF": SVC(kernel="rbf"),
}
q3, scores_te = [], {}
for name, est in models.items():
    m = make_pipeline(prep(), est).fit(X_tr, y_tr)
    s_tr, s_te = m.decision_function(X_tr), m.decision_function(X_te)
    scores_te[name] = s_te
    for part, yy, ss in [("train", y_tr, s_tr), ("test", y_te, s_te)]:
        q3.append({"model": name, "set": part, **list_stats(yy, ss)})
    if name == "SVM, RBF":
        rbf_model = m
q3 = pd.DataFrame(q3)
q3.to_csv(TAB / "q3_call_lists_all_features.csv", index=False)
print("Q3. Call list, all features:")
print(q3[["model", "set", "called", "leavers_reached (TP)", "precision", "recall"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))

fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.0))
for ax, name in zip(axes, models):
    r = q3[(q3.model == name) & (q3.set == "test")].iloc[0]
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["leavers_reached (TP)"]]])
    ax.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, f"{v}", ha="center", va="center", fontsize=7, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not called", "called"], fontsize=6)
    ax.set_yticks([0, 1], ["stayed", "left"], fontsize=6)
    ax.set_title(name.replace("SVM, ", ""), fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q3_confusion_call_lists.png")
plt.close(fig)

# ---------------------------------------------------------------- Q4: C x gamma
Cs, gammas = [0.1, 1, 10], [0.01, 0.1, 1]
rec_tr = pd.DataFrame(index=Cs, columns=gammas, dtype=float)
rec_te = rec_tr.copy()
for C in Cs:
    for g in gammas:
        m = make_pipeline(prep(), SVC(kernel="rbf", C=C, gamma=g)).fit(X_tr, y_tr)
        rec_tr.loc[C, g] = list_stats(y_tr, m.decision_function(X_tr))["recall"]
        rec_te.loc[C, g] = list_stats(y_te, m.decision_function(X_te))["recall"]
rec_tr.index.name = rec_te.index.name = "C \\ gamma"
rec_tr.to_csv(TAB / "q4_recall_train.csv")
rec_te.to_csv(TAB / "q4_recall_test.csv")
print("Q4. Call-list recall, train:\n" + (rec_tr * 100).round(1).to_string())
print("   test:\n" + (rec_te * 100).round(1).to_string())

# ---------------------------------------------------------------- Q5: side by side
q5 = q3[q3.set == "test"].set_index("model")[["leavers_reached (TP)", "precision", "recall"]].astype(float)
for name, v in q2.items():
    q5.loc[f"{name}, Age + NumOfProducts only"] = [v["leavers_reached (TP)"], v["precision"], v["recall"]]
q5.loc["Random list (expected)"] = [CALL * len(y_te) * y_te.mean(), y_te.mean(), CALL]
q5.to_csv(TAB / "q5_model_comparison.csv")
print("Q5.\n" + q5.to_string(float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------- Q6: scores are not probabilities
naive_p = 1 / (1 + np.exp(-scores_te["SVM, RBF"]))
platt = CalibratedClassifierCV(make_pipeline(prep(), SVC(kernel="rbf")), method="sigmoid", cv=5).fit(X_tr, y_tr)
p_cal = platt.predict_proba(X_te)[:, 1]
overlap = (call_list(scores_te["SVM, RBF"]) & call_list(p_cal)).sum() / call_list(p_cal).sum()
print(f"Q6. Actual share of leavers (test) {y_te.mean():.1%}; logistic of the RBF score averages {naive_p.mean():.1%}; "
      f"Platt-calibrated {p_cal.mean():.1%}")
print(f"   Call lists from raw score and calibrated probability overlap {overlap:.0%}")
pd.DataFrame({"actual_share": [y_te.mean()], "logistic_of_score_mean": [naive_p.mean()],
              "platt_mean": [p_cal.mean()], "call_list_overlap": [overlap]}).to_csv(TAB / "q6_calibration.csv", index=False)

fig, ax = plt.subplots(figsize=(3.6, 2.9))
for p, lab, c in [(naive_p, "logistic of SVM score", RED), (p_cal, "Platt-calibrated", BLUE)]:
    frac, mean_p = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, "o-", ms=3, color=c, label=lab)
ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
ax.set_xlabel("Predicted probability of leaving")
ax.set_ylabel("Observed share who left")
ax.set_title("A score is not a probability until calibrated", fontsize=8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q6_calibration.png")
plt.close(fig)

# ---------------------------------------------------------------- Q7: rank by probability or by deposits at risk
bal = X_te["Balance"].to_numpy()
leaver_deposits = bal[y_te == 1].sum()
q7 = {}
for name, s in [("Ranked by probability of leaving", p_cal), ("Ranked by deposits at risk (p x Balance)", p_cal * bal)]:
    c = call_list(s)
    q7[name] = {"leavers_reached": int((c & (y_te == 1)).sum()),
                "share_of_leavers_deposits_reached": bal[c & (y_te == 1)].sum() / leaver_deposits,
                "customers_with_zero_balance_called": int((c & (bal == 0)).sum())}
q7 = pd.DataFrame(q7).T
q7.to_csv(TAB / "q7_probability_vs_deposits.csv")
print("Q7.\n" + q7.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"   Customers with zero balance in test: {(bal == 0).mean():.0%}")
