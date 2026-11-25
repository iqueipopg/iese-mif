"""Lending Club: who gets the loan? Decision trees, Random Forest and XGBoost on 2007-2011 loans.

Answers Q1-Q9 of the case in order. The decision rule is the credit committee's: approve the 80% of applicants
with the lowest predicted risk. Every model is judged on the default rate of the loans it approves.
Every figure goes to output/figures and every table to output/tables.

    uv run python src/lending.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
APPROVE = 0.80  # the committee approves the 80% with the lowest predicted risk
LGD = 0.70
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#2b5c8a", "#c0392b", "#9a9a9a", "#3a7d44"


def approved_mask(p, rate=APPROVE):
    """True for the `rate` share of applicants with the lowest predicted risk."""
    order = np.argsort(p, kind="stable")
    mask = np.zeros(len(p), dtype=bool)
    mask[order[: int(round(rate * len(p)))]] = True
    return mask


def book(y, p, amount, rate=APPROVE):
    """What the approval rule does to the loan book."""
    y = np.asarray(y)
    amount = np.asarray(amount)
    a = approved_mask(np.asarray(p), rate)
    lent = amount[a].sum()
    lost = (amount[a] * y[a]).sum() * LGD
    return {
        "default_rate_approved": y[a].mean(),
        "loss_per_100M": lost / lent * 100,  # $M lost per $100M lent
        "approved_defaults (FN)": int((a & (y == 1)).sum()),
        "approved_good (TN)": int((a & (y == 0)).sum()),
        "declined_defaults (TP)": int((~a & (y == 1)).sum()),
        "declined_good (FP)": int((~a & (y == 0)).sum()),
    }


def dr(y, p, rate=APPROVE):
    return np.asarray(y)[approved_mask(np.asarray(p), rate)].mean()


# ---------------------------------------------------------------- Q1: load, filter, explore
path = Path(kagglehub.dataset_download("samaxtech/lending-club-20072011-data"))
raw = pd.read_csv(path / "loans_2007.csv", low_memory=False)
df = raw[raw["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
df["default"] = (df["loan_status"] == "Charged Off").astype(int)
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)
df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)
df["emp_length"] = (df["emp_length"].replace({"< 1 year": "0", "10+ years": "10"})
                    .str.extract(r"(\d+)")[0].astype(float))
ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
df["credit_history_years"] = (df["issue_d"] - ecl).dt.days / 365.25
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
print(f"Q1. {len(raw)} rows in the file, {len(df)} loans with a known outcome, default rate {df['default'].mean():.1%}")

explore = []
bins = {
    "term_months": df["term_months"],
    "annual_inc": pd.qcut(df["annual_inc"], 5),
    "revol_util": pd.qcut(df["revol_util"], 5),
    "inq_last_6mths": df["inq_last_6mths"].clip(upper=4),
    "loan_amnt": pd.qcut(df["loan_amnt"], 5),
    "purpose": df["purpose"],
}
for var, b in bins.items():
    g = df.groupby(b, observed=True)["default"].agg(["mean", "size"])
    for lvl, r in g.iterrows():
        explore.append((var, str(lvl), r["mean"], int(r["size"])))
explore = pd.DataFrame(explore, columns=["variable", "group", "default_rate", "loans"])
explore.to_csv(TAB / "q1_default_by_group.csv", index=False)
for var in bins:
    e = explore[explore.variable == var]
    lo, hi = e.loc[e.default_rate.idxmin()], e.loc[e.default_rate.idxmax()]
    print(f"   {var}: lowest {lo.group} {lo.default_rate:.1%}, highest {hi.group} {hi.default_rate:.1%}")

fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.0), sharey=True)
for ax, var in zip(axes.ravel(), bins):
    e = explore[explore.variable == var]
    if var == "purpose":
        e = e.sort_values("default_rate").iloc[[0, 1, 2, -3, -2, -1]]
    labels = [g.replace("_", " ")[:14] for g in e.group]
    ax.bar(range(len(e)), e.default_rate, color=BLUE)
    ax.axhline(df["default"].mean(), color=RED, ls=":", lw=1)
    ax.set_xticks(range(len(e)), labels, rotation=35, ha="right", fontsize=6)
    ax.set_title(var, fontsize=8)
axes[0, 0].set_ylabel("Default rate")
axes[1, 0].set_ylabel("Default rate")
fig.tight_layout()
fig.savefig(FIG / "q1_default_by_group.png")
plt.close(fig)

# Features available on the day the application arrives. Geography and free text are excluded (case rules).
APP_NUM = ["loan_amnt", "term_months", "emp_length", "annual_inc", "dti", "credit_history_years", "delinq_2yrs",
           "inq_last_6mths", "open_acc", "total_acc", "revol_bal", "revol_util", "pub_rec",
           "pub_rec_bankruptcies", "loan_to_income"]
APP_CAT = ["purpose", "home_ownership", "verification_status"]
LEAK = ["total_pymnt", "total_rec_prncp", "recoveries", "last_pymnt_amnt", "collection_recovery_fee"]


def features(frame, extra=()):
    X = pd.get_dummies(frame[APP_NUM + APP_CAT + list(extra)], columns=APP_CAT, dtype=float)
    return X


X_all = features(df)
X_all = X_all.fillna(X_all[df["issue_d"] <= "2010-12-31"].median())
train = (df["issue_d"] <= "2010-12-31").to_numpy()
test = ~train
X_tr, X_te = X_all[train], X_all[test]
y_tr, y_te = df.loc[train, "default"].to_numpy(), df.loc[test, "default"].to_numpy()
amt_te = df.loc[test, "loan_amnt"].to_numpy()
print(f"   Train {train.sum()} loans (to Dec-2010), default {y_tr.mean():.1%}; "
      f"test {test.sum()} loans (2011), default {y_te.mean():.1%}; {X_all.shape[1]} features")

# ---------------------------------------------------------------- Q2: naive policies
rs = np.random.default_rng(SEED)
results = {"Approve everybody": book(y_te, np.zeros(len(y_te)), amt_te, rate=1.0)}
rand = [dr(y_te, rs.random(len(y_te))) for _ in range(200)]
results["Decline 20% at random"] = book(y_te, rs.random(len(y_te)), amt_te)
print(f"Q2. Approve all: {results['Approve everybody']['default_rate_approved']:.1%}; decline 20% at random: "
      f"{np.mean(rand):.1%} on average over 200 draws")
print(f"   'Always repays' is {1 - y_te.mean():.1%} accurate on test and declines nobody")

# ---------------------------------------------------------------- Q3: the leaky tree
X_leak = features(df, LEAK)
X_leak = X_leak.fillna(X_leak[train].median())
leaky = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(X_leak[train], y_tr)
p_leak = leaky.predict_proba(X_leak[test])[:, 1]
print(f"Q3. Depth-6 tree with post-loan columns: approved default rate {dr(y_te, p_leak):.2%}, "
      f"AUC {roc_auc_score(y_te, p_leak):.3f}")
imp_leak = pd.Series(leaky.feature_importances_, index=X_leak.columns).sort_values(ascending=False)
imp_leak.rename("importance").to_csv(TAB / "q3_leaky_tree_importances.csv")
print("   Top splits:", ", ".join(f"{k} {v:.2f}" for k, v in imp_leak.head(4).items()))

fig, ax = plt.subplots(figsize=(9, 3.6))
plot_tree(leaky, max_depth=2, feature_names=X_leak.columns, class_names=["paid", "default"], filled=True,
          impurity=False, proportion=True, precision=2, fontsize=6, ax=ax)
ax.set_title("Depth-6 tree with post-loan columns (first three levels): it reads the outcome, not the risk",
             fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q3_leaky_tree.png")
plt.close(fig)

# ---------------------------------------------------------------- Q4: clean tree and depth
tree6 = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(X_tr, y_tr)
p_tree = tree6.predict_proba(X_te)[:, 1]
results["Decision tree, depth 6"] = book(y_te, p_tree, amt_te)
print(f"Q4. Clean depth-6 tree: approved default rate {dr(y_te, p_tree):.1%}, AUC {roc_auc_score(y_te, p_tree):.3f}")
fig, ax = plt.subplots(figsize=(9, 3.4))
plot_tree(tree6, max_depth=2, feature_names=X_tr.columns, class_names=["paid", "default"], filled=True,
          impurity=False, proportion=True, precision=3, fontsize=6, ax=ax)
ax.set_title("Clean depth-6 tree, first three levels", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q4_clean_tree.png")
plt.close(fig)

depth_rows = []
for d in list(range(1, 21)) + [None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
    depth_rows.append((d if d else 99, dr(y_tr, t.predict_proba(X_tr)[:, 1]), dr(y_te, t.predict_proba(X_te)[:, 1]),
                       t.get_n_leaves()))
depth = pd.DataFrame(depth_rows, columns=["max_depth", "train_dr", "test_dr", "leaves"])
depth.to_csv(TAB / "q4_depth_curve.csv", index=False)
best_d = depth.loc[depth.test_dr.idxmin()]
print(f"   Best test depth {int(best_d.max_depth)} ({best_d.test_dr:.1%}); unconstrained tree: train "
      f"{depth.iloc[-1].train_dr:.1%}, test {depth.iloc[-1].test_dr:.1%}, {int(depth.iloc[-1].leaves)} leaves")

# ---------------------------------------------------------------- Q5: Random Forest
rf_curve = []
for n in [1, 5, 10, 25, 50, 100, 200, 300]:
    rf = RandomForestClassifier(n_estimators=n, max_depth=10, max_features="sqrt", random_state=SEED,
                                n_jobs=-1).fit(X_tr, y_tr)
    rf_curve.append((n, dr(y_tr, rf.predict_proba(X_tr)[:, 1]), dr(y_te, rf.predict_proba(X_te)[:, 1])))
rf_curve = pd.DataFrame(rf_curve, columns=["trees", "train_dr", "test_dr"])
rf_curve.to_csv(TAB / "q5_rf_trees_curve.csv", index=False)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, max_features="sqrt", oob_score=True,
                            random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
p_rf = rf.predict_proba(X_te)[:, 1]
results["Random Forest, 300 trees, depth 10"] = book(y_te, p_rf, amt_te)
oob_dr = dr(y_tr, rf.oob_decision_function_[:, 1])
print(f"Q5. RF 300 trees depth 10: test {dr(y_te, p_rf):.1%}, train {dr(y_tr, rf.predict_proba(X_tr)[:, 1]):.1%}, "
      f"OOB {oob_dr:.1%}, AUC {roc_auc_score(y_te, p_rf):.3f}")

mat_tr = pd.DataFrame(index=[3, 6, 10, 15], columns=[5, 25, 100, 300], dtype=float)
mat_te = mat_tr.copy()
for d in mat_tr.index:
    for n in mat_tr.columns:
        m = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
        mat_tr.loc[d, n] = dr(y_tr, m.predict_proba(X_tr)[:, 1])
        mat_te.loc[d, n] = dr(y_te, m.predict_proba(X_te)[:, 1])
mat_tr.index.name = mat_te.index.name = "max_depth \\ trees"
mat_tr.to_csv(TAB / "q5_rf_matrix_train.csv")
mat_te.to_csv(TAB / "q5_rf_matrix_test.csv")
print("   Train matrix (rows depth, columns trees):\n" + (mat_tr * 100).round(1).to_string())
print("   Test matrix:\n" + (mat_te * 100).round(1).to_string())

# ---------------------------------------------------------------- Q6: XGBoost with early stopping on 2010 H2
fit_part = (df.loc[train, "issue_d"] < "2010-07-01").to_numpy()
xgb_rows = []
for lr in [0.3, 0.1, 0.03]:
    es = xgb.XGBClassifier(n_estimators=3000, learning_rate=lr, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", early_stopping_rounds=50, random_state=SEED)
    es.fit(X_tr[fit_part], y_tr[fit_part], eval_set=[(X_tr[~fit_part], y_tr[~fit_part])], verbose=False)
    n_best = es.best_iteration + 1
    final = xgb.XGBClassifier(n_estimators=n_best, learning_rate=lr, max_depth=3, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED).fit(X_tr, y_tr)
    p = final.predict_proba(X_te)[:, 1]
    xgb_rows.append((lr, n_best, dr(y_te, p), roc_auc_score(y_te, p)))
    if lr == 0.1:
        xgb_final, p_xgb = final, p
xgb_lr = pd.DataFrame(xgb_rows, columns=["learning_rate", "trees_by_early_stopping", "test_dr", "test_auc"])
xgb_lr.to_csv(TAB / "q6_xgb_learning_rates.csv", index=False)
print("Q6. XGBoost, early stopping on 2010 H2:")
for r in xgb_rows:
    print(f"   lr {r[0]}: {r[1]} trees, test approved default {r[2]:.1%}, AUC {r[3]:.3f}")
n_xgb = int(xgb_lr.loc[xgb_lr.learning_rate == 0.1, "trees_by_early_stopping"].iloc[0])
results[f"XGBoost, lr 0.1, {n_xgb} trees"] = book(y_te, p_xgb, amt_te)
print(f"   (Q2) With the usual 0.5 cut-off XGBoost would decline {(p_xgb >= 0.5).mean():.2%} of 2011 applicants; "
      f"the policy's cut-off is a predicted default of {np.quantile(p_xgb, APPROVE):.1%}")

long = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED).fit(X_tr, y_tr)
xgb_curve = []
for n in [1, 5, 10, 25, 50, 98, 200, 400, 800, 1200, 2000]:
    xgb_curve.append((n, dr(y_tr, long.predict_proba(X_tr, iteration_range=(0, n))[:, 1]),
                      dr(y_te, long.predict_proba(X_te, iteration_range=(0, n))[:, 1])))
xgb_curve = pd.DataFrame(xgb_curve, columns=["rounds", "train_dr", "test_dr"])
xgb_curve.to_csv(TAB / "q6_xgb_rounds_curve.csv", index=False)
print("   Rounds curve (lr 0.1):", ", ".join(f"{int(r.rounds)}: {r.train_dr:.1%}/{r.test_dr:.1%}"
                                             for _, r in xgb_curve.iterrows()))

# One figure, three knobs: tree depth, forest size, boosting rounds.
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=True)
base = results["Approve everybody"]["default_rate_approved"]
for ax, (data, x, title) in zip(axes, [(depth[depth.max_depth < 99], "max_depth", "Single tree: depth"),
                                       (rf_curve, "trees", "Random Forest (depth 10): trees"),
                                       (xgb_curve, "rounds", "XGBoost (lr 0.1): rounds")]):
    ax.plot(data[x], data.train_dr, "o-", ms=3, color=GREY, label="train")
    ax.plot(data[x], data.test_dr, "o-", ms=3, color=BLUE, label="test (2011)")
    ax.axhline(base, color=RED, ls=":", lw=1, label="approve all (test)")
    if x != "max_depth":
        ax.set_xscale("log")
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel("Default rate among approved")
axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "q4_q6_overfitting_knobs.png")
plt.close(fig)

# ---------------------------------------------------------------- Q7: side by side
# Lending Club's own grade as a benchmark: approve the 80% with the best sub-grade.
sub = df.loc[test, "sub_grade"].rank(method="first").to_numpy()
results["Lending Club sub-grade (benchmark)"] = book(y_te, sub, amt_te)
comp = pd.DataFrame(results).T
comp.to_csv(TAB / "q7_model_comparison.csv")
print("Q7.\n" + comp.to_string(float_format=lambda v: f"{v:.3f}"))
allin = results["Approve everybody"]
xg = results[f"XGBoost, lr 0.1, {n_xgb} trees"]
print(f"   XGBoost vs approving all: {xg['declined_good (FP)']} good customers turned away to avoid "
      f"{xg['declined_defaults (TP)']} defaults")

rates = np.linspace(0.5, 1.0, 26)
fig, ax = plt.subplots(figsize=(4.4, 2.8))
for name, p, c in [("Decision tree, depth 6", p_tree, GREY), ("Random Forest", p_rf, GREEN),
                   ("XGBoost", p_xgb, BLUE), ("Lending Club sub-grade", sub, RED)]:
    ax.plot(rates * 100, [dr(y_te, p, r) for r in rates], color=c, label=name, lw=1.3)
ax.axvline(80, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Approval rate (%)")
ax.set_ylabel("Default rate among approved")
ax.set_title("Approving fewer applicants cleans the book, at a falling rate", fontsize=8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "q7_default_vs_approval.png")
plt.close(fig)
curve = pd.DataFrame({"approval_rate": rates, "xgb_dr": [dr(y_te, p_xgb, r) for r in rates]})
curve.to_csv(TAB / "q7_xgb_default_vs_approval.csv", index=False)

# ---------------------------------------------------------------- Q8: SHAP
explainer = shap.TreeExplainer(xgb_final)
sv = explainer.shap_values(X_te)
mean_abs = pd.Series(np.abs(sv).mean(0), index=X_te.columns).sort_values(ascending=False)
mean_abs.rename("mean_abs_shap").to_csv(TAB / "q8_shap_importance.csv")
print("Q8. Mean |SHAP|:", ", ".join(f"{k} {v:.3f}" for k, v in mean_abs.head(5).items()))
worst = int(np.argmax(p_xgb))
contrib = pd.Series(sv[worst], index=X_te.columns).sort_values(key=np.abs, ascending=False)
pd.DataFrame({"value": X_te.iloc[worst][contrib.index], "shap": contrib}).to_csv(TAB / "q8_riskiest_applicant.csv")
print(f"   Riskiest test applicant: predicted default {p_xgb[worst]:.1%}; top reasons:",
      ", ".join(f"{k}={X_te.iloc[worst][k]:g} ({v:+.2f})" for k, v in contrib.head(4).items()))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw={"wspace": 0.9})
top = mean_abs.head(10)[::-1]
axes[0].barh(top.index, top.values, color=BLUE)
axes[0].set_xlabel("Mean |SHAP| (log-odds)")
axes[0].set_title("Drivers of risk, all 2011 applicants", fontsize=8)
c = contrib.head(8)[::-1]
axes[1].barh([f"{k} = {X_te.iloc[worst][k]:g}" for k in c.index], c.values,
             color=[RED if v > 0 else GREEN for v in c.values])
axes[1].set_xlabel("SHAP (log-odds, + = towards default)")
axes[1].set_title(f"Riskiest applicant (p = {p_xgb[worst]:.0%})", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q8_shap.png")
plt.close(fig)

# ---------------------------------------------------------------- Q9: reverse-engineer the interest rate
r_tr, r_te = df.loc[train, "int_rate"].to_numpy(), df.loc[test, "int_rate"].to_numpy()
reg = {"Naive: training mean": np.full(len(r_te), r_tr.mean())}
reg["Random Forest"] = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=SEED,
                                             n_jobs=-1).fit(X_tr, r_tr).predict(X_te)
reg["XGBoost"] = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, random_state=SEED).fit(X_tr, r_tr).predict(X_te)
q9 = pd.DataFrame({k: {"MAE": mean_absolute_error(r_te, v), "RMSE": mean_squared_error(r_te, v) ** 0.5}
                   for k, v in reg.items()}).T
q9.to_csv(TAB / "q9_interest_rate.csv")
print(f"Q9. Interest rate (test mean {r_te.mean():.2f}%, train mean {r_tr.mean():.2f}%):\n"
      + q9.to_string(float_format=lambda v: f"{v:.2f}"))
