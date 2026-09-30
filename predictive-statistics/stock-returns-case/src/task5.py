"""Task 5: M3 with stock fixed effects (within estimator), in-sample fit and OOS forecasts."""

import pandas as pd
import statsmodels.api as sm

from task3 import MODELS
from task4 import load_usable

COLS = MODELS["M3"]


def r2(y: pd.Series, pred: pd.Series) -> float:
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> None:
    train, test = load_usable()
    y = train["ret_next"]
    g = train.groupby("permno")

    # Within transformation: remove each stock's training mean from y and X.
    y_dm = y - g["ret_next"].transform("mean")
    X_dm = train[COLS] - g[COLS].transform("mean")
    fe = sm.OLS(y_dm, X_dm).fit(cov_type="HC1")
    beta = fe.params

    # Recover alpha_i = mean(y_i) - beta' mean(X_i).
    means = g[["ret_next", *COLS]].mean()
    alpha = means["ret_next"] - means[COLS] @ beta

    fitted = train["permno"].map(alpha) + train[COLS] @ beta
    pooled = sm.OLS(y, sm.add_constant(train[COLS])).fit(cov_type="HC1")

    print(f"train: {len(train):,} obs, {len(alpha):,} stock intercepts\n")
    print("M3 coefficient (t-stat, HC1)   pooled OLS        stock FE")
    for c in COLS:
        print(f"  {c:<11} {pooled.params[c]:+.4f} ({pooled.tvalues[c]:+6.1f})   "
              f"{beta[c]:+.4f} ({fe.tvalues[c]:+6.1f})")
    print(f"\nR2 pooled OLS (overall)     = {100 * pooled.rsquared:.3f}%")
    print(f"R2 stock FE, overall        = {100 * r2(y, fitted):.3f}%")
    print(f"R2 stock FE, within         = {100 * r2(y_dm, X_dm @ beta):.3f}%")

    # Out of sample: only stocks already seen in training have an intercept.
    seen = test["permno"].isin(alpha.index)
    t = test[seen]
    yt = t["ret_next"]
    bench = train["ret_next"].mean()
    pred_pooled = pooled.predict(sm.add_constant(t[COLS], has_constant="add"))
    pred_fe = t["permno"].map(alpha) + t[COLS] @ beta
    mse_b = ((yt - bench) ** 2).mean()
    print(f"\ntest obs: {len(test):,}; from stocks seen in training: {seen.sum():,} ({seen.mean():.1%}); "
          f"new stocks (no intercept): {(~seen).sum():,}")
    print(f"test stocks: {test['permno'].nunique():,}; seen in training: {t['permno'].nunique():,}")
    for name, p in [("benchmark", pd.Series(bench, index=t.index)), ("pooled M3", pred_pooled), ("stock FE M3", pred_fe)]:
        mse = ((yt - p) ** 2).mean()
        print(f"  {name:<12} MSE = {mse:.6f}   R2 OOS = {100 * (1 - mse / mse_b):+.3f}%")

    # Are historical average returns persistent across stocks?
    test_mean = t.groupby("permno")["ret_next"].mean()
    both = pd.concat([means["ret_next"].rename("train_mean"), test_mean.rename("test_mean")], axis=1).dropna()
    print(f"\ncorrelation across {len(both):,} stocks of mean return 2001-19 vs 2020-25: "
          f"{both.corr().iloc[0, 1]:+.3f}")
    print(f"dispersion of alpha_i: SD = {100 * alpha.std():.2f} pp/month; "
          f"training obs per stock: median {int(g.size().median())}, "
          f"{(g.size() < 24).mean():.0%} of stocks have < 24")


if __name__ == "__main__":
    main()
