"""Task 2: simple predictive regression R(t+1) = a + b * Rank(R(t)) on the training sample."""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "panel_features.parquet"
RANKS = ["r_ret", "r_mom", "r_size", "r_vol", "r_turnover"]


def load_train() -> pd.DataFrame:
    df = pd.read_parquet(FEAT)
    # Same complete-case sample that M1-M3 will use, so Task 2's beta equals M1's.
    usable = df.dropna(subset=["ret_next", *RANKS])
    return usable[usable["sample"] == "train"]


def main() -> None:
    train = load_train()
    X = sm.add_constant(train[["r_ret"]])
    fit = sm.OLS(train["ret_next"], X).fit(cov_type="HC1")

    a, b = fit.params["const"], fit.params["r_ret"]
    print(f"N = {int(fit.nobs):,}")
    print(f"alpha = {a:.5f}")
    print(f"beta  = {b:.5f}   SE (HC1) = {fit.bse['r_ret']:.5f}   t = {fit.tvalues['r_ret']:.2f}   "
          f"p = {fit.pvalues['r_ret']:.2e}")
    print(f"R2 in-sample = {fit.rsquared:.5f}  ({fit.rsquared:.3%})")
    print(f"95% CI beta = [{fit.conf_int().loc['r_ret', 0]:.5f}, {fit.conf_int().loc['r_ret', 1]:.5f}]")
    print(f"\npredicted R at p10 = {a + 0.1 * b:.4%}   at p90 = {a + 0.9 * b:.4%}")
    print(f"p90 - p10 = 0.8 * beta = {0.8 * b:.4%} per month")
    print(f"\nmean next-month return (train) = {train['ret_next'].mean():.4%}")
    print(f"SD next-month return (train)   = {train['ret_next'].std():.4%}")


if __name__ == "__main__":
    main()
