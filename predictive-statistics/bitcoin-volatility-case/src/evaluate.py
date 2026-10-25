"""Tasks 4-6: score the forecasts against realized variance, RV-OLS in sample, rolling parameter stability."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import FORECASTS, HAC_LAGS, PARAMS, TRAIN_END, load, save
from forecast import rv_ols, rv_pairs

RULES = ["GARCH-t expanding", "RV-OLS expanding", "GARCH-t rolling 2y", "RV-OLS rolling 2y", "RV-OLS frozen"]


def scores(rv: pd.Series, h: pd.Series) -> dict:
    mz = sm.OLS(rv, sm.add_constant(h)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {
        "MSE": ((rv - h) ** 2).mean(),
        "QLIKE": (np.log(h) + rv / h).mean(),
        "Corr": np.corrcoef(h, rv)[0, 1],
        "mean forecast": h.mean(),
        "mean RV": rv.mean(),
        "forecast / RV": h.mean() / rv.mean(),
        "MZ slope": mz.params.iloc[1],  # RV = a + b*h: b = 1 if the forecast is well scaled
        "MZ R2": mz.rsquared,
        "days h < RV": (h < rv).mean(),
    }


def dm_test(rv: pd.Series, h1: pd.Series, h2: pd.Series, loss: str) -> tuple[float, float]:
    """Diebold-Mariano: is the loss difference (model 1 minus model 2) zero on average? HAC t-stat."""
    if loss == "MSE":
        d = (rv - h1) ** 2 - (rv - h2) ** 2
    else:
        d = (np.log(h1) + rv / h1) - (np.log(h2) + rv / h2)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return res.tvalues.iloc[0], res.pvalues.iloc[0]


def main() -> None:
    df = load()
    fc = pd.read_parquet(FORECASTS)
    ev = fc[fc["RV_VALID"]]
    print(f"OOS days {len(fc)}, with valid RV {len(ev)}; invalid RV at origin: {int((~df.loc[fc.index - pd.Timedelta('1D'), 'RV_VALID'].astype(bool)).sum())}")

    # Task 5 in sample: RV_{t+1} = a + b RV_t on training pairs.
    tr = rv_pairs(df).loc[:TRAIN_END]
    ols = rv_ols(tr)
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
    insample = pd.Series({"a": hac.params[0], "t(a) HAC": hac.tvalues[0], "b": hac.params[1],
                          "t(b) HAC": hac.tvalues[1], "t(b) OLS": ols.tvalues.iloc[1],
                          "b 95% lo": hac.conf_int()[1, 0], "b 95% hi": hac.conf_int()[1, 1],
                          "R2": ols.rsquared, "pairs": ols.nobs,
                          "half-life (days)": np.log(0.5) / np.log(hac.params[1])}).to_frame("training")
    save(insample, "t5_rvols_insample")

    tab = pd.DataFrame({k: scores(ev["RV"], ev[k]) for k in RULES}).T
    save(tab, "t4_t6_oos_scores")

    # Same scores by half-year, to see whether the ranking is stable over the OOS period.
    half = ev.index.year.astype(str) + np.where(ev.index.month <= 6, "H1", "H2")
    byh = {k: ev.groupby(half).apply(lambda g, k=k: ((g["RV"] - g[k]) ** 2).mean()) for k in RULES[:4]}
    save(pd.DataFrame(byh), "t4_mse_by_halfyear", "{:.2f}")
    byq = {k: ev.groupby(half).apply(lambda g, k=k: (np.log(g[k]) + g["RV"] / g[k]).mean()) for k in RULES[:4]}
    save(pd.DataFrame(byq), "t4_qlike_by_halfyear", "{:.3f}")

    dm = {}
    for a, b in [("GARCH-t expanding", "RV-OLS expanding"), ("GARCH-t rolling 2y", "RV-OLS rolling 2y"),
                 ("GARCH-t rolling 2y", "GARCH-t expanding"), ("RV-OLS rolling 2y", "RV-OLS expanding")]:
        for loss in ("MSE", "QLIKE"):
            t, p = dm_test(ev["RV"], ev[a], ev[b], loss)
            dm[(f"{a} vs {b}", loss)] = {"DM t (neg = first better)": t, "p-value": p}
    save(pd.DataFrame(dm).T, "t4_t6_diebold_mariano", "{:.3f}")

    # Dynamics: how much each forecast moves day to day, and how fast it reacts to RV jumps.
    dyn = {}
    for k in RULES[:4]:
        dyn[k] = {"sd of forecast": fc[k].std(), "mean |daily change|": fc[k].diff().abs().mean(),
                  "corr with RV_t (origin)": fc[k].corr(fc["RV_origin"]),
                  "corr with r_t^2 (origin)": fc[k].corr(fc["r"].shift(1) ** 2)}
    save(pd.DataFrame(dyn).T, "t5_forecast_dynamics")

    # Task 6: rolling parameter stability.
    par = pd.read_parquet(PARAMS)
    stab = {}
    for lab in ("expanding", "rolling 2y"):
        stab[lab] = {
            "b min": par[f"b {lab}"].min(), "b max": par[f"b {lab}"].max(),
            "share b CI > 0": (par[f"b lo {lab}"] > 0).mean(),
            "a+b min": par[f"a+b {lab}"].min(), "a+b max": par[f"a+b {lab}"].max(),
            "share a+b > 0.99": (par[f"a+b {lab}"] > 0.99).mean(),
            "share a+b CI contains 1": ((par[f"a+b lo {lab}"] <= 1) & (par[f"a+b hi {lab}"] >= 1)).mean(),
            "alpha range": f"{par[f'alpha {lab}'].min():.3f}-{par[f'alpha {lab}'].max():.3f}",
            "nu range": f"{par[f'nu {lab}'].min():.2f}-{par[f'nu {lab}'].max():.2f}",
        }
    stab_df = pd.DataFrame(stab)
    stab_df.to_csv("../output/tables/t6_param_stability.csv")
    print("\n== t6_param_stability ==\n" + stab_df.to_string())
    print("\nfirst and last rolling origin:\n" + par[[c for c in par if "rolling" in c and " lo " not in c and " hi " not in c]].iloc[[0, -1]].T.to_string())

    # Volatility regimes: average realized vol by year, the context for "does distant history help?"
    yearly = pd.DataFrame({"mean RV": df["RV"].groupby(df.index.year).mean(),
                           "mean r^2": (df["r"] ** 2).groupby(df.index.year).mean(),
                           "sd of daily r": df["r"].groupby(df.index.year).std()})
    yearly["RV / r^2"] = yearly["mean RV"] / yearly["mean r^2"]
    save(yearly, "t6_vol_by_year", "{:.2f}")


if __name__ == "__main__":
    main()
