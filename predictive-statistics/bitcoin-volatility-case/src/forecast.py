"""One-day-ahead variance forecasts for every day of 2025-01-01 to 2026-08-31.

Four forecasting rules, each re-estimated at every origin t with information up to t only:
  GARCH-t expanding, GARCH-t rolling 2y, RV-OLS expanding, RV-OLS rolling 2y.
Plus RV-OLS frozen at end-2024 (the literal "training information" reading of Task 5).
Writes output/forecasts.parquet (one row per target day) and output/rolling_params.parquet (one row per origin).
"""

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from data import FORECASTS, HAC_LAGS, OOS_END, OOS_START, PARAMS, TRAIN_END, load


def garch_t(r: pd.Series, start: np.ndarray | None):
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
        disp="off", starting_values=start, options={"maxiter": 500})
    if res.convergence_flag != 0:  # fall back to arch's own starting values
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    return res


def rv_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """(RV_s, RV_{s+1}) pairs, kept only when both days have a valid 24-hour RV."""
    p = pd.DataFrame({"x": df["RV"].shift(1), "y": df["RV"],
                      "ok": df["RV_VALID"] & df["RV_VALID"].shift(1, fill_value=False)})
    return p[p["ok"]].dropna()


def rv_ols(pairs: pd.DataFrame):
    return sm.OLS(pairs["y"], sm.add_constant(pairs["x"])).fit()


def main() -> None:
    df = load()
    pairs = rv_pairs(df)
    targets = df.loc[OOS_START:OOS_END].index
    frozen = rv_ols(pairs.loc[:TRAIN_END])

    fc, par = [], []
    start = {"exp": None, "roll": None}
    t0 = time.time()
    for i, day in enumerate(targets):
        origin = day - pd.Timedelta("1D")  # information set ends at t
        win_start = origin - pd.DateOffset(years=2) + pd.Timedelta("1D")  # 1 Jan 2023 for the first origin
        rv_t = df.at[origin, "RV"]
        row = {"date": day, "RV": df.at[day, "RV"], "RV_VALID": df.at[day, "RV_VALID"], "r": df.at[day, "r"],
               "RV_origin": rv_t, "RV-OLS frozen": frozen.params["const"] + frozen.params["x"] * rv_t}
        prow = {"origin": origin}
        for key, sample in {"exp": df.loc[:origin], "roll": df.loc[win_start:origin]}.items():
            label = "expanding" if key == "exp" else "rolling 2y"
            g = garch_t(sample["r"], start[key])
            start[key] = g.params.values
            row[f"GARCH-t {label}"] = g.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]

            ols = rv_ols(pairs.loc[sample.index[1]:origin])
            row[f"RV-OLS {label}"] = ols.params["const"] + ols.params["x"] * rv_t

            hac = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
            b, se = hac.params[1], hac.bse[1]
            cov = g.param_cov
            s = g.params["alpha[1]"] + g.params["beta[1]"]
            s_se = (cov.loc["alpha[1]", "alpha[1]"] + cov.loc["beta[1]", "beta[1]"]
                    + 2 * cov.loc["alpha[1]", "beta[1]"]) ** 0.5
            prow |= {f"b {label}": b, f"b lo {label}": b - 1.96 * se, f"b hi {label}": b + 1.96 * se,
                     f"R2 {label}": ols.rsquared, f"n pairs {label}": int(ols.nobs),
                     f"alpha {label}": g.params["alpha[1]"], f"beta {label}": g.params["beta[1]"],
                     f"a+b {label}": s, f"a+b lo {label}": s - 1.96 * s_se, f"a+b hi {label}": s + 1.96 * s_se,
                     f"nu {label}": g.params["nu"], f"n {label}": len(sample)}
        fc.append(row)
        par.append(prow)
        if i % 50 == 0:
            print(f"{day.date()}  {i + 1}/{len(targets)}  {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(fc).set_index("date").to_parquet(FORECASTS)
    pd.DataFrame(par).set_index("origin").to_parquet(PARAMS)
    print(f"done: {len(fc)} forecast days in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
