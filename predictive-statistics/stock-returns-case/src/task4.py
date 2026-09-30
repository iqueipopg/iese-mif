"""Task 4: freeze M1-M3 at end-2019 and forecast 2020-2025 against the historical-mean benchmark."""

import pandas as pd
import statsmodels.api as sm

from task2 import FEAT, RANKS
from task3 import MODELS, fit_models


def load_usable() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEAT).dropna(subset=["ret_next", *RANKS])
    return df[df["sample"] == "train"], df[df["sample"] == "test"]


def main() -> None:
    train, test = load_usable()
    fits = fit_models(train)
    y = test["ret_next"]

    # Benchmark: intercept-only OLS on train = historical mean return.
    bench = train["ret_next"].mean()
    preds = {"Benchmark": pd.Series(bench, index=test.index)}
    for name, fit in fits.items():
        preds[name] = fit.predict(sm.add_constant(test[MODELS[name]], has_constant="add"))

    mse_bench = ((y - preds["Benchmark"]) ** 2).mean()
    rows = []
    for name, p in preds.items():
        mse = ((y - p) ** 2).mean()
        rows.append({
            "model": name,
            "predictors": 0 if name == "Benchmark" else len(MODELS[name]),
            "R2 in-sample (%)": 0.0 if name == "Benchmark" else 100 * fits[name].rsquared,
            "MSE test": mse,
            "R2 OOS (%)": 100 * (1 - mse / mse_bench),
        })
    table = pd.DataFrame(rows).set_index("model")
    print(f"train N = {len(train):,}   test N = {len(test):,}   benchmark forecast = {bench:.4%}\n")
    print(table.to_string(float_format=lambda v: f"{v:.6f}"))

    # Stability: OOS R2 by calendar year of the target month.
    year = (test["target_m"] - 1) // 12
    by_year = {}
    for name in MODELS:
        se_m = (y - preds[name]) ** 2
        se_b = (y - preds["Benchmark"]) ** 2
        by_year[name] = 100 * (1 - se_m.groupby(year).sum() / se_b.groupby(year).sum())
    print("\nR2 OOS (%) by target year:")
    print(pd.DataFrame(by_year).to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
