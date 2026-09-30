"""Task 3: pooled OLS M1-M3 on the same complete-case training observations."""

import pandas as pd
import statsmodels.api as sm

from task2 import load_train

MODELS = {
    "M1": ["r_ret"],
    "M2": ["r_ret", "r_mom", "r_size"],
    "M3": ["r_ret", "r_mom", "r_size", "r_vol", "r_turnover"],
}


def fit_models(train: pd.DataFrame) -> dict:
    return {
        name: sm.OLS(train["ret_next"], sm.add_constant(train[cols])).fit(cov_type="HC1")
        for name, cols in MODELS.items()
    }


def main() -> None:
    train = load_train()
    fits = fit_models(train)

    rows = []
    for name, fit in fits.items():
        row = {"model": name, "predictors": len(MODELS[name]), "N": int(fit.nobs),
               "R2 (%)": 100 * fit.rsquared, "adj R2 (%)": 100 * fit.rsquared_adj}
        for col in MODELS["M3"]:
            if col in fit.params:
                row[col] = f"{fit.params[col]:+.4f} ({fit.tvalues[col]:.1f})"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model").fillna("")
    pd.set_option("display.width", 200)
    print("coefficient (t-stat, HC1)\n")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\np90 - p10 predicted difference in M3 (0.8 * beta, pp per month):")
    for col in MODELS["M3"]:
        print(f"  {col:<11} {80 * fits['M3'].params[col]:+.2f}")


if __name__ == "__main__":
    main()
