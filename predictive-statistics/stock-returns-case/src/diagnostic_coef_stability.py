"""Ex-post diagnostic (not used for any modelling choice): did the M3 relationships change in 2020-2025?

Re-estimates M3 separately on the training and test samples and compares coefficients.
"""

import pandas as pd
import statsmodels.api as sm

from task3 import MODELS
from task4 import load_usable


def main() -> None:
    train, test = load_usable()
    cols = MODELS["M3"]
    rows = {}
    for name, d in [("train 2001-2019", train), ("test 2020-2025", test)]:
        fit = sm.OLS(d["ret_next"], sm.add_constant(d[cols])).fit(cov_type="HC1")
        rows[name] = {c: f"{fit.params[c]:+.4f} ({fit.tvalues[c]:+.1f})" for c in cols}
        rows[name]["R2 (%)"] = f"{100 * fit.rsquared:.3f}"
        rows[name]["mean ret_next (%)"] = f"{100 * d['ret_next'].mean():.2f}"
        rows[name]["SD ret_next (%)"] = f"{100 * d['ret_next'].std():.2f}"
    print("M3 coefficient (t-stat, HC1), estimated separately in each period\n")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
