"""Tasks 2-3: ARCH(1) and GARCH(1,1), Normal vs Student-t, estimated on the training sample."""

import numpy as np
import pandas as pd
from arch import arch_model

from data import load, save, train


def fit(r: pd.Series, p: int, q: int, dist: str):
    return arch_model(r, mean="Constant", vol="GARCH", p=p, q=q, dist=dist).fit(disp="off")


def persistence_ci(res) -> tuple[float, float, float]:
    """alpha+beta and its 95% CI using the joint covariance (alpha and beta are correlated)."""
    cov = res.param_cov
    s = res.params["alpha[1]"] + res.params["beta[1]"]
    var = cov.loc["alpha[1]", "alpha[1]"] + cov.loc["beta[1]", "beta[1]"] + 2 * cov.loc["alpha[1]", "beta[1]"]
    return s, s - 1.96 * var**0.5, s + 1.96 * var**0.5


def main() -> None:
    r = train(load())["r"]
    models = {
        "ARCH(1) Normal": fit(r, 1, 0, "normal"),
        "GARCH(1,1) Normal": fit(r, 1, 1, "normal"),
        "GARCH(1,1) Student-t": fit(r, 1, 1, "t"),
    }

    rows = {}
    for name, res in models.items():
        row = {}
        for k in ["mu", "omega", "alpha[1]", "beta[1]", "nu"]:
            if k in res.params:
                row[k] = res.params[k]
                row[f"t({k})"] = res.tvalues[k]
        row |= {"log-lik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic}
        if "beta[1]" in res.params:
            s, lo, hi = persistence_ci(res)
            row |= {"alpha+beta": s, "a+b 95% lo": lo, "a+b 95% hi": hi,
                    "half-life (days)": np.log(0.5) / np.log(s) if s < 1 - 1e-6 else np.inf,
                    "uncond. vol (%/day)": (res.params["omega"] / (1 - s)) ** 0.5 if s < 1 - 1e-6 else np.inf,
                    "corr(alpha,beta)": res.param_cov.loc["alpha[1]", "beta[1]"]
                    / (res.param_cov.loc["alpha[1]", "alpha[1]"] * res.param_cov.loc["beta[1]", "beta[1]"]) ** 0.5}
        else:
            a = res.params["alpha[1]"]
            row |= {"alpha+beta": a, "half-life (days)": np.log(0.5) / np.log(a),
                    "uncond. vol (%/day)": (res.params["omega"] / (1 - a)) ** 0.5}
        rows[name] = row
    save(pd.DataFrame(rows), "t2_t3_models")

    # How hard a 10% day hits tomorrow's variance, starting from a calm day (sigma = 2.5%, the 2023-24 norm).
    paths = {}
    for name in ["GARCH(1,1) Normal", "GARCH(1,1) Student-t"]:
        g = models[name].params
        a, b, w = g["alpha[1]"], g["beta[1]"], g["omega"]
        h = [w + a * (10.0 - g["mu"]) ** 2 + b * 2.5**2]
        for _ in range(59):
            h.append(w + (a + b) * h[-1])  # E[eps^2] = h for horizons beyond one day
        paths[name] = np.sqrt(h)
    imp = pd.DataFrame(paths, index=pd.RangeIndex(1, 61, name="days after the 10% shock"))
    save(imp.loc[[1, 2, 5, 10, 20, 30, 60]], "t2_shock_10pct_vol")

    # Standardised residuals: are the tails still fat once volatility clustering is removed?
    for name in ["GARCH(1,1) Normal", "GARCH(1,1) Student-t"]:
        z = models[name].std_resid.dropna()
        print(f"{name}: std resid excess kurtosis {z.kurt():.2f}, share |z|>4: {(z.abs() > 4).mean():.4%}")


if __name__ == "__main__":
    main()
