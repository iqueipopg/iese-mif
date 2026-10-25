"""Task 1: is there volatility to forecast? Descriptives, ACF of r and r^2, Ljung-Box and ARCH-LM on 2017-2024."""

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

from data import load, save, train


def main() -> None:
    r = train(load())["r"]

    desc = pd.Series({
        "N (days)": len(r),
        "mean (%)": r.mean(),
        "std dev (%)": r.std(),
        "annualised vol (%, x sqrt 365)": r.std() * 365 ** 0.5,
        "skewness": stats.skew(r),
        "excess kurtosis": stats.kurtosis(r),
        "min (%)": r.min(),
        "max (%)": r.max(),
        "Jarque-Bera p-value": stats.jarque_bera(r).pvalue,
        "days with |r| > 4 sd": int((r.abs() > 4 * r.std()).sum()),
        "expected under Normal": len(r) * 2 * stats.norm.sf(4),
    }).to_frame("training 18-08-2017 to 31-12-2024")
    save(desc, "t1_descriptives")

    lags = [1, 2, 5, 10, 20]
    a_r, a_r2 = acf(r, nlags=20), acf(r**2, nlags=20)
    acf_tab = pd.DataFrame({"ACF r": a_r[lags], "ACF r^2": a_r2[lags]}, index=pd.Index(lags, name="lag"))
    acf_tab["95% band"] = 1.96 / len(r) ** 0.5
    save(acf_tab, "t1_acf")

    lb = pd.DataFrame({
        "LB Q (r)": acorr_ljungbox(r, lags=[10, 20])["lb_stat"],
        "p (r)": acorr_ljungbox(r, lags=[10, 20])["lb_pvalue"],
        "LB Q (r^2)": acorr_ljungbox(r**2, lags=[10, 20])["lb_stat"],
        "p (r^2)": acorr_ljungbox(r**2, lags=[10, 20])["lb_pvalue"],
    })
    lb.index.name = "lags"
    save(lb, "t1_ljungbox", "{:.4g}")

    eps = r - r.mean()
    arch_rows = {}
    for q in (1, 5, 10):
        lm, lm_p, _, _ = het_arch(eps, nlags=q)
        arch_rows[q] = {"LM stat": lm, "p-value": lm_p}
    save(pd.DataFrame(arch_rows).T.rename_axis("lags"), "t1_archlm", "{:.4g}")


if __name__ == "__main__":
    main()
