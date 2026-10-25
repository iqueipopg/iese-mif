"""All figures in output/figures/."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from data import FIGS, FORECASTS, OOS_START, PARAMS, load, train

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"RV": "#b0b0b0", "GARCH": "#c0392b", "OLS": "#2c5d8f", "dark": "#222222"}


def f1_price_returns(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax[0].semilogy(df.index, df["Price"], color=C["dark"], lw=0.8)
    ax[0].set_ylabel("Price (USD, log)")
    ax[1].plot(df.index, df["r"], color=C["dark"], lw=0.4)
    ax[1].set_ylabel("Daily return (%)")
    for a in ax:
        a.axvline(OOS_START, color=C["GARCH"], ls="--", lw=1)
    ax[0].text(OOS_START, ax[0].get_ylim()[1] * 0.6, "  out of sample", color=C["GARCH"], fontsize=8)
    fig.savefig(FIGS / "f1_price_returns.png")


def f2_acf(df: pd.DataFrame) -> None:
    r = train(df)["r"]
    lags = np.arange(1, 31)
    band = 1.96 / len(r) ** 0.5
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for a, x, title in [(ax[0], r, "Returns $r_t$"), (ax[1], r**2, "Squared returns $r_t^2$")]:
        a.bar(lags, acf(x, nlags=30)[1:], color=C["OLS"], width=0.7)
        a.axhspan(-band, band, color=C["RV"], alpha=0.35, lw=0)
        a.set_title(title, fontsize=9)
        a.set_xlabel("lag (days)")
    ax[0].set_ylabel("autocorrelation")
    fig.savefig(FIGS / "f2_acf.png")


def f3_forecasts(fc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for a, lab in [(ax[0], "expanding"), (ax[1], "rolling 2y")]:
        a.plot(fc.index, fc["RV"], color=C["RV"], lw=0.7, label="realized variance $RV_{t+1}$")
        a.plot(fc.index, fc[f"RV-OLS {lab}"], color=C["OLS"], lw=0.8, label="RV-OLS forecast")
        a.plot(fc.index, fc[f"GARCH-t {lab}"], color=C["GARCH"], lw=1.1, label="GARCH-t forecast")
        a.set_ylim(0, 60)
        a.set_title(f"{lab.capitalize()} estimation window", fontsize=9, loc="left")
        a.set_ylabel("daily variance (%$^2$)")
    ax[0].legend(loc="upper right", fontsize=7.5, frameon=False, ncol=3)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.savefig(FIGS / "f3_forecasts.png")


def f4_zoom(fc: pd.DataFrame) -> None:
    """Two episodes: a high-RV day with a flat close (Jan 2025) and a large return shock (Feb 2026)."""
    eps = [("2025-01-12", "2025-02-08", "20 Jan 2025: RV = 60, daily return +0.9%"),
           ("2026-01-29", "2026-02-28", "5-6 Feb 2026: returns -15.2% and +11.8%")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for a, (s0, s1, title) in zip(ax, eps):
        w = fc.loc[s0:s1]
        a.bar(w.index, w["RV"], color=C["RV"], width=0.8, label="realized variance")
        a.plot(w.index, w["RV-OLS rolling 2y"], color=C["OLS"], marker="o", ms=2, lw=1, label="RV-OLS")
        a.plot(w.index, w["GARCH-t rolling 2y"], color=C["GARCH"], marker="o", ms=2, lw=1.2, label="GARCH-t")
        a.set_title(title, fontsize=8, loc="left")
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        a.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        a.tick_params(axis="x", labelsize=7)
    ax[0].set_ylabel("daily variance (%$^2$)")
    ax[1].legend(fontsize=7, frameon=False)
    fig.savefig(FIGS / "f4_zoom_spike.png")


def f5_rolling_params(par: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    lab = "rolling 2y"
    ax[0].fill_between(par.index, par[f"b lo {lab}"], par[f"b hi {lab}"], color=C["OLS"], alpha=0.2, lw=0,
                       label="95% HAC band")
    ax[0].plot(par.index, par[f"b {lab}"], color=C["OLS"], lw=1, label="rolling $b$")
    ax[0].axhline(par["b expanding"].iloc[0], color=C["dark"], ls=":", lw=1, label="expanding $b$ (end-2024)")
    ax[0].axhline(0, color=C["dark"], lw=0.6)
    ax[0].set_title("RV-OLS persistence $b$", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, frameon=False, loc="lower right")
    ax[1].plot(par.index, par[f"a+b {lab}"], color=C["GARCH"], lw=1.2, label=r"$\alpha+\beta$")
    ax[1].plot(par.index, par[f"beta {lab}"], color=C["dark"], lw=0.8, ls="--", label=r"$\beta$")
    ax2 = ax[1].twinx()
    ax2.plot(par.index, par[f"alpha {lab}"], color=C["OLS"], lw=0.8, label=r"$\alpha$ (right axis)")
    ax2.set_ylim(0, 0.12)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax[1].set_title("Rolling GARCH-t parameters", fontsize=9, loc="left")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower left")
    for a in ax:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        a.tick_params(axis="x", labelsize=7)
    fig.savefig(FIGS / "f5_rolling_params.png")


def f6_regimes(df: pd.DataFrame) -> None:
    y = df.groupby(df.index.year).agg(rv=("RV", "mean"), r2=("r", lambda x: (x**2).mean()))
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    x = np.arange(len(y))
    ax.bar(x - 0.2, y["rv"], 0.4, color=C["OLS"], label="mean realized variance")
    ax.bar(x + 0.2, y["r2"], 0.4, color=C["RV"], label="mean squared daily return")
    ax.axvspan(len(y) - 2.5, len(y) - 0.5, color=C["GARCH"], alpha=0.08, lw=0)
    ax.set_xticks(x, y.index)
    ax.set_ylabel("%$^2$ per day")
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(FIGS / "f6_regimes.png")


def f7_cum_loss(fc: pd.DataFrame) -> None:
    ql = {k: np.log(fc[k]) + fc["RV"] / fc[k] for k in ["GARCH-t expanding", "RV-OLS expanding",
                                                        "GARCH-t rolling 2y", "RV-OLS rolling 2y"]}
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(fc.index, (ql["RV-OLS expanding"] - ql["GARCH-t expanding"]).cumsum(), color=C["dark"],
            label="expanding: RV-OLS minus GARCH-t")
    ax.plot(fc.index, (ql["RV-OLS rolling 2y"] - ql["GARCH-t rolling 2y"]).cumsum(), color=C["OLS"],
            label="rolling 2y: RV-OLS minus GARCH-t")
    ax.axhline(0, color=C["dark"], lw=0.6)
    ax.set_ylabel("cumulative QLIKE difference")
    ax.legend(fontsize=7.5, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.savefig(FIGS / "f7_cum_qlike.png")


def main() -> None:
    df = load()
    fc = pd.read_parquet(FORECASTS)
    par = pd.read_parquet(PARAMS)
    f1_price_returns(df)
    f2_acf(df)
    f3_forecasts(fc)
    f4_zoom(fc)
    f5_rolling_params(par)
    f6_regimes(df)
    f7_cum_loss(fc)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
