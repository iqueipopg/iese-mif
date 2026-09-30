"""All report figures (English). Writes PDFs to output/figures/. Run after src/features.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from task3 import MODELS, fit_models
from task4 import load_usable

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "panel_features.parquet"
RAW = ROOT / "data" / "US_Stocks_PanelDataset.parquet"
OUT = ROOT / "output" / "figures"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#d9d8d4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
VIOLET = "#4a3aa7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8.5,
})

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LABELS = {"r_ret": "Current return", "r_mom": "Momentum 12-2", "r_size": "Size",
          "r_vol": "Volatility", "r_turnover": "Turnover"}


def mlabel(m: int) -> str:
    return f"{MONTHS[(m - 1) % 12]}\n{(m - 1) // 12}"


def yfrac(m):
    return (np.asarray(m) - 1) / 12


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", dpi=200)
    fig.savefig(OUT / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"output/figures/{name}")


# --------------------------------------------------------------------------- data construction


def fig_panel(df: pd.DataFrame) -> None:
    first = df.groupby("permno")["m"].min()
    last = df.groupby("permno")["m"].max()
    order = pd.DataFrame({"f": first, "l": last}).sort_values(["f", "l"]).index
    m0, m1 = df["m"].min(), df["m"].max()
    grid = np.zeros((len(order), m1 - m0 + 1))
    pos = pd.Series(np.arange(len(order)), index=order)
    grid[pos[df["permno"]].to_numpy(), (df["m"] - m0).to_numpy()] = 1
    gaps = df.sort_values(["permno", "m"]).groupby("permno")["m"].diff().gt(1)
    rng = np.random.default_rng(3)
    zoom = np.sort(pos[rng.choice(df.loc[gaps, "permno"].unique(), 40, replace=False)].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    cmap = plt.matplotlib.colors.ListedColormap([SURFACE, BLUE])
    x0, x1 = yfrac(m0), yfrac(m1 + 1)
    axes[0].imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest", extent=[x0, x1, len(order), 0],
                   rasterized=True)
    axes[0].set_title(f"(a) Full panel: {len(order):,} stocks x 312 months")
    axes[0].set_ylabel("stocks (sorted by entry date)")
    axes[1].imshow(grid[zoom], aspect="auto", cmap=cmap, interpolation="nearest", extent=[x0, x1, len(zoom), 0],
                   rasterized=True)
    axes[1].set_title("(b) Zoom: 40 random stocks with gaps")
    axes[1].set_yticks([])
    for ax in axes:
        ax.axvline(2020, color=ORANGE, lw=1.3)
        ax.text(2020.3, 0.98, "train | test", transform=ax.get_xaxis_transform(), color=ORANGE, fontsize=8, va="top")
    fig.legend(handles=[Patch(color=BLUE, label="stock in the sample that month"),
                        Patch(facecolor=SURFACE, edgecolor=MUTED, label="absent")],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    save(fig, "a1_panel.pdf")


def fig_gap(df: pd.DataFrame) -> None:
    d = df.sort_values(["permno", "m"]).copy()
    d["gap"] = d.groupby("permno")["m"].diff()
    cand = d[(d["gap"] >= 4) & (d["gap"] <= 6) & (d["m"] > 2005 * 12)]
    for _, row in cand.iterrows():
        s = d[d["permno"] == row["permno"]].set_index("m")
        if (s["gap"] > 1).sum() == 1 and len(s) > 40:
            break
    after = int(row["m"])
    before = after - int(row["gap"])
    months = np.arange(before - 5, after + 5)
    present = [m for m in months if m in s.index]

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, title, color in [(axes[0], "(a) Shifting by rows: WRONG", RED),
                             (axes[1], "(b) Shifting by calendar month: CORRECT", AQUA)]:
        ax.set_title(title, color=color)
        ax.set_ylim(-1.25, 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for m in months:
            on = m in s.index
            ax.scatter(m, 0, s=170, color=BLUE if on else SURFACE, edgecolor=BLUE if on else MUTED, lw=1.3, zorder=3)
            ax.text(m, 0.3, f"{s.loc[m, 'ret']:+.1%}" if on else "missing", ha="center", fontsize=7.5, color=INK2)
    for i, m in enumerate(present[:-1]):
        nxt = present[i + 1]
        bad = nxt != m + 1
        axes[0].add_patch(FancyArrowPatch((m, -0.15), (nxt, -0.15), connectionstyle="arc3,rad=0.4", arrowstyle="-|>",
                                          mutation_scale=10, color=RED if bad else INK2, lw=1.8 if bad else 0.8))
        if bad:
            axes[0].text((m + nxt) / 2, -1.15, f"uses the {mlabel(nxt).replace(chr(10), ' ')} return "
                         f"({nxt - m} months later) as if it were next month's", ha="center", color=RED, fontsize=8.5)
        ok = m + 1 in s.index
        axes[1].add_patch(FancyArrowPatch((m, -0.15), (m + 1, -0.15), connectionstyle="arc3,rad=0.4", arrowstyle="-|>",
                                          mutation_scale=10, color=INK2 if ok else AQUA, lw=0.8 if ok else 1.8))
        if not ok:
            axes[1].text(m + 1.3, -1.15, "next month does not exist: no target, row is dropped",
                         ha="left", color=AQUA, fontsize=8.5)
    axes[1].set_xticks(months, [mlabel(m) for m in months], fontsize=7)
    save(fig, "a2_gap.pdf")


def fig_windows() -> None:
    offsets = np.arange(-13, 2)
    rows = [("Target R(t+1)", lambda k: k == 1, AQUA),
            ("Volatility (t-11 ... t, min. 8)", lambda k: -11 <= k <= 0, VIOLET),
            ("Current return R(t)", lambda k: k == 0, ORANGE),
            ("Momentum 12-2 (t-12 ... t-2)", lambda k: -12 <= k <= -2, BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), sharey=True)
    for ax, missing, title in [(axes[0], None, "(a) Complete history"), (axes[1], -5, "(b) Month t-5 missing")]:
        for y, (_, used, color) in enumerate(rows):
            for k in offsets:
                if k == missing:
                    ax.add_patch(Rectangle((k - 0.45, y - 0.38), 0.9, 0.76, facecolor=SURFACE, edgecolor=RED,
                                           hatch="///", lw=0.8))
                    continue
                on = used(k)
                ax.add_patch(Rectangle((k - 0.45, y - 0.38), 0.9, 0.76, facecolor=color if on else SURFACE,
                                       edgecolor=SURFACE if on else MUTED, lw=0.8))
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.text(0.35, 3.62, "today (end of t)", ha="right", fontsize=7.5, color=INK2)
        ax.set_xlim(-13.6, 1.6)
        ax.set_ylim(-0.6, 3.95)
        ax.set_xticks(offsets, ["t" if k == 0 else f"t{k:+d}" for k in offsets], fontsize=7)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title)
    for y, txt, c in [(0, "unchanged", INK2), (1, "11 of 12 >= 8: computed", VIOLET),
                      (2, "unchanged", INK2), (3, "10 of 11: missing", RED)]:
        axes[1].text(1.9, y, txt, va="center", fontsize=7.5, color=c)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "a3_windows.pdf")


def fig_ranks(df: pd.DataFrame) -> None:
    x = df.loc[df["yyyymm"] == 201912, ["ticker", "size", "r_size"]].copy()
    x["bn"] = x["size"] / 1e6
    x = x.sort_values("bn").reset_index(drop=True)
    picks = [(x.iloc[-1], ORANGE), (x.iloc[len(x) // 2], AQUA), (x.iloc[len(x) // 20], VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    panels = [(lambda v: v["bn"], "market cap ($ bn)", "(a) Raw: almost all near zero"),
              (lambda v: np.log(v["bn"]), "log(market cap)", "(b) Log: easier to read"),
              (lambda v: v["r_size"], "monthly percentile rank", "(c) Rank: uniform on [0, 1]")]
    for ax, (f, xl, title) in zip(axes, panels):
        ax.hist(f(x), bins=60, color=BLUE, edgecolor=SURFACE, lw=0.4)
        ax.set_title(title, pad=24)
        ax.set_xlabel(xl)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for p, c in picks:
            ax.axvline(f(p), color=c, lw=1.6)
            if ax is axes[0] and c is not ORANGE:
                continue
            fmt = f"{f(p):.2f}" if ax is axes[2] else f"{f(p):,.1f}"
            ax.text(f(p), 1.01, f"{p['ticker']}\n{fmt}", color=c, ha="center", va="bottom", fontsize=7,
                    transform=ax.get_xaxis_transform())
    fig.subplots_adjust(wspace=0.1)
    save(fig, "a4_ranks.pdf")


def fig_split(df: pd.DataFrame, usable: pd.DataFrame) -> None:
    counts = usable.groupby(["m", "sample"]).size().unstack(fill_value=0)
    allm = df.groupby("m").size()
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.bar(yfrac(allm.index), allm.values, width=1 / 12, color=MUTED, label="raw rows")
    ax.bar(yfrac(counts.index), counts["train"], width=1 / 12, color=BLUE, label="usable: training (estimation)")
    ax.bar(yfrac(counts.index), counts["test"], width=1 / 12, color=ORANGE, label="usable: test (evaluation)")
    cut = yfrac(2019 * 12 + 12)
    ax.axvline(cut, color=INK, lw=0.8)
    ax.annotate("forecast made in Dec 2019 for Jan 2020\ngoes to test: on 31 Dec 2019 the\nJanuary return is not yet known",
                xy=(cut, 2500), xytext=(2013.0, 900), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    top = allm.max()
    ax.text(2000.05, top * 1.02, "2000: no 12-month history\nfor momentum yet", fontsize=7.5, color=INK2, va="bottom")
    ax.text(2025.95, top * 1.02, "Dec 2025:\nno next month", fontsize=7.5, color=INK2, va="bottom", ha="right")
    ax.set_ylim(0, top * 1.3)
    ax.set_xlim(2000, 2026)
    ax.set_ylabel("stock-months per month")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    save(fig, "f1_split.pdf")


def fig_funnel(df: pd.DataFrame) -> None:
    names, n = ["raw rows"], [len(df)]
    mask = pd.Series(True, index=df.index)
    for col, name in [("ret_next", "with a consecutive next month"), ("r_mom", "with full 11-month momentum"),
                      ("r_vol", "with volatility (>= 8 months)")]:
        mask &= df[col].notna()
        names.append(name)
        n.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(8, 2.2))
    y = np.arange(len(n))[::-1]
    ax.barh(y, n, color=[MUTED] + [BLUE] * (len(n) - 1), height=0.6)
    for yi, ni, prev in zip(y, n, [None] + n[:-1]):
        ax.text(ni + 8000, yi, f"{ni:,}" + ("" if prev is None else f"  (-{prev - ni:,})"), va="center", fontsize=8.5)
    ax.set_yticks(y, names)
    ax.set_xticks([])
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(0, max(n) * 1.3)
    save(fig, "a5_funnel.pdf")


# --------------------------------------------------------------------------- results


def fig_r2(fits, preds, y_test) -> None:
    mse_b = ((y_test - preds["Benchmark"]) ** 2).mean()
    ins = [100 * fits[m].rsquared for m in MODELS]
    oos = [100 * (1 - ((y_test - preds[m]) ** 2).mean() / mse_b) for m in MODELS]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(x, ins, marker="o", ms=6, lw=1.8, color=BLUE, label="in-sample R$^2$ (2001-2019)")
    ax.plot(x, oos, marker="o", ms=6, lw=1.8, color=ORANGE, label="out-of-sample R$^2$ (2020-2025)")
    for xi, (a, b) in enumerate(zip(ins, oos)):
        ax.annotate(f"{a:.3f}%", (xi, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)
        ax.annotate(f"{b:+.3f}%", (xi, b), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5)
    ax.set_xticks(x, ["M1\n1 predictor", "M2\n3 predictors", "M3\n5 predictors"])
    ax.set_ylabel("R$^2$ (%)")
    ax.set_ylim(-0.04, 0.17)
    ax.set_xlim(-0.4, 2.4)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
    save(fig, "f2_r2.pdf")


def fig_pred_vs_real(train, test, fits) -> None:
    cols = MODELS["M3"]
    p_tr = fits["M3"].predict(sm.add_constant(train[cols], has_constant="add"))
    p_te = fits["M3"].predict(sm.add_constant(test[cols], has_constant="add"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.8))

    ax = axes[0]
    bins = np.linspace(-0.4, 0.4, 161)
    ax.hist(test["ret_next"].clip(-0.4, 0.4), bins=bins, color=MUTED, label="realized returns", density=True)
    ax.hist(p_te, bins=bins, color=ORANGE, label="M3 forecasts", density=True, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("next-month return (test period, 2020-2025)")
    ax.set_ylabel("density (log scale)")
    ax.set_title("(a) Forecasts vs. what actually happened")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left")
    ax.text(0.98, 0.95, f"forecasts: {p_te.quantile(0.01):.1%} to {p_te.quantile(0.99):.1%}\n"
            f"realized: {test['ret_next'].quantile(0.01):.0%} to {test['ret_next'].quantile(0.99):.0%}\n(1st-99th pct.)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)

    ax = axes[1]
    for d, p, color, name in [(train, p_tr, BLUE, "training 2001-2019"), (test, p_te, ORANGE, "test 2020-2025")]:
        dec = pd.qcut(p, 10, labels=False)
        g = pd.DataFrame({"p": p, "y": d["ret_next"], "dec": dec}).groupby("dec").mean()
        ax.plot(g["p"], g["y"], marker="o", ms=4.5, lw=1.6, color=color, label=name)
    lo, hi = 0.0, 0.022
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", label="perfect forecast")
    ax.set_xlabel("average M3 forecast (by forecast decile)")
    ax.set_ylabel("average realized return")
    ax.set_title("(b) Do higher forecasts deliver higher returns?")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.legend(loc="upper left")
    fig.subplots_adjust(wspace=0.28)
    save(fig, "f3_pred_vs_real.pdf")


def fig_horizon(test, preds) -> None:
    """Frozen models used month after month over 2020-2025."""
    d = pd.DataFrame({"m": test["target_m"], "y": test["ret_next"]})
    for name in ["Benchmark", *MODELS]:
        d[name] = preds[name]
    x = yfrac(np.sort(d["m"].unique()) + 1)  # plot at the end of the target month
    styles = {"Benchmark": (INK2, "--", "historical mean"), "M1": (BLUE, "-", "M1"),
              "M2": (ORANGE, "-", "M2"), "M3": (AQUA, "-", "M3")}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))

    ax = axes[0]
    g = d.groupby("m").mean()
    ax.plot(x, (1 + g["y"]).cumprod(), color=INK, lw=1.8, label="reality")
    for name, (c, ls, lab) in styles.items():
        ax.plot(x, (1 + g[name]).cumprod(), color=c, ls=ls, lw=1.3, label=lab)
    ax.set_title("(a) Value of \\$1 in the average stock")
    ax.set_ylabel("growth of \\$1")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    ax.axhline(0, color=INK2, lw=0.6)
    se_b = ((d["y"] - d["Benchmark"]) ** 2).groupby(d["m"]).sum()
    for name in MODELS:
        se_m = ((d["y"] - d[name]) ** 2).groupby(d["m"]).sum()
        ax.plot(x, (se_b - se_m).cumsum(), color=styles[name][0], lw=1.4, label=name)
    ax.set_title("(b) Cumulative error saved vs. mean")
    ax.set_ylabel("above 0: model ahead so far")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[2]
    ax.axhline(1, color=INK2, lw=0.6)
    for name in MODELS:
        dec = d.groupby("m")[name].transform(lambda p: pd.qcut(p, 10, labels=False, duplicates="drop"))
        top = d["y"].where(dec == 9).groupby(d["m"]).mean()
        bottom = d["y"].where(dec == 0).groupby(d["m"]).mean()
        ax.plot(x, (1 + top - bottom).cumprod(), color=styles[name][0], lw=1.4, label=name)
    ax.set_title("(c) Long top 10%, short bottom 10%")
    ax.set_ylabel("growth of \\$1, before costs")
    ax.legend(loc="upper left", fontsize=7.5)
    for ax in axes:
        ax.set_xticks([2020, 2022, 2024, 2026])
    fig.subplots_adjust(wspace=0.32)
    save(fig, "f6_horizon.pdf")


def fig_coefs(train, test) -> None:
    cols = MODELS["M3"]
    res = {}
    for name, d in [("train", train), ("test", test)]:
        f = sm.OLS(d["ret_next"], sm.add_constant(d[cols])).fit(cov_type="HC1")
        res[name] = (80 * f.params[cols], 80 * 1.96 * f.bse[cols])
    y = np.arange(len(cols))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.axvline(0, color=INK2, lw=0.8)
    for off, name, color, lab in [(0.17, "train", BLUE, "estimated on 2001-2019"),
                                  (-0.17, "test", ORANGE, "re-estimated on 2020-2025")]:
        b, ci = res[name]
        ax.barh(y + off, b, height=0.32, color=color, label=lab)
        ax.errorbar(b, y + off, xerr=ci, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_yticks(y, [LABELS[c] for c in cols])
    ax.set_xlabel("90th minus 10th percentile stock:\npredicted return gap (pp per month)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    save(fig, "f4_coefs.pdf")


def fig_fe(train, test) -> None:
    cols = MODELS["M3"]
    g = train.groupby("permno")
    y_dm = train["ret_next"] - g["ret_next"].transform("mean")
    X_dm = train[cols] - g[cols].transform("mean")
    beta = sm.OLS(y_dm, X_dm).fit().params
    means = g[["ret_next", *cols]].mean()
    alpha = means["ret_next"] - means[cols] @ beta
    t = test[test["permno"].isin(alpha.index)]
    both = pd.concat([means["ret_next"].rename("tr"), t.groupby("permno")["ret_next"].mean().rename("te")],
                     axis=1).dropna()

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.scatter(both["tr"].clip(-0.1, 0.1), both["te"].clip(-0.1, 0.1), s=5, color=BLUE, alpha=0.35, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("stock's average monthly return, 2001-2019")
    ax.set_ylabel("same stock, 2020-2025")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.text(0.03, 0.97, f"{len(both):,} stocks\ncorrelation = {both.corr().iloc[0, 1]:.2f}", transform=ax.transAxes,
            va="top", fontsize=8, color=INK2)
    save(fig, "f5_fe_persistence.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEAT).merge(pd.read_parquet(RAW, columns=["permno", "yyyymm", "ticker"]),
                                     on=["permno", "yyyymm"], how="left")
    train, test = load_usable()
    usable = pd.concat([train, test])
    fits = fit_models(train)
    preds = {"Benchmark": pd.Series(train["ret_next"].mean(), index=test.index)}
    for name, fit in fits.items():
        preds[name] = fit.predict(sm.add_constant(test[MODELS[name]], has_constant="add"))

    fig_split(df, usable)
    fig_r2(fits, preds, test["ret_next"])
    fig_pred_vs_real(train, test, fits)
    fig_horizon(test, preds)
    fig_coefs(train, test)
    fig_fe(train, test)
    fig_panel(df)
    fig_gap(df)
    fig_windows()
    fig_ranks(df)
    fig_funnel(df)


if __name__ == "__main__":
    main()
