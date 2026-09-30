"""Build the target and the five ranked predictors for the stock-return case.

All time shifts are done on a calendar grid (permno x month), never on row
order, so a missing month can never be mistaken for the previous/next month.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "US_Stocks_PanelDataset.csv"  # WRDS/CRSP export, not distributed
RAW = ROOT / "data" / "US_Stocks_PanelDataset.parquet"
OUT = ROOT / "data" / "panel_features.parquet"

# Training targets are returns realised up to Dec 2019; everything after is test.
LAST_TRAIN_TARGET = 2019 * 12 + 12

PREDICTORS = ["ret", "mom", "size", "vol", "turnover"]


def month_index(yyyymm: pd.Series) -> pd.Series:
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["permno", "yyyymm", "ret", "market_cap", "volume", "shares_out"]].copy()
    df["m"] = month_index(df["yyyymm"])

    # Wide return matrix: one row per stock, one column per calendar month.
    # Months in which a stock is absent are NaN, so shifts are calendar-exact.
    months = np.arange(df["m"].min(), df["m"].max() + 1)
    wide = df.pivot(index="permno", columns="m", values="ret").reindex(columns=months)
    log_gross = np.log1p(wide)

    # Target: next calendar month's return (NaN if that month is missing).
    target = wide.shift(-1, axis=1)
    # Momentum 12-2: compounded return over t-12..t-2, all 11 months required.
    mom = np.expm1(log_gross.shift(2, axis=1).T.rolling(11, min_periods=11).sum().T)
    # Volatility: SD of returns t-11..t, at least 8 of the 12 months observed.
    vol = wide.T.rolling(12, min_periods=8).std().T

    def to_long(mat: pd.DataFrame, name: str) -> pd.Series:
        return mat.stack().rename(name)

    extra = pd.concat(
        [to_long(target, "ret_next"), to_long(mom, "mom"), to_long(vol, "vol")], axis=1
    )
    extra.index.names = ["permno", "m"]
    df = df.merge(extra, left_on=["permno", "m"], right_index=True, how="left")

    df["size"] = df["market_cap"]
    # Units of volume/shares_out do not matter: a constant factor leaves ranks unchanged.
    df["turnover"] = df["volume"] / df["shares_out"]

    # Cross-sectional percentile rank within each month, among stocks with the
    # characteristic available that month.
    for col in PREDICTORS:
        df[f"r_{col}"] = df.groupby("m")[col].rank(pct=True)

    df["target_m"] = df["m"] + 1
    df["sample"] = np.where(df["target_m"] <= LAST_TRAIN_TARGET, "train", "test")
    return df


def main() -> None:
    if not RAW.exists():
        pd.read_csv(CSV).to_parquet(RAW, index=False)
    df = build_panel(pd.read_parquet(RAW))
    rank_cols = [f"r_{c}" for c in PREDICTORS]
    usable = df.dropna(subset=["ret_next", *rank_cols])

    print(f"raw rows: {len(df):,}")
    print(f"rows without a consecutive next-month target: {df['ret_next'].isna().sum():,}")
    print(f"usable complete-case rows: {len(usable):,}\n")
    summary = usable.groupby("sample").agg(
        obs=("permno", "size"),
        securities=("permno", "nunique"),
        months=("m", "nunique"),
        first_t=("yyyymm", "min"),
        last_t=("yyyymm", "max"),
    )
    print(summary.loc[["train", "test"]].to_string())

    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwritten {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
