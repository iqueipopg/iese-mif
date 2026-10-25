"""Shared data loading: daily percentage log returns and hourly realized variance."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
FORECASTS = OUT / "forecasts.parquet"
PARAMS = OUT / "rolling_params.parquet"

TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END = pd.Timestamp("2026-08-31")
HAC_LAGS = 7  # one week of daily data; Bitcoin trades every calendar day

for d in (TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    """One row per calendar day: price, return r_t = 100*dlog(P_t), RV_t, RV validity flag."""
    px = pd.read_csv(DATA / "bitcoin_daily_prices.csv", parse_dates=["Date"], index_col="Date")
    rv = pd.read_csv(DATA / "bitcoin_realized_variance.csv", parse_dates=["Date"], index_col="Date")
    df = px.join(rv, how="left")
    df["r"] = 100 * np.log(df["Price"]).diff()
    df = df.dropna(subset=["r"])  # first return is 18 Aug 2017
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all(), "calendar gaps in prices"
    return df


def train(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:TRAIN_END]


def oos(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[OOS_START:OOS_END]


def save(table: pd.DataFrame, name: str, fmt: str = "{:.4f}") -> None:
    table.to_csv(TABLES / f"{name}.csv")
    print(f"\n== {name} ==")
    print(table.to_string(float_format=lambda v: fmt.format(v)))
