"""Market data, feature engineering, feature selection and time-series splits.

This is the data half of the training pipeline (the model half is
`src/model.py`). The feature functions are the ones developed in
`notebooks/Financial_Analysis.ipynb`, moved here so that training is a
repeatable script instead of a notebook run.

    python -m src.data --refresh     # download fresh prices and fundamentals
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
PRICES_PATH = SNAPSHOT_DIR / "prices.csv.gz"
FUNDAMENTALS_PATH = SNAPSHOT_DIR / "fundamentals.csv"
METADATA_PATH = SNAPSHOT_DIR / "metadata.csv"

CFG: dict = {
    "MARKET": "NSE India",
    "START": "2017-01-01",
    "HORIZON": 21,  # forecast the return over the next 21 trading days
    "EMBARGO": 10,  # extra trading days dropped between train and validation
    "TEST_FRAC": 0.20,  # most recent 20% of dates = final hold-out
    "N_SPLITS": 5,  # walk-forward cross-validation folds
    "QUANTILES": [0.1, 0.5, 0.9],
    "N_FEATURES": 10,  # features kept by the selection step
    "FRAC_THRESH": 1e-4,
    "RF_ANNUAL": 0.065,  # Indian risk-free rate used in Sharpe ratios
    "SEED": 42,
}

# Five liquid large caps per sector, so every sector is represented.
SECTOR_UNIVERSE: dict[str, list[str]] = {
    "Banks & NBFC": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT Services": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Oil, Gas & Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Automobiles": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
    "Pharma & Healthcare": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Capital Goods & Defence": ["LT.NS", "ABB.NS", "SIEMENS.NS", "BEL.NS", "HAL.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "VEDL.NS"],
    "Utilities & Power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "JSWENERGY.NS", "TORNTPOWER.NS"],
    "Consumer Discretionary & Retail": ["TITAN.NS", "TRENT.NS", "DMART.NS", "JUBLFOOD.NS", "ETERNAL.NS"],
    "Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS", "PIIND.NS", "DEEPAKNTR.NS"],
    "Real Estate & Construction": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS"],
}
UNIVERSE = [ticker for tickers in SECTOR_UNIVERSE.values() for ticker in tickers]
SECTOR_OF = {ticker: sector for sector, tickers in SECTOR_UNIVERSE.items() for ticker in tickers}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
FUNDAMENTAL_KEYS = {
    "trailingPE": "f_PE",
    "priceToBook": "f_PB",
    "returnOnEquity": "f_ROE",
    "debtToEquity": "f_DE",
    "profitMargins": "f_NPM",
    "dividendYield": "f_DivYld",
    "revenueGrowth": "f_RevG",
}
METADATA_KEYS = [
    "sector", "industry", "marketCap", "forwardPE", "trailingEps", "earningsGrowth",
    "grossMargins", "operatingMargins", "currentRatio", "quickRatio", "beta", "recommendationKey",
]

# Candidate model inputs. All are computed from past prices and volumes only.
# Company fundamentals are NOT model inputs: only today's snapshot is available,
# and repeating it across past dates would leak future information into
# training. They are used by the app's separate fundamental score instead.
PER_STOCK_FEATURES = [
    "RSI_14", "MACD_signal", "Boll_BW", "ATR_14", "Stoch_K", "OBV_delta", "Williams_R",
    "ret_5", "ret_21", "vol_21", "fracdiff_close",
    "ret_63", "mom_126_21", "vol_63", "dist_52w_high", "volume_surge",
]
CROSS_SECTION_FEATURES = ["rel_ret_21", "mkt_ret_21", "mkt_vol_21"]
CANDIDATE_FEATURES = PER_STOCK_FEATURES + CROSS_SECTION_FEATURES
LABEL = "fwd_ret"


# ------------------------------------------------------------------- downloads
def download_prices(tickers: list[str], start: str, end: str | None = None, batch_size: int = 50) -> dict[str, pd.DataFrame]:
    """Daily adjusted OHLCV per ticker from Yahoo Finance."""
    import yfinance as yf

    loaded: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, group_by="ticker", threads=True, progress=False)
        for ticker in batch:
            try:
                frame = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                frame = frame[OHLCV].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(frame) > 250:
                frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                loaded[ticker] = frame.astype(float)
    return loaded


def download_fundamentals(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Today's fundamental ratios and descriptive metadata per ticker."""
    import yfinance as yf

    ratios, meta = {}, {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:  # noqa: BLE001 - provider metadata is optional
            info = {}
        ratios[ticker] = {name: info.get(key, np.nan) for key, name in FUNDAMENTAL_KEYS.items()}
        meta[ticker] = {key: info.get(key, np.nan) for key in METADATA_KEYS}
    funda = pd.DataFrame(ratios).T.apply(pd.to_numeric, errors="coerce")
    metadata = pd.DataFrame(meta).T.rename(columns={"marketCap": "market_cap"})
    metadata["sector"] = [SECTOR_OF.get(t, metadata.loc[t, "sector"]) for t in metadata.index]
    for column in metadata.columns.difference(["sector", "industry", "recommendationKey"]):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    return funda, metadata


def save_snapshot(prices: dict[str, pd.DataFrame], funda: pd.DataFrame, metadata: pd.DataFrame) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    long = pd.concat([frame.assign(ticker=ticker) for ticker, frame in prices.items()])
    long.index.name = "date"
    long.round(4).to_csv(PRICES_PATH, compression="gzip")
    funda.to_csv(FUNDAMENTALS_PATH)
    metadata.to_csv(METADATA_PATH)


def load_snapshot() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """The committed data snapshot: training from it is fully reproducible."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"No data snapshot at {PRICES_PATH}. Run `python -m src.data --refresh`.")
    long = pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True)
    prices = {ticker: frame[OHLCV] for ticker, frame in long.groupby("ticker")}
    funda = pd.read_csv(FUNDAMENTALS_PATH, index_col=0)
    metadata = pd.read_csv(METADATA_PATH, index_col=0)
    return prices, funda, metadata


def refresh_snapshot(tickers: list[str] | None = None, start: str | None = None) -> None:
    tickers = tickers or UNIVERSE
    prices = download_prices(tickers, start or CFG["START"])
    if len(prices) < 5:
        raise RuntimeError(f"Only {len(prices)} tickers returned price data.")
    funda, metadata = download_fundamentals(sorted(prices))
    save_snapshot(prices, funda, metadata)
    last = max(frame.index.max() for frame in prices.values())
    print(f"Snapshot saved: {len(prices)} tickers, prices to {last.date()}")


# ------------------------------------------------------------------ indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd - macd.ewm(span=signal, adjust=False).mean()


def bollinger_bandwidth(close: pd.Series, n: int = 20) -> pd.Series:
    return 4 * close.rolling(n).std() / close.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False).mean()


def stochastic_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lowest, highest = df["Low"].rolling(n).min(), df["High"].rolling(n).max()
    return 100 * (df["Close"] - lowest) / (highest - lowest)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    highest, lowest = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    return -100 * (highest - df["Close"]) / (highest - lowest)


def frac_weights(d: float, thresh: float) -> np.ndarray:
    weights, k = [1.0], 1
    while True:
        nxt = -weights[-1] * (d - k + 1) / k
        if abs(nxt) < thresh or k > 2000:
            break
        weights.append(nxt)
        k += 1
    return np.array(weights[::-1])  # oldest -> newest


def frac_diff(series: pd.Series, d: float, thresh: float) -> pd.Series:
    """Fractional differencing: removes the trend but keeps price memory."""
    weights = frac_weights(d, thresh)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= len(weights):
        out[len(weights) - 1 :] = np.correlate(values, weights, mode="valid")
    return pd.Series(out, index=series.index)


def optimal_d(series: pd.Series, thresh: float, pval: float = 0.05) -> float:
    """Smallest differencing order that passes the ADF stationarity test."""
    for d in np.linspace(0, 1, 11):
        diffed = frac_diff(series, d, thresh).dropna()
        if len(diffed) >= 100 and adfuller(diffed, maxlag=1, autolag=None)[1] < pval:
            return float(d)
    return 1.0


def stock_features(df: pd.DataFrame, cfg: dict, fit_until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-stock features. `fit_until` limits the data used to choose the
    differencing order, so the hold-out period cannot influence it."""
    close = df["Close"]
    daily = close.pct_change()
    log_close = np.log(close)
    history = log_close if fit_until is None else log_close[log_close.index <= fit_until]
    d_star = optimal_d(history, cfg["FRAC_THRESH"])
    volume_base = df["Volume"].rolling(60).mean()
    return pd.DataFrame(
        {
            "RSI_14": rsi(close),
            "MACD_signal": macd_histogram(close) / close,  # scaled by price so stocks compare
            "Boll_BW": bollinger_bandwidth(close),
            "ATR_14": atr(df) / close,
            "Stoch_K": stochastic_k(df),
            "OBV_delta": obv(df).pct_change(5).clip(-5, 5),
            "Williams_R": williams_r(df),
            "ret_5": close.pct_change(5),
            "ret_21": close.pct_change(21),
            "vol_21": daily.rolling(21).std(),
            "fracdiff_close": frac_diff(log_close, d_star, cfg["FRAC_THRESH"]),
            "ret_63": close.pct_change(63),
            "mom_126_21": close.shift(21) / close.shift(126) - 1,  # 6-month momentum, skipping last month
            "vol_63": daily.rolling(63).std(),
            "dist_52w_high": close / close.rolling(252).max() - 1,
            "volume_surge": df["Volume"].rolling(5).mean() / volume_base.replace(0, np.nan) - 1,
        },
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan)


def build_panel(prices: dict[str, pd.DataFrame], cfg: dict, fit_until: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (date, stock): features, ticker, and the forward return.

    The label is the return over the next `HORIZON` trading days. It is
    missing for the most recent `HORIZON` days; those rows are kept so the
    app can score today's market, and are never used for training.
    """
    frames = []
    for ticker, df in prices.items():
        feat = stock_features(df, cfg, fit_until)
        feat[LABEL] = df["Close"].shift(-cfg["HORIZON"]) / df["Close"] - 1
        feat["ticker"] = ticker
        frames.append(feat)
    panel = pd.concat(frames).sort_index()

    by_day = panel.groupby(level=0)
    panel["mkt_ret_21"] = by_day["ret_21"].transform("mean")
    panel["mkt_vol_21"] = by_day["vol_21"].transform("mean")
    panel["rel_ret_21"] = panel["ret_21"] - panel["mkt_ret_21"]
    panel.index.name = "date"
    return panel.dropna(subset=CANDIDATE_FEATURES)


# ------------------------------------------------------------ feature selection
def select_features(X: pd.DataFrame, y: pd.Series, k: int, redundancy: float = 0.5, seed: int = 0) -> tuple[list[str], pd.DataFrame]:
    """Greedy relevance-minus-redundancy selection using mutual information.

    Start with the feature most informative about the label, then repeatedly
    add the one with the best `relevance - redundancy * (mean mutual
    information with the features already chosen)`.
    """
    relevance = pd.Series(mutual_info_regression(X.values, y.values, random_state=seed), index=X.columns)
    chosen = [relevance.idxmax()]
    rows = [{"feature": chosen[0], "relevance": relevance[chosen[0]], "redundancy": 0.0, "score": relevance[chosen[0]]}]
    pool = [c for c in X.columns if c not in chosen]
    pair: dict[tuple[str, str], float] = {}
    while len(chosen) < k and pool:
        best, best_score, best_red = None, -np.inf, 0.0
        for candidate in pool:
            for held in chosen:
                if (candidate, held) not in pair:
                    pair[(candidate, held)] = mutual_info_regression(X[[candidate]].values, X[held].values, random_state=seed)[0]
            red = float(np.mean([pair[(candidate, held)] for held in chosen]))
            score = relevance[candidate] - redundancy * red
            if score > best_score:
                best, best_score, best_red = candidate, score, red
        chosen.append(best)
        pool.remove(best)
        rows.append({"feature": best, "relevance": relevance[best], "redundancy": best_red, "score": best_score})
    table = pd.DataFrame(rows)
    dropped = pd.DataFrame({"feature": pool, "relevance": relevance[pool].values})
    table["selected"] = True
    dropped["selected"] = False
    return chosen, pd.concat([table, dropped], ignore_index=True)


# ------------------------------------------------------------------ time splits
def purge_gap(cfg: dict) -> int:
    """Trading days removed before each validation block. A training label
    looks `HORIZON` days ahead, so without this gap it would overlap the
    validation period; `EMBARGO` adds a further safety margin."""
    return cfg["HORIZON"] + cfg["EMBARGO"]


def holdout_dates(dates: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Development dates and final hold-out dates, separated by the purge gap."""
    cut = int(len(dates) * (1 - cfg["TEST_FRAC"]))
    return dates[: cut - purge_gap(cfg)], dates[cut:]


def walk_forward_splits(dates: np.ndarray, cfg: dict) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: always train on the past, validate on the
    next block, with the purge gap in between."""
    n_blocks = cfg["N_SPLITS"] + 1
    edges = np.linspace(0, len(dates), n_blocks + 1).astype(int)
    gap = purge_gap(cfg)
    for fold in range(1, n_blocks):
        train = dates[: edges[fold] - gap]
        validate = dates[edges[fold] : edges[fold + 1]]
        yield train, validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download fresh data into data/snapshot/")
    args = parser.parse_args()
    if args.refresh:
        refresh_snapshot()
    else:
        prices, _, _ = load_snapshot()
        panel = build_panel(prices, CFG)
        print(f"{len(prices)} tickers | panel {panel.shape} | {panel.index.min().date()} to {panel.index.max().date()}")
