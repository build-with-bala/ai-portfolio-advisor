"""Build the production NSE artifact from the notebook's post-Step-18 logic."""

from __future__ import annotations

import ast
import io
import json
import pickle
import urllib.request
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, skew
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
FALLBACK_SECTOR_UNIVERSE = {
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


def nifty500_universe() -> tuple[list[str], dict[str, str], str]:
    """Fetch the current Nifty 500 symbols and industry labels."""

    request = urllib.request.Request(NIFTY500_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            frame = pd.read_csv(io.BytesIO(response.read()))
        symbol_column = next(column for column in frame.columns if str(column).strip().lower() == "symbol")
        industry_column = next((column for column in frame.columns if str(column).strip().lower() in {"industry", "sector"}), None)
        symbols = frame[symbol_column].astype(str).str.strip()
        symbols = symbols[symbols.ne("") & symbols.ne("nan")]
        tickers = [symbol if symbol.endswith(".NS") else f"{symbol}.NS" for symbol in symbols]
        sectors = {
            ticker: str(frame.iloc[index][industry_column]).strip()
            for index, ticker in zip(frame.index, tickers)
            if industry_column and pd.notna(frame.iloc[index][industry_column])
        }
        if len(tickers) >= 400:
            return tickers, sectors, "Nifty 500 constituent file"
    except Exception as exc:  # noqa: BLE001 - a build can use the explicit fallback
        print(f"Nifty 500 file unavailable: {exc}")
    fallback = [ticker for values in FALLBACK_SECTOR_UNIVERSE.values() for ticker in values]
    return fallback, {ticker: sector for sector, values in FALLBACK_SECTOR_UNIVERSE.items() for ticker in values}, "fallback liquid sector sample"


def bulk_load_prices(tickers: list[str], start: str, end: str, batch_size: int = 50) -> dict[str, pd.DataFrame]:
    """Download many NSE histories in batches so the broad universe is practical."""

    loaded: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - skip an unavailable batch
            print(f"price batch unavailable ({offset}:{offset + len(batch)}): {exc}")
            continue
        for ticker in batch:
            try:
                if len(batch) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                    frame = raw.copy()
                elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                    frame = raw[ticker].copy()
                elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(1):
                    frame = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
                required = ["Open", "High", "Low", "Close", "Volume"]
                if all(column in frame for column in required) and len(frame) > 250:
                    loaded[ticker] = frame[required].dropna(subset=["Close"])
            except (KeyError, TypeError, ValueError):
                continue
        print(f"Price batches: {min(offset + batch_size, len(tickers))}/{len(tickers)} | loaded {len(loaded)}")
    return loaded


def notebook_functions(notebook: Path) -> dict[str, object]:
    cells = json.loads(notebook.read_text())["cells"]
    namespace: dict[str, object] = {
        "np": np,
        "pd": pd,
        "yf": yf,
        "lgb": lgb,
        "shap": shap,
        "linkage": linkage,
        "squareform": squareform,
        "norm": norm,
        "skew": skew,
        "kurtosis": kurtosis,
        "adfuller": adfuller,
        "mutual_info_regression": mutual_info_regression,
    }
    for index in range(3, 19):
        source = "".join(cells[index].get("source", []))
        nodes = [
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if nodes:
            module = ast.Module(body=nodes, type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, f"notebook-cell-{index}", "exec"), namespace)
    return namespace


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ns = notebook_functions(root / "notebooks" / "Financial_Analysis.ipynb")
    universe, sector_map, universe_source = nifty500_universe()
    sector_universe: dict[str, list[str]] = {}
    for ticker in universe:
        sector_universe.setdefault(sector_map.get(ticker, "Unclassified"), []).append(ticker)
    cfg = {
        "UNIVERSE": universe,
        "UNIVERSE_SOURCE": universe_source,
        "SECTOR_UNIVERSE": sector_universe,
        "START": "2017-01-01",
        "END": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "HORIZON": 21,
        "TEST_FRAC": 0.20,
        "EMBARGO": 10,
        "QUANTILES": [0.1, 0.5, 0.9],
        "RF_ANNUAL": 0.02,
        "FRAC_THRESH": 1e-4,
    }
    prices = bulk_load_prices(cfg["UNIVERSE"], cfg["START"], cfg["END"])
    if len(prices) < 5:
        raise RuntimeError(f"Only {len(prices)} NSE tickers returned data")
    funda = pd.DataFrame({ticker: ns["fundamental_snapshot"](ticker) for ticker in prices}).T
    funda = funda.apply(pd.to_numeric, errors="coerce").fillna(funda.median())
    funda.columns = ["f_PE", "f_PB", "f_ROE", "f_DE", "f_NPM", "f_DivYld", "f_RevG"]
    panel = ns["build_panel"](prices, funda, cfg)
    selection = panel.sample(min(len(panel), 200_000), random_state=0) if len(panel) > 200_000 else panel
    features = ns["isf_mid"](selection.drop(columns=["fwd_ret", "ticker"]), selection["fwd_ret"], k=12)
    train, test = ns["purged_split"](panel, cfg)
    models, predictions = ns["train_quantile_models"](train, test, features, cfg["QUANTILES"])
    result = test[["ticker", "fwd_ret"]].copy()
    for quantile in cfg["QUANTILES"]:
        result[f"pred_{quantile}"] = predictions[quantile]
    band = result["pred_0.9"] - result["pred_0.1"]
    result["confidence"] = result.assign(_b=band).groupby(level=0)["_b"].transform(
        lambda values: 1 - values.rank(pct=True)
    )
    result = ns["assign_signals_cross_sectional"](result)

    keys = [
        "sector", "industry", "marketCap", "forwardPE", "trailingEps", "earningsGrowth",
        "grossMargins", "operatingMargins", "currentRatio", "quickRatio", "beta",
        "recommendationKey",
    ]
    metadata_rows = {}
    for ticker in sorted(prices):
        try:
            info = yf.Ticker(ticker).info
            metadata_rows[ticker] = {key: info.get(key, np.nan) for key in keys}
        except Exception as exc:  # provider metadata is optional for the artifact
            print(f"metadata unavailable for {ticker}: {exc}")
            metadata_rows[ticker] = {key: np.nan for key in keys}
    metadata = pd.DataFrame(metadata_rows).T.rename(columns={"marketCap": "market_cap"})
    metadata["sector"] = [sector_map.get(ticker, metadata.loc[ticker, "sector"]) for ticker in metadata.index]
    for column in metadata.columns:
        if column not in {"sector", "industry", "recommendationKey"}:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    funda_export = funda.copy()
    for column in [
        "forwardPE", "trailingEps", "earningsGrowth", "grossMargins", "operatingMargins",
        "currentRatio", "quickRatio", "beta",
    ]:
        if column in metadata:
            funda_export[f"f_{column}"] = metadata[column]
    funda_export["f_market_cap"] = metadata.get("market_cap", np.nan)

    bundle = {
        "res": result,
        "prices": prices,
        "models": models,
        "panel_test": test,
        "features": features,
        "cfg": cfg,
        "funda": funda_export,
        "metadata": metadata,
        "exporter_version": "2026-08-27-india-production-v2-nifty500",
        "market": "NSE India",
    }
    output = root / "data"
    output.mkdir(exist_ok=True)
    with (output / "artifacts.pkl").open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata.to_csv(output / "metadata.csv")
    manifest = {
        "exporter_version": bundle["exporter_version"],
        "market": "NSE India",
        "assets": sorted(prices),
        "selected_features": features,
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "fundamental_point_in_time": False,
        "notes": [
            "Universe is sourced from the current Nifty 500 constituent file when available.",
            "Fundamentals are current yfinance snapshots, not historical point-in-time data.",
            "Research and education only; no trades are placed.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"ARTIFACT_READY assets={len(prices)} rows={len(result):,} features={len(features)}")
    print("latest_signal_counts=", result.groupby(level=0)["signal"].last().value_counts().to_dict())


if __name__ == "__main__":
    main()
