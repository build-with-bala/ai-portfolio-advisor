from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bundle() -> dict:
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    dates = pd.bdate_range("2023-01-02", periods=280)
    prices = {}
    panels = []
    result_rows = []
    fundamentals = {}
    metadata = {}
    for number, ticker in enumerate(tickers):
        rng = np.random.default_rng(number)
        daily = rng.normal(0.00035 + number * 0.00003, 0.009 + number * 0.0004, len(dates))
        close = 100 * np.exp(np.cumsum(daily))
        frame = pd.DataFrame(
            {
                "Open": close * 0.998,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(len(dates), 1_000_000 + number * 10_000),
            },
            index=dates,
        )
        prices[ticker] = frame
        fundamentals[ticker] = {
            "f_PE": 12 + number * 4,
            "f_PB": 1.2 + number * 0.25,
            "f_ROE": 0.24 - number * 0.015,
            "f_DE": 25 + number * 15,
            "f_NPM": 0.18 - number * 0.01,
            "f_DivYld": 0.01 + number * 0.002,
            "f_RevG": 0.12 - number * 0.008,
        }
        metadata[ticker] = {"sector": "Technology" if number < 3 else "Healthcare", "industry": "Test"}
        for date in dates:
            panels.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "RSI_14": 55 + number,
                    "MACD_signal": 0.02 + number * 0.01,
                    "ret_5": 0.01 + number * 0.002,
                    "ret_21": 0.04 + number * 0.003,
                    "fracdiff_close": 0.01 + number * 0.001,
                    "OBV_delta": 0.02,
                    "Stoch_K": 58 + number,
                    "Williams_R": -42 - number,
                    "vol_21": 0.01 + number * 0.0005,
                    "ATR_14": 0.015 + number * 0.0005,
                    "Boll_BW": 0.12 + number * 0.01,
                }
            )
            result_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "fwd_ret": 0.01 + number * 0.001,
                    "pred_0.1": 0.002 + number * 0.001,
                    "pred_0.5": 0.015 + number * 0.002,
                    "pred_0.9": 0.028 + number * 0.003,
                    "confidence": 0.65 - number * 0.03,
                    "signal": "BUY" if number < 5 else "HOLD",
                }
            )
    panel = pd.DataFrame(panels).set_index("date")
    res = pd.DataFrame(result_rows).set_index("date")
    return {
        "res": res,
        "prices": prices,
        "models": {},
        "panel_test": panel,
        "features": ["RSI_14", "MACD_signal", "ret_21", "f_PE", "f_ROE"],
        "funda": pd.DataFrame(fundamentals).T,
        "metadata": pd.DataFrame(metadata).T,
    }
