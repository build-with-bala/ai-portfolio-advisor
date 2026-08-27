"""The portfolio decision engine used by both Streamlit and FastAPI.

The notebook remains the research/training surface. This module consumes the
post-Step-18 bundle and turns its model outputs into an auditable decision:
technical evidence + fundamental evidence + model uncertainty + portfolio fit.
It deliberately returns missing data as ``None`` instead of inventing zeroes.
"""

from __future__ import annotations

import math
import os
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

BUY_SIGNALS = {"BUY", "STRONG_BUY", "STRONG BUY", "ACCUMULATE"}
SELL_SIGNALS = {"SELL", "STRONG_SELL", "STRONG SELL"}

TECHNICAL_FEATURES = {
    "RSI_14",
    "MACD_signal",
    "Boll_BW",
    "ATR_14",
    "Stoch_K",
    "OBV_delta",
    "Williams_R",
    "ret_5",
    "ret_21",
    "vol_21",
    "fracdiff_close",
}

FUNDAMENTAL_FEATURES = {
    "f_PE",
    "f_PB",
    "f_ROE",
    "f_DE",
    "f_NPM",
    "f_DivYld",
    "f_RevG",
}


@dataclass(frozen=True)
class ProfilePreset:
    name: str
    description: str
    min_confidence: float
    max_annual_volatility: float
    max_drawdown: float
    max_position_weight: float
    min_fundamental_score: float
    technical_weight: float
    fundamental_weight: float
    model_weight: float
    risk_weight: float


PROFILE_PRESETS: dict[str, ProfilePreset] = {
    "Conservative": ProfilePreset(
        "Conservative",
        "Capital preservation first; stronger evidence, lower volatility, and wider diversification.",
        0.60,
        0.35,
        0.30,
        0.20,
        55.0,
        0.25,
        0.40,
        0.15,
        0.20,
    ),
    "Moderate": ProfilePreset(
        "Moderate",
        "Balances expected return, evidence agreement, volatility, and diversification.",
        0.40,
        0.60,
        0.45,
        0.30,
        45.0,
        0.30,
        0.30,
        0.25,
        0.15,
    ),
    "Aggressive": ProfilePreset(
        "Aggressive",
        "Accepts larger drawdowns and volatility when model return and momentum are stronger.",
        0.20,
        1.25,
        0.70,
        0.40,
        35.0,
        0.40,
        0.20,
        0.30,
        0.10,
    ),
}


def load_bundle(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the artifact produced by the notebook's post-Step-18 exporter."""

    artifact_path = Path(path or os.getenv("ARTIFACTS_PATH", "data/artifacts.pkl"))
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No portfolio artifact found at {artifact_path}. Run the notebook export step first."
        )
    try:
        with artifact_path.open("rb") as handle:
            bundle = pickle.load(handle)
    except NotImplementedError as exc:
        raise ValueError(
            "The artifact was created with a pandas string dtype that this runtime "
            "cannot restore. Re-run the notebook exporter from the latest repo, or "
            "rebuild the artifact with scripts/build_india_artifact.py."
        ) from exc
    if not isinstance(bundle, dict):
        raise TypeError("The artifact must be a dictionary created by the notebook exporter.")
    required = {"res", "prices", "models", "panel_test", "features"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Artifact is missing required keys: {', '.join(missing)}")
    return bundle


def make_pickle_portable(value: Any) -> Any:
    """Convert pandas string extension arrays to object strings before pickling.

    Colab and Docker can install different pandas point releases. Plain object
    string columns are slower, but they unpickle reliably across those runtimes.
    """

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.index = _portable_index(frame.index)
        frame.columns = _portable_index(frame.columns)
        for column in frame.columns:
            if is_string_dtype(frame[column].dtype):
                frame[column] = frame[column].astype("object")
        return frame
    if isinstance(value, pd.Series):
        series = value.copy()
        series.index = _portable_index(series.index)
        if is_string_dtype(series.dtype):
            series = series.astype("object")
        return series
    if isinstance(value, dict):
        return {key: make_pickle_portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_pickle_portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(make_pickle_portable(item) for item in value)
    return value


def _portable_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        arrays = []
        for level in range(index.nlevels):
            values = index.get_level_values(level)
            arrays.append(values.astype("object") if is_string_dtype(values.dtype) else values)
        return pd.MultiIndex.from_arrays(arrays, names=index.names)
    return index.astype("object") if is_string_dtype(index.dtype) else index


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp


def _date_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    return index


def _rows_on_date(frame: pd.DataFrame, asof: Any | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if frame.empty:
        raise ValueError("Cannot select a date from an empty frame.")
    index = _date_index(frame)
    target = _timestamp(asof) if asof is not None else index.max()
    mask = index.normalize() == target.normalize()
    if not mask.any():
        eligible = index[index <= target]
        if len(eligible) == 0:
            target = index.max()
        else:
            target = eligible.max()
        mask = index.normalize() == target.normalize()
    selected = frame.loc[mask].copy()
    return selected, pd.Timestamp(target).normalize()


def _numeric(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _clip(value: Any, low: float = -3.0, high: float = 3.0) -> float:
    number = _numeric(value)
    return float(np.clip(number if number is not None else 0.0, low, high))


def _zscore(series: pd.Series, inverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median(skipna=True)
    values = values.fillna(median if pd.notna(median) else 0.0)
    spread = values.std(ddof=0)
    if not np.isfinite(spread) or spread < 1e-12:
        result = pd.Series(0.0, index=series.index)
    else:
        result = ((values - values.mean()) / spread).clip(-3, 3)
    return -result if inverse else result


def _score_01(value: Any) -> float:
    return float(50.0 + 50.0 * np.tanh(_clip(value) / 1.5))


def _price_at(prices: Mapping[str, pd.DataFrame], ticker: str, asof: pd.Timestamp) -> float | None:
    frame = prices.get(ticker)
    if frame is None or "Close" not in frame or frame.empty:
        return None
    series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if series.empty:
        return None
    index = _date_index(series.to_frame())
    series = pd.Series(series.values, index=index)
    eligible = series[series.index <= asof]
    return _numeric(eligible.iloc[-1] if len(eligible) else series.iloc[-1])


def _risk_stats(prices: Mapping[str, pd.DataFrame], ticker: str, asof: pd.Timestamp) -> dict[str, Any]:
    frame = prices.get(ticker)
    empty = {
        "price": None,
        "annual_volatility": None,
        "max_drawdown": None,
        "var_95_daily": None,
        "cvar_95_daily": None,
        "momentum_63d": None,
        "observations": 0,
    }
    if frame is None or "Close" not in frame:
        return empty
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return empty
    close.index = _date_index(close.to_frame())
    close = close[close.index <= asof].tail(504)
    returns = close.pct_change().dropna()
    if returns.empty:
        empty["price"] = _numeric(close.iloc[-1])
        return empty
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    tail = returns.tail(252)
    var = float(tail.quantile(0.05)) if len(tail) else None
    cvar = float(tail[tail <= var].mean()) if var is not None and (tail <= var).any() else var
    momentum = close.iloc[-1] / close.iloc[-64] - 1.0 if len(close) > 64 else None
    return {
        "price": _numeric(close.iloc[-1]),
        "annual_volatility": _numeric(tail.std(ddof=1) * np.sqrt(252)) if len(tail) > 2 else None,
        "max_drawdown": _numeric(abs(drawdown.min())) if len(drawdown) else None,
        "var_95_daily": _numeric(var),
        "cvar_95_daily": _numeric(cvar),
        "momentum_63d": _numeric(momentum),
        "observations": len(close),
    }


def _fundamentals(bundle: Mapping[str, Any]) -> pd.DataFrame:
    funda = bundle.get("funda", bundle.get("fundamentals"))
    if funda is None:
        return pd.DataFrame()
    if isinstance(funda, dict):
        funda = pd.DataFrame(funda).T
    if not isinstance(funda, pd.DataFrame):
        return pd.DataFrame()
    result = funda.copy()
    result.index = result.index.astype(str)
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _metadata(bundle: Mapping[str, Any]) -> pd.DataFrame:
    metadata = bundle.get("metadata", bundle.get("asset_metadata"))
    if metadata is None:
        return pd.DataFrame()
    if isinstance(metadata, dict):
        metadata = pd.DataFrame(metadata).T
    if not isinstance(metadata, pd.DataFrame):
        return pd.DataFrame()
    result = metadata.copy()
    result.index = result.index.astype(str)
    return result


def _panel_day(bundle: Mapping[str, Any], asof: pd.Timestamp) -> pd.DataFrame:
    panel = bundle.get("panel_test")
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return pd.DataFrame()
    day, _ = _rows_on_date(panel, asof)
    if "ticker" not in day:
        return pd.DataFrame()
    return day.drop_duplicates("ticker").set_index("ticker", drop=False)


def _technical_scores(panel_day: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if panel_day.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    component_specs = {
        "RSI_14": lambda s: (s - 50.0) / 20.0,
        "MACD_signal": lambda s: s,
        "ret_5": lambda s: s,
        "ret_21": lambda s: s,
        "fracdiff_close": lambda s: s,
        "OBV_delta": lambda s: s,
        "Stoch_K": lambda s: (s - 50.0) / 25.0,
        "Williams_R": lambda s: -(s + 50.0) / 25.0,
        "vol_21": lambda s: -s,
        "ATR_14": lambda s: -s,
        "Boll_BW": lambda s: -s,
    }
    pieces: dict[str, pd.Series] = {}
    for feature, transform in component_specs.items():
        if feature in panel_day:
            raw = pd.to_numeric(panel_day[feature], errors="coerce")
            pieces[feature] = _zscore(transform(raw))
    if not pieces:
        return pd.Series(0.0, index=panel_day.index), pd.DataFrame(index=panel_day.index)
    components = pd.DataFrame(pieces, index=panel_day.index).fillna(0.0)
    weights = pd.Series(1.0, index=components.columns)
    for key in ("ret_21", "MACD_signal", "fracdiff_close"):
        if key in weights:
            weights[key] = 1.25
    score = components.mul(weights, axis=1).sum(axis=1) / weights.loc[components.columns].sum()
    return score.clip(-3, 3), components


def _fundamental_scores(funda: pd.DataFrame, tickers: pd.Index) -> tuple[pd.Series, pd.DataFrame]:
    if funda.empty:
        return pd.Series(0.0, index=tickers), pd.DataFrame(index=tickers)
    aligned = funda.reindex(tickers)
    specs = {
        "f_PE": True,
        "f_PB": True,
        "f_ROE": False,
        "f_DE": True,
        "f_NPM": False,
        "f_DivYld": False,
        "f_RevG": False,
    }
    pieces: dict[str, pd.Series] = {}
    for feature, inverse in specs.items():
        if feature not in aligned:
            continue
        values = pd.to_numeric(aligned[feature], errors="coerce")
        if feature in {"f_PE", "f_PB"}:
            values = values.where(values > 0, np.nan)
            values = np.log1p(values.clip(upper=250))
        pieces[feature] = _zscore(values, inverse=inverse)
    if not pieces:
        return pd.Series(0.0, index=tickers), pd.DataFrame(index=tickers)
    components = pd.DataFrame(pieces, index=tickers).fillna(0.0)
    weights = pd.Series(1.0, index=components.columns)
    for key in ("f_ROE", "f_NPM", "f_RevG"):
        if key in weights:
            weights[key] = 1.15
    score = components.mul(weights, axis=1).sum(axis=1) / weights.loc[components.columns].sum()
    return score.clip(-3, 3), components


def _model_scores(day: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    median = pd.to_numeric(day.get("pred_0.5", pd.Series(index=day.index)), errors="coerce")
    confidence = pd.to_numeric(day.get("confidence", pd.Series(index=day.index)), errors="coerce")
    return _zscore(median).reindex(day.index).fillna(0.0), confidence.clip(0.0, 1.0).fillna(0.0)


def _reason(technical: float, fundamental: float, model: float, risk: float) -> str:
    signs = [technical > 0, fundamental > 0, model > 0]
    if sum(signs) >= 2 and risk >= 0:
        return "Technical momentum, fundamental quality, and the model forecast agree with the selected risk budget."
    if sum(signs) >= 2:
        return "Technical, fundamental, and model evidence are positive, but trailing risk is the main constraint."
    if technical > 0 and fundamental < 0:
        return "Technical momentum is positive, while valuation or business-quality evidence needs confirmation."
    if fundamental > 0 and technical < 0:
        return "Fundamentals are supportive, while recent technical momentum is not yet confirming the thesis."
    return "The evidence is mixed; this name remains a watchlist candidate rather than a high-conviction pick."


def _flags(row: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    vol = _numeric(row.get("annual_volatility"))
    drawdown = _numeric(row.get("max_drawdown"))
    pe = _numeric(row.get("f_PE"))
    de = _numeric(row.get("f_DE"))
    if vol is not None and vol > 0.60:
        flags.append("high trailing volatility")
    if drawdown is not None and drawdown > 0.40:
        flags.append("deep trailing drawdown")
    if pe is not None and pe > 40:
        flags.append("elevated P/E")
    if de is not None and de > 150:
        flags.append("elevated debt/equity")
    if _numeric(row.get("confidence")) is not None and float(row["confidence"]) < 0.35:
        flags.append("wide model uncertainty")
    return flags


def _safe_hrp_weights(prices: Mapping[str, pd.DataFrame], tickers: list[str], asof: pd.Timestamp) -> np.ndarray:
    if len(tickers) <= 1:
        return np.ones(len(tickers), dtype=float)
    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        frame = prices.get(ticker)
        if frame is None or "Close" not in frame:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce")
        close.index = _date_index(close.to_frame())
        series[ticker] = close[close.index <= asof]
    if len(series) < len(tickers):
        return np.ones(len(tickers), dtype=float) / len(tickers)
    returns = pd.DataFrame(series).pct_change().dropna().tail(252)
    if len(returns) < 30:
        return np.ones(len(tickers), dtype=float) / len(tickers)
    cov = returns[tickers].cov().to_numpy() * 252.0
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2.0 + np.eye(len(tickers)) * 1e-8
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    try:
        tree = linkage(squareform(distance, checks=False), method="single")
        order = _quasi_diagonal_order(tree, len(tickers))
    except (FloatingPointError, IndexError, KeyError, TypeError, ValueError):
        return np.ones(len(tickers), dtype=float) / len(tickers)

    def cluster_variance(items: list[int]) -> float:
        block = cov[np.ix_(items, items)]
        inverse = 1.0 / np.maximum(np.diag(block), 1e-12)
        inverse /= inverse.sum()
        return float(inverse @ block @ inverse)

    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) > 1:
                midpoint = len(cluster) // 2
                left, right = cluster[:midpoint], cluster[midpoint:]
                left_var, right_var = cluster_variance(left), cluster_variance(right)
                alpha = 1.0 - left_var / max(left_var + right_var, 1e-12)
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
                next_clusters.extend([left, right])
        clusters = next_clusters
    return weights.reindex(range(len(tickers))).fillna(0.0).to_numpy()


def _quasi_diagonal_order(tree: np.ndarray, n: int) -> list[int]:
    children = tree[:, :2].astype(int)

    def expand(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = children[node - n]
        return expand(int(left)) + expand(int(right))

    return expand(2 * n - 2)


def _cap_weights(weights: np.ndarray, cap: float) -> np.ndarray:
    if len(weights) == 0:
        return weights
    cap = float(np.clip(cap, 0.01, 1.0))
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    weights = weights / weights.sum() if weights.sum() else np.ones(len(weights)) / len(weights)
    if cap >= 1.0:
        return weights
    result = np.zeros_like(weights)
    active = np.ones(len(weights), dtype=bool)
    remaining = 1.0
    while active.any() and remaining > 1e-10:
        proposed = np.zeros_like(weights)
        proposed[active] = weights[active] / weights[active].sum() * remaining
        over = active & (proposed > cap)
        if not over.any():
            result[active] = proposed[active]
            break
        result[over] = cap
        remaining -= cap * int(over.sum())
        active[over] = False
    return result


def _profile_from_input(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(profile or {})
    risk = str(supplied.get("risk", "Moderate"))
    if risk not in PROFILE_PRESETS:
        raise ValueError(f"risk must be one of: {', '.join(PROFILE_PRESETS)}")
    preset = PROFILE_PRESETS[risk]
    return {
        "risk": risk,
        "capital": max(float(supplied.get("capital", 100_000)), 1.0),
        "max_holdings": int(np.clip(supplied.get("max_holdings", 8), 1, 50)),
        "min_confidence": float(np.clip(supplied.get("min_confidence", preset.min_confidence), 0, 1)),
        "min_expected_return": float(supplied.get("min_expected_return", -1.0)),
        "max_annual_volatility": float(max(supplied.get("max_annual_volatility", preset.max_annual_volatility), 0.01)),
        "max_drawdown": float(np.clip(supplied.get("max_drawdown", preset.max_drawdown), 0.01, 1.0)),
        "max_position_weight": float(np.clip(supplied.get("max_position_weight", preset.max_position_weight), 0.01, 1.0)),
        "min_fundamental_score": float(np.clip(supplied.get("min_fundamental_score", preset.min_fundamental_score), 0, 100)),
        "exclude_tickers": [str(x).upper() for x in supplied.get("exclude_tickers", supplied.get("exclude", []))],
        "exclude_sectors": [str(x) for x in supplied.get("exclude_sectors", [])],
        "sector_preference": supplied.get("sector_preference", "Any"),
        "include_watchlist": bool(supplied.get("include_watchlist", True)),
    }


def profile_presets() -> list[dict[str, Any]]:
    return [asdict(preset) for preset in PROFILE_PRESETS.values()]


def build_recommendation_snapshot(
    bundle: Mapping[str, Any], profile: Mapping[str, Any] | None = None, asof: Any | None = None
) -> dict[str, Any]:
    """Return all analysis rows, selected positions, portfolio stats, and metadata."""

    res = bundle.get("res")
    if not isinstance(res, pd.DataFrame) or res.empty:
        raise ValueError("The artifact has no model result rows.")
    day, selected_date = _rows_on_date(res, asof)
    day = day.drop_duplicates("ticker").copy()
    day["ticker"] = day["ticker"].astype(str)
    day = day.set_index("ticker", drop=False)
    profile_data = _profile_from_input(profile)
    preset = PROFILE_PRESETS[profile_data["risk"]]
    prices = bundle.get("prices", {})
    funda = _fundamentals(bundle)
    metadata = _metadata(bundle)
    panel_day = _panel_day(bundle, selected_date)
    technical_raw, technical_components = _technical_scores(panel_day.reindex(day.index))
    fundamental_raw, fundamental_components = _fundamental_scores(funda, day.index)
    model_raw, confidence = _model_scores(day)
    risk_table = pd.DataFrame({ticker: _risk_stats(prices, ticker, selected_date) for ticker in day.index}).T
    volatility_z = _zscore(risk_table["annual_volatility"]) if "annual_volatility" in risk_table else pd.Series(0.0, index=day.index)
    drawdown_z = _zscore(risk_table["max_drawdown"]) if "max_drawdown" in risk_table else pd.Series(0.0, index=day.index)

    rows: list[dict[str, Any]] = []
    for ticker, source in day.iterrows():
        risk_stats = _risk_stats(prices, ticker, selected_date)
        fundamental_values = funda.loc[ticker].to_dict() if ticker in funda.index else {}
        metadata_values = metadata.loc[ticker].to_dict() if ticker in metadata.index else {}
        technical_values = {
            feature: _numeric(panel_day.loc[ticker].get(feature))
            for feature in TECHNICAL_FEATURES
            if ticker in panel_day.index and feature in panel_day.columns
        }
        risk_raw = -0.55 * _clip(volatility_z.get(ticker, 0.0))
        drawdown_raw = -0.45 * _clip(drawdown_z.get(ticker, 0.0))
        risk_score_raw = _clip(risk_raw + drawdown_raw)
        technical = float(technical_raw.get(ticker, 0.0))
        fundamental = float(fundamental_raw.get(ticker, 0.0))
        model = float(model_raw.get(ticker, 0.0))
        fit = float(risk_score_raw)
        composite_raw = (
            preset.technical_weight * technical
            + preset.fundamental_weight * fundamental
            + preset.model_weight * model
            + preset.risk_weight * fit
        )
        agreement = 1.0 if (technical > 0) == (fundamental > 0) else 0.5
        evidence_confidence = float(np.clip(0.60 * confidence.get(ticker, 0.0) + 0.25 * agreement + 0.15 * (1.0 if fundamental_values else 0.0), 0, 1))
        base_return = _numeric(source.get("pred_0.5"))
        low_return = _numeric(source.get("pred_0.1"))
        high_return = _numeric(source.get("pred_0.9"))
        original_signal = str(source.get("signal", "HOLD"))
        composite = _score_01(composite_raw)
        row = {
            "ticker": ticker,
            "model_signal": original_signal,
            "recommendation": "WATCH",
            "exp_ret_21d": base_return,
            "bear_return_21d": low_return,
            "bull_return_21d": high_return,
            "confidence": _numeric(confidence.get(ticker)),
            "evidence_confidence": evidence_confidence,
            "technical_score": _score_01(technical),
            "fundamental_score": _score_01(fundamental),
            "model_score": _score_01(model),
            "portfolio_fit_score": _score_01(fit),
            "overall_score": composite,
            "evidence_alignment": "aligned" if agreement == 1.0 else "mixed",
            "technical_evidence": technical_components.loc[ticker].abs().sort_values(ascending=False).head(4).index.tolist() if ticker in technical_components.index else [],
            "fundamental_evidence": fundamental_components.loc[ticker].abs().sort_values(ascending=False).head(4).index.tolist() if ticker in fundamental_components.index else [],
            "sector": metadata_values.get("sector", "Unknown"),
            "industry": metadata_values.get("industry", "Unknown"),
            "market_cap": _numeric(metadata_values.get("market_cap")),
            **risk_stats,
            **fundamental_values,
            **technical_values,
        }
        row["decision_reason"] = _reason(technical, fundamental, model, fit)
        row["risk_flags"] = _flags(row)
        model_positive = original_signal in BUY_SIGNALS or (base_return is not None and base_return >= 0)
        if model_positive and composite >= 70 and evidence_confidence >= preset.min_confidence:
            row["recommendation"] = "STRONG BUY"
        elif model_positive and composite >= 50 and evidence_confidence >= preset.min_confidence * 0.85:
            row["recommendation"] = "BUY"
        elif not model_positive or composite < 42:
            row["recommendation"] = "PASS"
        rows.append(row)

    analysis = pd.DataFrame(rows).set_index("ticker", drop=False)
    analysis["eligible"] = (
        analysis["recommendation"].isin(["STRONG BUY", "BUY"])
        & analysis["confidence"].fillna(0).ge(profile_data["min_confidence"])
        & analysis["evidence_confidence"].ge(profile_data["min_confidence"] * 0.75)
        & analysis["exp_ret_21d"].fillna(-np.inf).ge(profile_data["min_expected_return"])
        & analysis["annual_volatility"].fillna(np.inf).le(profile_data["max_annual_volatility"])
        & analysis["max_drawdown"].fillna(np.inf).le(profile_data["max_drawdown"])
        & analysis["fundamental_score"].ge(profile_data["min_fundamental_score"])
        & ~analysis["ticker"].isin(profile_data["exclude_tickers"])
        & ~analysis["sector"].isin(profile_data["exclude_sectors"])
    )
    if profile_data["sector_preference"] not in (None, "Any", ""):
        analysis["eligible"] &= analysis["sector"].eq(profile_data["sector_preference"])
    eligible = analysis[analysis["eligible"]].copy()
    eligible = eligible.sort_values(["overall_score", "evidence_confidence", "exp_ret_21d"], ascending=False)
    picks = eligible.head(profile_data["max_holdings"]).copy()
    if picks.empty and profile_data["include_watchlist"]:
        picks = analysis[analysis["recommendation"].eq("WATCH")].sort_values("overall_score", ascending=False).head(profile_data["max_holdings"]).copy()
        picks["is_watchlist"] = True
    else:
        picks["is_watchlist"] = False

    if not picks.empty:
        tickers = picks["ticker"].tolist()
        hrp = _safe_hrp_weights(prices, tickers, selected_date)
        tilt = np.clip(picks["overall_score"].to_numpy() / 100.0, 0.25, 1.25)
        intended = hrp * tilt
        intended /= intended.sum() if intended.sum() else 1.0
        target_weights = _cap_weights(intended, profile_data["max_position_weight"])
        picks["target_weight"] = target_weights
        picks["target_shares"] = [
            int(max(0, math.floor((weight * profile_data["capital"]) / price)))
            if _numeric(price) is not None and price > 0 else 0
            for weight, price in zip(target_weights, picks["price"])
        ]
        picks["target_notional"] = picks["target_shares"] * picks["price"].fillna(0.0)
        picks["realized_weight"] = picks["target_notional"] / profile_data["capital"]
    else:
        picks["target_weight"] = pd.Series(dtype=float)
        picks["target_shares"] = pd.Series(dtype=int)
        picks["target_notional"] = pd.Series(dtype=float)
        picks["realized_weight"] = pd.Series(dtype=float)

    summary = _portfolio_summary(picks, profile_data, prices, selected_date)
    quality = data_quality_report(bundle, selected_date)
    return {
        "asof": selected_date,
        "profile": profile_data,
        "analysis": analysis.reset_index(drop=True),
        "picks": picks.reset_index(drop=True),
        "summary": summary,
        "quality": quality,
    }


def _portfolio_summary(picks: pd.DataFrame, profile: Mapping[str, Any], prices: Mapping[str, Any], asof: pd.Timestamp) -> dict[str, Any]:
    if picks.empty:
        return {
            "positions": 0,
            "invested": 0.0,
            "cash": float(profile["capital"]),
            "cash_weight": 1.0,
            "expected_return_21d": None,
            "bear_return_21d": None,
            "bull_return_21d": None,
            "annual_volatility": None,
            "max_drawdown": None,
            "avg_confidence": None,
            "avg_technical_score": None,
            "avg_fundamental_score": None,
            "effective_positions": 0.0,
            "sector_count": 0,
        }
    weights = picks["realized_weight"].fillna(0).to_numpy()
    invested = float(picks["target_notional"].sum())
    cash = max(float(profile["capital"]) - invested, 0.0)
    sum_w = weights.sum()
    effective = float(1.0 / np.sum(np.square(weights))) if np.sum(np.square(weights)) else 0.0
    return {
        "positions": len(picks),
        "invested": invested,
        "cash": cash,
        "cash_weight": cash / profile["capital"],
        "expected_return_21d": _weighted(picks, "exp_ret_21d", weights, sum_w),
        "bear_return_21d": _weighted(picks, "bear_return_21d", weights, sum_w),
        "bull_return_21d": _weighted(picks, "bull_return_21d", weights, sum_w),
        "annual_volatility": _weighted(picks, "annual_volatility", weights, sum_w),
        "max_drawdown": _weighted(picks, "max_drawdown", weights, sum_w),
        "avg_confidence": _weighted(picks, "evidence_confidence", weights, sum_w),
        "avg_technical_score": _weighted(picks, "technical_score", weights, sum_w),
        "avg_fundamental_score": _weighted(picks, "fundamental_score", weights, sum_w),
        "effective_positions": effective,
        "sector_count": int(picks["sector"].replace("Unknown", np.nan).nunique(dropna=True)),
    }


def _weighted(frame: pd.DataFrame, column: str, weights: np.ndarray, total_weight: float) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any() or total_weight <= 0:
        return None
    return _numeric(np.sum(values[mask] * weights[mask]) / total_weight)


def execution_plan(snapshot: Mapping[str, Any], steps: int = 10) -> pd.DataFrame:
    """Create a displayable, integer-safe execution schedule for target shares."""

    picks = snapshot.get("picks", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if not isinstance(picks, pd.DataFrame):
        return pd.DataFrame(rows)
    for _, row in picks.iterrows():
        shares = int(max(row.get("target_shares", 0), 0))
        if shares <= 0:
            continue
        # A monotonically decreasing liquidation schedule, normalized to whole shares.
        raw = np.linspace(1.0, 0.35, max(steps, 2))
        slices = np.floor(raw / raw.sum() * shares).astype(int)
        slices[0] += shares - int(slices.sum())
        for number, quantity in enumerate(slices, 1):
            rows.append({
                "ticker": row["ticker"],
                "slice": number,
                "shares": int(quantity),
                "fraction_of_order": float(quantity / shares),
                "side": "BUY",
            })
    return pd.DataFrame(rows)


def validation_analysis(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate predictions on the held-out rows already present in the bundle."""

    res = bundle.get("res")
    if not isinstance(res, pd.DataFrame) or res.empty:
        return {"status": "unavailable", "reason": "No held-out result rows are present."}
    frame = res.copy()
    for col in ("fwd_ret", "pred_0.1", "pred_0.5", "pred_0.9"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["fwd_ret", "pred_0.5"])
    if len(frame) < 3:
        return {"status": "unavailable", "reason": "Fewer than three valid held-out observations."}
    errors = frame["pred_0.5"] - frame["fwd_ret"]
    directional = (np.sign(frame["pred_0.5"]) == np.sign(frame["fwd_ret"])).mean()
    ic_values: list[float] = []
    for _, group in frame.groupby(level=0):
        if len(group) >= 3 and group["pred_0.5"].nunique() > 1 and group["fwd_ret"].nunique() > 1:
            ic = spearmanr(group["pred_0.5"], group["fwd_ret"], nan_policy="omit").statistic
            if np.isfinite(ic):
                ic_values.append(float(ic))
    coverage = None
    if {"pred_0.1", "pred_0.9"}.issubset(frame.columns):
        interval = frame.dropna(subset=["pred_0.1", "pred_0.9"])
        coverage = float(((interval["fwd_ret"] >= interval["pred_0.1"]) & (interval["fwd_ret"] <= interval["pred_0.9"])).mean()) if len(interval) else None
    signal_table = pd.DataFrame()
    if "signal" in frame:
        signal_table = frame.groupby("signal").agg(
            observations=("fwd_ret", "size"),
            mean_forward_return=("fwd_ret", "mean"),
            median_forward_return=("fwd_ret", "median"),
            hit_rate=("fwd_ret", lambda x: float((x > 0).mean())),
        ).reset_index()
    return {
        "status": "available",
        "observations": len(frame),
        "date_start": _date_index(frame).min(),
        "date_end": _date_index(frame).max(),
        "mae": _numeric(errors.abs().mean()),
        "rmse": _numeric(np.sqrt(np.mean(np.square(errors)))),
        "directional_accuracy": _numeric(directional),
        "mean_information_coefficient": _numeric(np.mean(ic_values)) if ic_values else None,
        "quantile_coverage": coverage,
        "signal_performance": signal_table,
    }


def data_quality_report(bundle: Mapping[str, Any], asof: Any | None = None) -> dict[str, Any]:
    res = bundle.get("res")
    if not isinstance(res, pd.DataFrame) or res.empty:
        return {"status": "unavailable"}
    _, selected = _rows_on_date(res, asof)
    panel = bundle.get("panel_test")
    funda = _fundamentals(bundle)
    missing = {}
    if isinstance(panel, pd.DataFrame):
        missing = {str(k): float(v) for k, v in panel.isna().mean().sort_values(ascending=False).head(10).items()}
    return {
        "status": "available",
        "asof": selected,
        "assets": int(res["ticker"].nunique()) if "ticker" in res else 0,
        "held_out_rows": len(res),
        "selected_features": len(bundle.get("features", [])),
        "fundamental_snapshot_assets": len(funda),
        "fundamental_point_in_time": False,
        "missing_feature_rates": missing,
        "notes": [
            "Fundamentals are a current snapshot broadcast across historical rows in the original notebook.",
            "Recommendations are research outputs, not personalized financial advice or an order instruction.",
        ],
    }


def stock_detail(bundle: Mapping[str, Any], ticker: str, asof: Any | None = None) -> dict[str, Any]:
    snapshot = build_recommendation_snapshot(bundle, {"risk": "Moderate", "max_holdings": 50}, asof)
    ticker = str(ticker).upper()
    analysis = snapshot["analysis"]
    matches = analysis[analysis["ticker"].eq(ticker)]
    if matches.empty:
        raise KeyError(f"Ticker {ticker} is not present in the artifact universe.")
    row = matches.iloc[0].to_dict()
    frame = bundle["prices"].get(ticker)
    history = []
    if isinstance(frame, pd.DataFrame) and "Close" in frame:
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        close.index = _date_index(close.to_frame())
        for date, value in close[close.index <= snapshot["asof"]].tail(252).items():
            history.append({"date": pd.Timestamp(date), "close": _numeric(value)})
    shap_reasons = _shap_reasons(bundle, ticker, snapshot["asof"])
    fundamentals = {key: value for key, value in row.items() if str(key).startswith("f_")}
    return {
        "ticker": ticker,
        "asof": snapshot["asof"],
        "analysis": row,
        "fundamentals": fundamentals,
        "price_history": pd.DataFrame(history),
        "model_drivers": shap_reasons,
    }


def _shap_reasons(bundle: Mapping[str, Any], ticker: str, asof: pd.Timestamp, k: int = 8) -> list[dict[str, Any]]:
    try:
        import shap  # type: ignore

        panel = _panel_day(bundle, asof)
        features = list(bundle.get("features", []))
        models = bundle.get("models", {})
        model = models.get(0.5) if isinstance(models, Mapping) else None
        if ticker not in panel.index or model is None or not features:
            return []
        values = shap.TreeExplainer(model).shap_values(panel.loc[[ticker], features])
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values).reshape(-1)
        order = np.argsort(np.abs(values))[::-1][:k]
        return [{"feature": features[index], "impact": _numeric(values[index]), "direction": "positive" if values[index] >= 0 else "negative"} for index in order]
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return []


def jsonable(value: Any) -> Any:
    """Convert pandas/numpy objects for API responses without changing meaning."""

    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
