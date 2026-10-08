"""Model tuning, cross-validation, evaluation, explainability and export.

    python -m src.model            # train from the committed data snapshot
    python -m src.model --refresh  # download fresh data first

Outputs
    data/artifacts.pkl        bundle consumed by app.py and api_app.py
    data/manifest.json        what the bundle contains
    reports/metrics.json      every number in the app's "Model report" tab
    reports/cv_results.csv    one row per hyperparameter trial and fold
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import kurtosis, norm, skew, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from portfolio_advisor.core import make_pickle_portable

ARTIFACT_PATH = data.ROOT / "data" / "artifacts.pkl"
MANIFEST_PATH = data.ROOT / "data" / "manifest.json"
METRICS_PATH = data.ROOT / "reports" / "metrics.json"
CV_RESULTS_PATH = data.ROOT / "reports" / "cv_results.csv"
EXPORTER_VERSION = "2026-10-capstone-v3"

# The settings hard-coded in the original notebook, kept as trial 0 so the
# report can show what tuning changed.
NOTEBOOK_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0,
)
SEARCH_SPACE = dict(
    n_estimators=[100, 200, 400, 800],  # number of boosted trees
    learning_rate=[0.01, 0.02, 0.05],  # shrinkage applied to each tree
    num_leaves=[4, 8, 16, 31],  # tree complexity
    min_child_samples=[50, 200, 500, 1000],  # minimum rows per leaf
    subsample=[0.6, 0.8, 1.0],  # share of rows sampled per tree
    colsample_bytree=[0.5, 0.7, 1.0],  # share of features sampled per tree
    reg_lambda=[0.0, 1.0, 10.0, 50.0],  # L2 penalty on leaf values
)
N_TRIALS = 24
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


# ---------------------------------------------------------------------- metrics
def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Quantile loss: the score a q-quantile forecast is trained to minimise."""
    diff = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def daily_ic(frame: pd.DataFrame, pred: str = "pred_0.5") -> np.ndarray:
    """Spearman rank correlation between forecast and outcome, per day."""
    values = []
    for _, day in frame.groupby(level=0):
        if len(day) >= 5 and day[pred].nunique() > 1:
            ic = spearmanr(day[pred], day[data.LABEL]).statistic
            if np.isfinite(ic):
                values.append(ic)
    return np.asarray(values)


def score_forecasts(frame: pd.DataFrame, quantiles: list[float]) -> dict[str, float]:
    """All accuracy measures for a frame holding the label and `pred_q` columns."""
    y = frame[data.LABEL].to_numpy()
    out: dict[str, float] = {}
    if all(f"pred_{q}" in frame for q in quantiles):
        losses = [pinball_loss(y, frame[f"pred_{q}"], q) for q in quantiles]
        out["pinball"] = float(np.mean(losses))
        low, high = frame[f"pred_{quantiles[0]}"], frame[f"pred_{quantiles[-1]}"]
        out["coverage"] = float(((y >= low) & (y <= high)).mean())
        out["band_width"] = float((high - low).mean())
    median = frame["pred_0.5"].to_numpy()
    ic = daily_ic(frame)
    out.update(
        mae=float(np.mean(np.abs(y - median))),
        rmse=float(np.sqrt(np.mean((y - median) ** 2))),
        directional_accuracy=float((np.sign(median) == np.sign(y)).mean()),
        ic_mean=float(ic.mean()) if len(ic) else 0.0,
        ic_days=int(len(ic)),
        ic_positive_share=float((ic > 0).mean()) if len(ic) else 0.0,
    )
    return out


# ----------------------------------------------------------------------- models
def fit_quantile_models(train: pd.DataFrame, features: list[str], quantiles: list[float], params: dict, seed: int) -> dict[float, lgb.LGBMRegressor]:
    """One gradient-boosted model per quantile, each minimising pinball loss."""
    models = {}
    for q in quantiles:
        models[q] = lgb.LGBMRegressor(
            objective="quantile", alpha=q, subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1, **params
        ).fit(train[features], train[data.LABEL])
    return models


def predict_quantiles(models: dict[float, Any], frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Quantile forecasts per row, sorted so that P10 <= P50 <= P90 always holds."""
    raw = np.column_stack([models[q].predict(frame[features]) for q in sorted(models)])
    return pd.DataFrame(np.sort(raw, axis=1), index=frame.index, columns=[f"pred_{q}" for q in sorted(models)])


def with_forecasts(models: dict[float, Any], frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """`ticker`, label and forecasts side by side. Dates repeat once per
    stock, so columns are attached by position, never by an index join."""
    out = frame[["ticker", data.LABEL]].copy()
    forecasts = predict_quantiles(models, frame, features)
    for column in forecasts:
        out[column] = forecasts[column].to_numpy()
    return out


def _labelled(panel: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    return panel[panel.index.isin(dates) & panel[data.LABEL].notna()]


def _lgbm_fold(panel, features, cfg, params, trial, fold, train_dates, val_dates):
    train, val = _labelled(panel, train_dates), _labelled(panel, val_dates)
    models = fit_quantile_models(train, features, cfg["QUANTILES"], params, cfg["SEED"])
    scored = with_forecasts(models, val, features)
    row = {"model": "LightGBM quantile", "trial": trial, "fold": fold, "train_rows": len(train), "val_rows": len(val),
           "val_start": str(val.index.min().date()), "val_end": str(val.index.max().date()),
           **params, **score_forecasts(scored, cfg["QUANTILES"])}
    return row, scored


def tune_lightgbm(panel: pd.DataFrame, features: list[str], dev_dates: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Random search scored by purged walk-forward cross-validation."""
    trials = [NOTEBOOK_PARAMS] + list(ParameterSampler(SEARCH_SPACE, n_iter=N_TRIALS, random_state=cfg["SEED"]))
    folds = list(data.walk_forward_splits(dev_dates, cfg))
    jobs = [
        delayed(_lgbm_fold)(panel, features, cfg, dict(params), trial, fold, tr, va)
        for trial, params in enumerate(trials)
        for fold, (tr, va) in enumerate(folds, start=1)
    ]
    results = Parallel(n_jobs=-1)(jobs)
    rows = pd.DataFrame([row for row, _ in results])
    oof: dict[int, list[pd.DataFrame]] = {}
    for row, scored in results:
        oof.setdefault(row["trial"], []).append(scored)
    return rows, {trial: pd.concat(parts).sort_index() for trial, parts in oof.items()}


def cross_validate_baselines(panel: pd.DataFrame, features: list[str], dev_dates: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Two reference models on the same folds.

    * Naive: always predict the training-period quantiles (no features).
    * Ridge: linear regression on standardised features, L2 penalty `alpha`.
    """
    rows = []
    for fold, (train_dates, val_dates) in enumerate(data.walk_forward_splits(dev_dates, cfg), start=1):
        train, val = _labelled(panel, train_dates), _labelled(panel, val_dates)
        naive = val[["ticker", data.LABEL]].copy()
        for q in cfg["QUANTILES"]:
            naive[f"pred_{q}"] = train[data.LABEL].quantile(q)
        rows.append({"model": "Naive (historical quantiles)", "trial": 0, "fold": fold, **score_forecasts(naive, cfg["QUANTILES"])})
        for trial, alpha in enumerate(RIDGE_ALPHAS):
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train[features], train[data.LABEL])
            scored = val[["ticker", data.LABEL]].copy()
            scored["pred_0.5"] = ridge.predict(val[features])
            rows.append({"model": "Ridge (scaled features)", "trial": trial, "fold": fold, "alpha": alpha,
                         **score_forecasts(scored, cfg["QUANTILES"])})
    return pd.DataFrame(rows)


def summarise_trials(rows: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    agg = rows.groupby(by, dropna=False).agg(
        mean=(metric, "mean"), std=(metric, "std"), ic_mean=("ic_mean", "mean"), ic_std=("ic_mean", "std"),
        mae=("mae", "mean"), folds=("fold", "count"),
    )
    return agg.reset_index().sort_values("mean")


# ------------------------------------------------------------ signals + backtest
def add_confidence_and_signals(res: pd.DataFrame, buy_z: float = 0.3, strong_z: float = 1.0, conf_min: float = 0.5) -> pd.DataFrame:
    """Confidence = how narrow a stock's P10-P90 band is versus peers that
    day. Signals rank the day's median forecasts (cross-sectional z-score)."""
    out = res.copy()
    band = out["pred_0.9"] - out["pred_0.1"]
    out["confidence"] = 1 - band.groupby(level=0).rank(pct=True)
    grouped = out.groupby(level=0)["pred_0.5"]
    z = (out["pred_0.5"] - grouped.transform("mean")) / (grouped.transform("std") + 1e-9)
    signal = pd.Series("HOLD", index=out.index)
    signal[z > buy_z] = "BUY"
    signal[z < -buy_z] = "SELL"
    signal[(z > strong_z) & (out["confidence"] >= conf_min)] = "STRONG_BUY"
    signal[(z < -strong_z) & (out["confidence"] >= conf_min)] = "STRONG_SELL"
    out["signal"] = signal.to_numpy()
    return out


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """Probability that the true Sharpe ratio exceeds `benchmark_sr`,
    allowing for sample length, skew and fat tails (Bailey & López de Prado)."""
    r = np.asarray(returns)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    denom = np.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr**2)
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(len(r) - 1) / denom))


def backtest(res: pd.DataFrame, cfg: dict, top_n: int = 5, cost_bps: float = 20.0, offset: int = 0) -> dict[str, Any]:
    """Every `HORIZON` days buy the `top_n` stocks by median forecast, equal
    weight, and hold to the next rebalance. Benchmark: all stocks, equal weight.

    Costs: `cost_bps` per unit of traded value (one basis point = 0.01%).
    """
    labelled = res[res[data.LABEL].notna()]
    dates = np.sort(labelled.index.unique())[offset :: cfg["HORIZON"]]
    strategy, benchmark, held = [], [], set()
    for date in dates:
        day = labelled[labelled.index == date]
        if len(day) < top_n * 2:
            continue
        picks = day.nlargest(top_n, "pred_0.5")
        names = set(picks["ticker"])
        turnover = 2 * len(names - held) / top_n if held else 1.0  # sell old + buy new
        held = names
        strategy.append(picks[data.LABEL].mean() - turnover * cost_bps / 10_000)
        benchmark.append(day[data.LABEL].mean())
    strategy, benchmark = np.array(strategy), np.array(benchmark)
    if len(strategy) < 2:
        return {"periods": int(len(strategy))}
    per_year = 252 / cfg["HORIZON"]
    excess = strategy - benchmark

    def stats(r: np.ndarray) -> dict[str, float]:
        curve = np.cumprod(1 + r)
        peak = np.maximum.accumulate(curve)
        rf = cfg["RF_ANNUAL"] / per_year
        return {
            "total_return": float(curve[-1] - 1),
            "annual_return": float(curve[-1] ** (per_year / len(r)) - 1),
            "sharpe": float((r - rf).mean() / r.std(ddof=1) * np.sqrt(per_year)),
            "max_drawdown": float(((peak - curve) / peak).max()),
        }

    return {
        "periods": int(len(strategy)),
        "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()),
        "top_n": top_n,
        "cost_bps": cost_bps,
        "strategy": stats(strategy),
        "benchmark": stats(benchmark),
        "excess_per_period_mean": float(excess.mean()),
        "excess_t_stat": float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))),
        "periods_beating_benchmark": float((excess > 0).mean()),
        "prob_true_excess_positive": probabilistic_sharpe(excess),
        "curve": {"strategy": np.cumprod(1 + strategy).tolist(), "benchmark": np.cumprod(1 + benchmark).tolist()},
    }


def backtest_all_start_days(res: pd.DataFrame, cfg: dict, **kwargs) -> dict[str, Any]:
    """Repeat the backtest for every possible rebalance start day, so the
    result does not depend on which day the calendar happened to start."""
    runs = [backtest(res, cfg, offset=o, **kwargs) for o in range(cfg["HORIZON"])]
    runs = [r for r in runs if r.get("periods", 0) >= 2]
    gap = np.array([r["strategy"]["annual_return"] - r["benchmark"]["annual_return"] for r in runs])
    return {
        "runs": len(runs),
        "annual_excess_median": float(np.median(gap)),
        "annual_excess_min": float(gap.min()),
        "annual_excess_max": float(gap.max()),
        "share_of_runs_beating_benchmark": float((gap > 0).mean()),
        "strategy_annual_return_median": float(np.median([r["strategy"]["annual_return"] for r in runs])),
        "benchmark_annual_return_median": float(np.median([r["benchmark"]["annual_return"] for r in runs])),
        "strategy_max_drawdown_median": float(np.median([r["strategy"]["max_drawdown"] for r in runs])),
        "benchmark_max_drawdown_median": float(np.median([r["benchmark"]["max_drawdown"] for r in runs])),
        "first_run": runs[0],
    }


# -------------------------------------------------------- explain + error study
def shap_summary(model: lgb.LGBMRegressor, frame: pd.DataFrame, features: list[str], seed: int, rows: int = 5000) -> list[dict]:
    """Global SHAP importance of the median model on hold-out rows."""
    sample = frame[features].sample(min(rows, len(frame)), random_state=seed)
    values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
    out = []
    for i in np.argsort(-np.abs(values).mean(axis=0)):
        direction = spearmanr(sample.iloc[:, i], values[:, i]).statistic
        out.append({
            "feature": features[i],
            "mean_abs_shap": float(np.abs(values[:, i]).mean()),
            "direction": float(direction) if np.isfinite(direction) else 0.0,
        })
    return out


def error_analysis(scored: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> dict[str, list[dict]]:
    """Forecast quality sliced by market volatility, market direction and sector."""
    frame = scored.copy()
    frame["mkt_vol_21"] = frame.index.map(panel["mkt_vol_21"].groupby(level=0).first())
    frame["volatility_regime"] = pd.qcut(frame["mkt_vol_21"], 3, labels=["calm", "normal", "turbulent"])
    market_move = frame.groupby(level=0)[data.LABEL].transform("mean")
    frame["market_direction"] = np.where(market_move >= 0, "market rose", "market fell")
    frame["sector"] = frame["ticker"].map(data.SECTOR_OF).fillna("Other")

    def table(column: str) -> list[dict]:
        rows = []
        for level, part in frame.groupby(column, observed=True):
            rows.append({"group": str(level), "rows": int(len(part)), **score_forecasts(part, cfg["QUANTILES"])})
        return rows

    return {name: table(name) for name in ["volatility_regime", "market_direction", "sector"]}


# ------------------------------------------------------------------------ train
def train(refresh: bool = False, verbose: bool = True) -> dict[str, Any]:
    cfg = dict(data.CFG)
    started = time.perf_counter()
    if refresh:
        data.refresh_snapshot()
    prices, funda, metadata = data.load_snapshot()

    all_dates = np.sort(pd.concat(prices.values()).index.unique())
    dev_end = data.holdout_dates(all_dates, cfg)[0][-1]
    panel = data.build_panel(prices, cfg, fit_until=dev_end)
    dates = np.sort(panel.index.unique())
    dev_dates, test_dates = data.holdout_dates(dates, cfg)
    dev = _labelled(panel, dev_dates)
    test = panel[panel.index.isin(test_dates)]  # includes the newest, still unlabelled days

    # Feature selection sees development data only.
    sample = dev.sample(min(len(dev), 20_000), random_state=cfg["SEED"])
    features, selection = data.select_features(sample[data.CANDIDATE_FEATURES], sample[data.LABEL], cfg["N_FEATURES"], seed=cfg["SEED"])
    if verbose:
        print(f"Panel {panel.shape} | dev {len(dev):,} rows to {dev.index.max().date()} | hold-out {len(test):,} rows from {test.index.min().date()}")
        print("Selected features:", features)

    lgbm_rows, oof = tune_lightgbm(panel, features, dev_dates, cfg)
    base_rows = cross_validate_baselines(panel, features, dev_dates, cfg)
    cv_rows = pd.concat([lgbm_rows, base_rows], ignore_index=True)

    lgbm_summary = summarise_trials(lgbm_rows, ["trial"], "pinball")
    best_trial = int(lgbm_summary.iloc[0]["trial"])
    best_params = {k: _plain(lgbm_rows.loc[lgbm_rows["trial"] == best_trial, k].iloc[0]) for k in SEARCH_SPACE}
    ridge_summary = summarise_trials(base_rows[base_rows["model"].str.startswith("Ridge")], ["alpha"], "mae")
    best_alpha = float(ridge_summary.iloc[0]["alpha"])

    def cv_line(rows: pd.DataFrame, name: str, settings: str) -> dict:
        return {
            "model": name,
            "settings": settings,
            "pinball_mean": _mean(rows, "pinball"), "pinball_std": _std(rows, "pinball"),
            "mae_mean": _mean(rows, "mae"), "mae_std": _std(rows, "mae"),
            "ic_mean": _mean(rows, "ic_mean"), "ic_std": _std(rows, "ic_mean"),
            "coverage_mean": _mean(rows, "coverage"),
            "directional_accuracy_mean": _mean(rows, "directional_accuracy"),
            "folds": rows[["fold", "pinball", "mae", "ic_mean"]].replace({np.nan: None}).to_dict("records"),
        }

    candidates = [
        cv_line(base_rows[base_rows["model"].str.startswith("Naive")], "Naive (historical quantiles)", "no features"),
        cv_line(base_rows[base_rows["alpha"] == best_alpha], "Ridge (scaled features)", f"alpha={best_alpha:g}"),
        cv_line(lgbm_rows[lgbm_rows["trial"] == 0], "LightGBM quantile, notebook settings", _fmt(NOTEBOOK_PARAMS)),
        cv_line(lgbm_rows[lgbm_rows["trial"] == best_trial], "LightGBM quantile, tuned", _fmt(best_params)),
    ]

    # Final models: refit on all development data with the chosen settings.
    models = fit_quantile_models(dev, features, cfg["QUANTILES"], best_params, cfg["SEED"])
    res = add_confidence_and_signals(with_forecasts(models, test, features))
    has_label = res[data.LABEL].notna().to_numpy()
    scored, scored_inputs = res[has_label], test[has_label]

    naive_test = scored[["ticker", data.LABEL]].copy()
    for q in cfg["QUANTILES"]:
        naive_test[f"pred_{q}"] = dev[data.LABEL].quantile(q)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(dev[features], dev[data.LABEL])
    ridge_test = scored[["ticker", data.LABEL]].assign(**{"pred_0.5": ridge.predict(scored_inputs[features])})

    signal_table = (
        scored.groupby("signal")[data.LABEL]
        .agg(rows="size", mean_forward_return="mean", share_positive=lambda x: float((x > 0).mean()))
        .reset_index().to_dict("records")
    )
    oof_best = add_confidence_and_signals(oof[best_trial])

    report = {
        "product": "AI Portfolio Advisor",
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data": {
            "market": cfg["MARKET"],
            "source": "Yahoo Finance daily adjusted prices (yfinance)",
            "tickers": len(prices),
            "sectors": len(set(data.SECTOR_OF[t] for t in prices if t in data.SECTOR_OF)),
            "first_date": str(dates[0].astype("datetime64[D]")),
            "last_date": str(dates[-1].astype("datetime64[D]")),
            "panel_rows": int(len(panel)),
            "development_rows": int(len(dev)),
            "holdout_rows_scored": int(len(scored)),
            "holdout_rows_latest_unlabelled": int(len(res) - len(scored)),
            "development_end": str(dev.index.max().date()),
            "holdout_start": str(test.index.min().date()),
            "holdout_end_scored": str(scored.index.max().date()),
            "survivorship_note": "The universe is today's large caps, so companies that failed or shrank earlier are absent.",
        },
        "config": {k: cfg[k] for k in ["HORIZON", "EMBARGO", "TEST_FRAC", "N_SPLITS", "QUANTILES", "N_FEATURES", "SEED"]},
        "preprocessing": {
            "features_are_ratios": "All inputs are returns, ratios or oscillators, so no price-level scaling is needed.",
            "scaling": "StandardScaler inside the Ridge pipeline, refitted on each training fold. "
                       "Tree models split on thresholds and are unaffected by feature scale, so LightGBM gets unscaled inputs.",
            "stationarity": "Log price is fractionally differenced with the smallest order that passes the ADF test, "
                            "chosen on development data only.",
            "missing_values": "Rows missing any candidate feature (indicator warm-up) are dropped.",
            "leakage_controls": [
                f"Labels look {cfg['HORIZON']} trading days ahead; {data.purge_gap(cfg)} trading days are removed before every validation block.",
                "Feature selection, differencing order and hyperparameters are chosen on development data only.",
                "Fundamental ratios are excluded from the model because only today's snapshot exists.",
            ],
        },
        "feature_selection": {
            "method": "greedy mutual-information relevance minus 0.5 x redundancy",
            "candidates": data.CANDIDATE_FEATURES,
            "selected": features,
            "table": selection.round(5).to_dict("records"),
        },
        "validation": {
            "scheme": f"purged walk-forward cross-validation, {cfg['N_SPLITS']} expanding folds",
            "tuning_metric": "mean pinball loss over the three quantiles",
            "folds": lgbm_rows[lgbm_rows["trial"] == best_trial][["fold", "train_rows", "val_rows", "val_start", "val_end"]].to_dict("records"),
            "lightgbm_trials": int(lgbm_rows["trial"].nunique()),
            "lightgbm_fits": int(len(lgbm_rows) * len(cfg["QUANTILES"])),
            "search_space": SEARCH_SPACE,
        },
        "candidates": candidates,
        "selected": {
            "model": "LightGBM quantile regression (three models: P10, P50, P90)",
            "params": best_params,
            "notebook_params": NOTEBOOK_PARAMS,
            "pinball_improvement_vs_notebook": float(1 - candidates[3]["pinball_mean"] / candidates[2]["pinball_mean"]),
            "pinball_improvement_vs_naive": float(1 - candidates[3]["pinball_mean"] / candidates[0]["pinball_mean"]),
        },
        "trials": lgbm_summary.merge(
            lgbm_rows.drop_duplicates("trial")[["trial", *SEARCH_SPACE]], on="trial"
        ).round(6).to_dict("records"),
        "holdout": {
            "lightgbm": score_forecasts(scored, cfg["QUANTILES"]),
            "naive": score_forecasts(naive_test, cfg["QUANTILES"]),
            "ridge": score_forecasts(ridge_test, cfg["QUANTILES"]),
            "target_coverage": cfg["QUANTILES"][-1] - cfg["QUANTILES"][0],
            "signals": signal_table,
        },
        "backtest": {
            "rule": f"every {cfg['HORIZON']} trading days hold the 5 highest median forecasts, equal weight; "
                    "benchmark = all stocks equal weight; 20 bps cost on traded value",
            "holdout": backtest_all_start_days(res, cfg),
            "cross_validation_folds": backtest_all_start_days(oof_best, cfg),
        },
        "shap": shap_summary(models[0.5], scored_inputs, features, cfg["SEED"]),
        "error_analysis": error_analysis(scored, panel, cfg),
        "limits": [
            "Returns over 21 days are mostly noise; the model explains a small part of the differences between stocks.",
            "Universe is 60 of today's large caps (survivorship bias) and one country.",
            "The backtest ignores taxes, market impact and slippage beyond the flat cost assumption.",
            "Fundamental scores in the app use today's ratios and are not part of the validated forecast.",
        ],
    }

    funda_export = funda.copy()
    for column in ["forwardPE", "trailingEps", "earningsGrowth", "grossMargins", "operatingMargins", "currentRatio", "quickRatio", "beta"]:
        if column in metadata:
            funda_export[f"f_{column}"] = metadata[column]
    funda_export["f_market_cap"] = metadata.get("market_cap", np.nan)

    bundle = make_pickle_portable({
        "res": res,
        "prices": prices,
        "models": models,
        "panel_test": test,
        "features": features,
        "cfg": {**cfg, "UNIVERSE": sorted(prices), "SECTOR_UNIVERSE": data.SECTOR_UNIVERSE},
        "funda": funda_export,
        "metadata": metadata,
        "report": report,
        "exporter_version": EXPORTER_VERSION,
        "market": cfg["MARKET"],
    })
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    with ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    MANIFEST_PATH.write_text(json.dumps({
        "exporter_version": EXPORTER_VERSION,
        "market": cfg["MARKET"],
        "assets": sorted(prices),
        "selected_features": features,
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "fundamentals_used_as_model_inputs": False,
        "notes": [
            "Fundamentals are today's snapshot and feed only the app's fundamental score.",
            "Research and education only; no trades are placed.",
        ],
    }, indent=2))
    METRICS_PATH.write_text(json.dumps(report, indent=2, default=_plain))
    cv_rows.to_csv(CV_RESULTS_PATH, index=False)

    if verbose:
        for c in candidates:
            print(f"  {c['model']:40s} pinball {_show(c['pinball_mean'])}  MAE {c['mae_mean']:.4f}  IC {c['ic_mean']:+.4f} ± {c['ic_std']:.4f}")
        h = report["holdout"]["lightgbm"]
        print(f"Hold-out: pinball {h['pinball']:.5f} | MAE {h['mae']:.4f} | IC {h['ic_mean']:+.4f} | coverage {h['coverage']:.3f} | dir {h['directional_accuracy']:.3f}")
        b = report["backtest"]["holdout"]
        print(f"Backtest hold-out: annual excess median {b['annual_excess_median']:+.3f} (min {b['annual_excess_min']:+.3f}, max {b['annual_excess_max']:+.3f})")
        print(f"Best params: {best_params}  | {time.perf_counter() - started:.0f}s")
    return report


def load_report() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text())


def _mean(rows: pd.DataFrame, col: str) -> float | None:
    return float(rows[col].mean()) if col in rows and rows[col].notna().any() else None


def _std(rows: pd.DataFrame, col: str) -> float | None:
    return float(rows[col].std()) if col in rows and rows[col].notna().any() else None


def _show(value: float | None) -> str:
    return "   n/a " if value is None else f"{value:.5f}"


def _fmt(params: dict) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in params.items())


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download fresh data before training")
    train(refresh=parser.parse_args().refresh)
