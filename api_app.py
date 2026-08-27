"""FastAPI surface for the portfolio advisor.

Run with: uvicorn api_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from portfolio_advisor.core import (
    build_recommendation_snapshot,
    data_quality_report,
    execution_plan,
    jsonable,
    load_bundle,
    profile_presets,
    stock_detail,
    validation_analysis,
)


class ProfileRequest(BaseModel):
    risk: Literal["Conservative", "Moderate", "Aggressive"] = "Moderate"
    capital: float = Field(default=100_000, gt=0, le=1_000_000_000)
    max_holdings: int = Field(default=8, ge=1, le=50)
    min_confidence: float = Field(default=0.40, ge=0, le=1)
    min_expected_return: float = Field(default=-1.0, ge=-1, le=1)
    max_annual_volatility: float = Field(default=0.60, gt=0, le=5)
    max_drawdown: float = Field(default=0.45, gt=0, le=1)
    max_position_weight: float = Field(default=0.30, gt=0, le=1)
    min_fundamental_score: float = Field(default=45, ge=0, le=100)
    exclude_tickers: list[str] = Field(default_factory=list)
    exclude_sectors: list[str] = Field(default_factory=list)
    sector_preference: str = "Any"
    include_watchlist: bool = True
    asof: date | None = None


app = FastAPI(
    title="AI Portfolio Advisor API",
    version="0.1.0",
    description="Profile-aware technical and fundamental portfolio analysis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()


def _require_bundle() -> dict:
    try:
        return get_bundle()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
        res = bundle["res"]
        return {
            "status": "ok",
            "artifacts_loaded": True,
            "assets": int(res["ticker"].nunique()) if "ticker" in res else 0,
            "asof": pd.Timestamp(res.index.max()).date().isoformat(),
        }
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        return {"status": "degraded", "artifacts_loaded": False, "detail": str(exc)}


@app.get("/api/v1/profile-presets")
def get_profile_presets() -> list[dict]:
    return profile_presets()


@app.get("/api/v1/universe")
def universe() -> dict:
    bundle = _require_bundle()
    res = bundle["res"]
    metadata = bundle.get("metadata")
    sectors = []
    if isinstance(metadata, pd.DataFrame) and "sector" in metadata:
        sectors = sorted(str(item) for item in metadata["sector"].dropna().unique())
    return {"tickers": sorted(str(item) for item in res["ticker"].unique()), "sectors": sectors}


@app.post("/api/v1/recommendations")
def recommendations(request: ProfileRequest) -> dict:
    bundle = _require_bundle()
    payload = request.model_dump()
    snapshot = build_recommendation_snapshot(bundle, payload, request.asof)
    snapshot["execution_plan"] = execution_plan(snapshot)
    return jsonable(snapshot)


@app.get("/api/v1/stock/{ticker}")
def stock(ticker: str, asof: date | None = None) -> dict:
    bundle = _require_bundle()
    try:
        return jsonable(stock_detail(bundle, ticker, asof))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/validation")
def validation() -> dict:
    return jsonable(validation_analysis(_require_bundle()))


@app.get("/api/v1/data-quality")
def data_quality() -> dict:
    return jsonable(data_quality_report(_require_bundle()))
