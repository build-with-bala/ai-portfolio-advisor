"""Append the production handoff to the existing notebook.

The first 19 cells are copied without modification. This is intentionally a
small, auditable transformation because the research notebook is user-owned.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("/Users/root1/Downloads/Financial_Analysis.ipynb")
TARGET = Path(__file__).resolve().parents[1] / "notebooks" / "Financial_Analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    original = json.loads(SOURCE.read_text())
    if len(original.get("cells", [])) < 19:
        raise ValueError("The source notebook does not contain Steps 0-18.")
    cells = list(original["cells"][:19])

    cells.extend(
        [
            markdown(
                """## Step 19 — Re-train the production universe for India

Steps 0–18 are preserved exactly as supplied. This production handoff retrains the same validated pipeline on an Indian-market universe using Yahoo Finance NSE symbols (`.NS`). It uses the technical, fundamental, quantile-forecast, risk, and portfolio logic already defined above; it does not silently mix the original US sample with the India artifact.

Run this cell after Steps 0–18. Change `INDIA_CFG["UNIVERSE"]` if you want a different NSE universe. For exchange-grade intraday quotes, the companion app uses a broker WebSocket adapter; this notebook remains the daily research/training layer.
"""
            ),
            code(
                '''# Production universe: liquid NSE large-/mid-cap examples. Replace with your approved universe.
INDIA_CFG = dict(CFG)
INDIA_CFG.update({
    "UNIVERSE": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "TCS.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
        "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "AXISBANK.NS", "KOTAKBANK.NS", "TATAMOTORS.NS", "HCLTECH.NS",
        "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS", "ASIANPAINT.NS",
        "ULTRACEMCO.NS", "WIPRO.NS",
    ],
    "START": "2017-01-01",
    "END": pd.Timestamp.today().strftime("%Y-%m-%d"),
})

india_prices = load_prices(INDIA_CFG["UNIVERSE"], INDIA_CFG["START"], INDIA_CFG["END"])
if len(india_prices) < 5:
    raise RuntimeError("Fewer than five India tickers returned data; check symbols or the data provider.")

india_funda = pd.DataFrame({t: fundamental_snapshot(t) for t in india_prices}).T
india_funda = india_funda.apply(pd.to_numeric, errors="coerce")
india_funda = india_funda.fillna(india_funda.median())
india_funda.columns = ["f_PE", "f_PB", "f_ROE", "f_DE", "f_NPM", "f_DivYld", "f_RevG"]

india_panel = build_panel(india_prices, india_funda, INDIA_CFG)
INDIA_TOPK = isf_mid(
    india_panel.drop(columns=["fwd_ret", "ticker"]),
    india_panel["fwd_ret"],
    k=12,
)
india_train, india_test = purged_split(india_panel, INDIA_CFG)
india_models, india_preds = train_quantile_models(
    india_train, india_test, INDIA_TOPK, INDIA_CFG["QUANTILES"]
)
india_res = india_test[["ticker", "fwd_ret"]].copy()
for q in INDIA_CFG["QUANTILES"]:
    india_res[f"pred_{q}"] = india_preds[q]
india_band = india_res["pred_0.9"] - india_res["pred_0.1"]
india_res["confidence"] = (
    india_res.assign(_b=india_band)
    .groupby(level=0)["_b"]
    .transform(lambda band: 1 - band.rank(pct=True))
)
india_res = assign_signals_cross_sectional(india_res)
print(f"India training complete: {len(india_prices)} assets | {len(india_res):,} held-out rows")
print(
    "Latest India signals:",
    india_res.groupby(level=0)["signal"].last().value_counts().to_dict(),
)
'''
            ),
            markdown(
                """## Step 20 — Export the production artifact

This is the notebook step that packages the India outputs for the standalone product. It does not alter the original research cells. It adds a metadata snapshot so the product can display sector, industry, market-cap, and extra fundamental context.
"""
            ),
            code(
                '''from pathlib import Path
import json
import pickle

# Prefer Drive so the artifact survives a Colab runtime reset.
ARTIFACT_DIR = (
    Path("/content/drive/MyDrive/AI_Portfolio_Advisor")
    if Path("/content/drive/MyDrive").exists()
    else Path("/content/ai_portfolio_advisor")
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

META_KEYS = [
    "sector", "industry", "marketCap", "forwardPE", "trailingEps",
    "earningsGrowth", "grossMargins", "operatingMargins", "currentRatio",
    "quickRatio", "beta", "recommendationKey",
]


def enriched_metadata(tickers):
    rows = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows[ticker] = {key: info.get(key, np.nan) for key in META_KEYS}
        except Exception as exc:
            print(f"metadata unavailable for {ticker}: {exc}")
            rows[ticker] = {key: np.nan for key in META_KEYS}
    meta = pd.DataFrame(rows).T
    return meta.rename(columns={"marketCap": "market_cap"})


metadata = enriched_metadata(sorted(india_prices))
for col in metadata.columns:
    if col not in ("sector", "industry", "recommendationKey"):
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

# Keep the original seven ratios used by the model and add display-only fields.
funda_export = india_funda.copy()
for col in [
    "forwardPE", "trailingEps", "earningsGrowth", "grossMargins",
    "operatingMargins", "currentRatio", "quickRatio", "beta",
]:
    if col in metadata:
        funda_export["f_" + col] = metadata[col]
funda_export["f_market_cap"] = metadata.get("market_cap", np.nan)

bundle = dict(
    res=india_res,
    prices=india_prices,
    models=india_models,
    panel_test=india_test,
    features=INDIA_TOPK,
    cfg=INDIA_CFG,
    funda=funda_export,
    metadata=metadata,
    exporter_version="2026-08-27-india-production-v1",
    market="NSE India",
)
with open(ARTIFACT_DIR / "artifacts.pkl", "wb") as handle:
    pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
metadata.to_csv(ARTIFACT_DIR / "metadata.csv")
manifest = {
    "exporter_version": bundle["exporter_version"],
    "market": "NSE India",
    "assets": sorted(india_prices),
    "selected_features": INDIA_TOPK,
    "test_start": str(india_test.index.min()),
    "test_end": str(india_test.index.max()),
    "fundamental_point_in_time": False,
    "notes": [
        "Fundamentals come from the current yfinance snapshot and are not historical point-in-time data.",
        "The artifact is for research and educational use; it does not place trades.",
    ],
}
(ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"Saved production artifact to {ARTIFACT_DIR / 'artifacts.pkl'}")
print(f"Saved metadata to {ARTIFACT_DIR / 'metadata.csv'}")
print(
    f"Assets: {len(india_prices)} | held-out rows: {len(india_res):,} | "
    f"selected features: {len(INDIA_TOPK)}"
)
'''
            ),
            markdown(
                """## Step 21 — Validate the handoff

This checks that the artifact has the fields required by the standalone Python product. It intentionally does not claim the model is profitable; it checks packaging and coverage only.
"""
            ),
            code(
                '''with open(ARTIFACT_DIR / "artifacts.pkl", "rb") as handle:
    production_bundle = pickle.load(handle)

required = {"res", "prices", "models", "panel_test", "features", "funda", "metadata"}
missing = required.difference(production_bundle)
assert not missing, f"Missing artifact keys: {sorted(missing)}"
assert len(production_bundle["res"]) > 0
assert len(production_bundle["features"]) > 0
print("Artifact validation passed.")
print("Copy artifacts.pkl, metadata.csv, and manifest.json into the app project's data/ directory.")
'''
            ),
            markdown(
                """## Step 22 — Run the product

The production frontend and API live in the companion repository. In Colab, copy the exported `artifacts.pkl` into that repository's `data/` directory. For a persistent service, deploy the repository with its Dockerfile or Docker Compose configuration in Coolify.

The frontend is Python-only Streamlit. The optional API is Python-only FastAPI. No Django or JavaScript application is required.
"""
            ),
            code(
                '''print("Local UI: streamlit run streamlit_app.py")
print("Local API: uvicorn api_app:app --host 0.0.0.0 --port 8000")
print("Docker UI: docker build -t ai-portfolio-advisor . && docker run -p 8501:8501 -v $PWD/data:/app/data:ro ai-portfolio-advisor")
print("Docker Compose: docker compose up --build")
'''
            ),
        ]
    )

    output = dict(original)
    output["cells"] = cells
    TARGET.write_text(json.dumps(output, indent=1) + "\n")
    print(f"Wrote {TARGET} with {len(cells)} cells")


if __name__ == "__main__":
    main()
