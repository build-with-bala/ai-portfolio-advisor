# AI Portfolio Advisor

A Python-only research product that turns the outputs of the accompanying
Colab model into profile-aware portfolio decisions. The product combines
technical evidence, fundamental evidence, quantile forecasts, uncertainty,
trailing risk, HRP allocation, staged execution, validation diagnostics, and
an explicit data-quality boundary.

## What the user gets

- A risk-profile intake for Conservative, Moderate, and Aggressive investors.
- Capital, holding-count, confidence, expected-return, volatility, drawdown,
  position-cap, sector, ticker, and fundamental-quality controls.
- A ranked recommendation book with BUY / STRONG BUY / WATCH / PASS decisions.
- Technical, fundamental, model, and portfolio-fit scores for every ticker.
- P10 / P50 / P90 scenarios, evidence alignment, risk flags, and decision
  reasons.
- HRP-based weights with a position cap, integer share sizing, cash remainder,
  sector exposure, and an illustrative staged execution schedule.
- Stock research pages with price history, fundamentals, pillar scores, and
  SHAP drivers when the serialized model supports them.
- Held-out validation: MAE, RMSE, directional accuracy, mean information
  coefficient, quantile coverage, and realized return by original signal.
- A Streamlit frontend and an optional FastAPI JSON API. No Django and no
  JavaScript frontend are required.

## Repository layout

```text
streamlit_app.py                 Python frontend
api_app.py                       Optional FastAPI service
src/portfolio_advisor/core.py    Shared analysis and allocation engine
notebooks/Financial_Analysis.ipynb  Steps 0–18 preserved, production handoff after 18
scripts/append_notebook_steps.py Notebook exporter builder
tests/                            Pure-engine and API tests
Dockerfile                        Streamlit / Coolify UI image
Dockerfile.api                    FastAPI image
docker-compose.yml                UI + API deployment
```

## Artifact handoff from Colab

The notebook is the training surface. Run the original research cells through
Step 18, then run the new Step 19 exporter. It writes:

- `artifacts.pkl` — model outputs, prices, models, features, fundamentals, and metadata.
- `metadata.csv` — optional sector, industry, market-cap, and additional fundamentals.
- `manifest.json` — export version, coverage, and known limitations.

Copy `artifacts.pkl` into this repository's `data/` directory. The app refuses
to invent recommendations when the artifact is absent or malformed; it shows
the exact setup state instead.

## Local run

Use Python 3.12 for the broadest compatibility with the pinned scientific
stack:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# after copying data/artifacts.pkl
streamlit run streamlit_app.py
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8501` for the UI or `http://localhost:8000/docs` for the
API documentation.

## API

- `GET /health` — process and artifact state.
- `GET /api/v1/profile-presets` — supported profile defaults.
- `GET /api/v1/universe` — available tickers and sectors.
- `POST /api/v1/recommendations` — profile-aware recommendation snapshot.
- `GET /api/v1/stock/{ticker}` — stock evidence and history.
- `GET /api/v1/validation` — held-out diagnostics from the artifact.
- `GET /api/v1/data-quality` — coverage and provenance warnings.

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H 'content-type: application/json' \
  -d '{
    "risk": "Moderate",
    "capital": 100000,
    "max_holdings": 6,
    "min_confidence": 0.4,
    "min_expected_return": 0.0,
    "max_annual_volatility": 0.6,
    "max_drawdown": 0.45,
    "max_position_weight": 0.3,
    "min_fundamental_score": 45,
    "include_watchlist": true
  }'
```

## Coolify deployment

### Recommended first deployment: UI only

Create a Coolify application from this repository, choose the Dockerfile
builder, expose port `8501`, and mount a persistent/read-only artifact path so
`/app/data/artifacts.pkl` exists. The health path is
`/_stcore/health`. Set the public domain only after the container reports
healthy.

### UI + API

Use the repository's Docker Compose deployment when two internal services are
desired. Publish the UI domain on port `8501`; keep the API private unless a
separate authentication layer is configured. The API health path is `/health`.

Do not commit `artifacts.pkl`, API keys, broker credentials, or Streamlit
secrets. The included `.gitignore` and `.dockerignore` intentionally exclude
the artifact and data files.

## Important model boundary

This product is an analytical research tool, not a broker and not a promise of
returns. The supplied notebook broadcasts a current yfinance fundamental
snapshot across historical rows. The frontend exposes that limitation and the
validation page does not label a backtest as live performance. A production
investment workflow should replace that snapshot with point-in-time
fundamentals, add transaction costs and turnover constraints, and require an
independent review before any capital is connected.
