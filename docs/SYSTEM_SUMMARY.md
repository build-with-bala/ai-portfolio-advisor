# AI-Powered Portfolio Optimization and Indian-Market Recommendation System

## 1. Purpose

This project is an analytical decision system for an investor who wants stock
recommendations matched to a personal risk profile. It does not connect to a
broker, place orders, or claim that a model forecast is a guaranteed return.
The application makes the reasoning inspectable: every candidate is scored
through technical evidence, fundamental evidence, model output, uncertainty,
trailing risk, and portfolio fit.

The deliverable has two modes. The Colab notebook is the research and training
surface. The deployed application is the decision surface: it consumes the
serialized artifact produced by the notebook and provides a Streamlit UI plus
an optional FastAPI API.

## 2. End-to-end flow

```text
Colab / research
  OHLCV + fundamentals
        -> technical indicators + fractional differentiation
        -> redundancy-aware feature selection
        -> purged / embargoed train-test split
        -> LightGBM P10 / P50 / P90 forecasts
        -> model signal + uncertainty
        -> artifact exporter
               |
               v
Coolify / fn.iimbg.com
  profile intake
        -> technical pillar
        -> fundamental pillar
        -> model forecast pillar
        -> risk and portfolio-fit gates
        -> HRP weights + position cap + cash remainder
        -> recommendation book + evidence ledger
        -> optional Indian live quote overlay
```

## 3. Notebook contract

Steps 0–18 remain the original research pipeline. The added steps after Step
18 do not rewrite those cells. They run a separate India configuration using
the same functions already defined in the notebook, train the quantile models
on Indian NSE symbols, enrich the artifact with metadata, and validate the
handoff.

The India training universe is intentionally a configurable list of liquid
large-cap NSE symbols such as `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`,
`ICICIBANK.NS`, `ITC.NS`, `SBIN.NS`, `LT.NS`, `BHARTIARTL.NS`, `SUNPHARMA.NS`,
`MARUTI.NS`, `TATAMOTORS.NS`, `HINDUNILVR.NS`, `AXISBANK.NS`, `KOTAKBANK.NS`,
`NTPC.NS`, `POWERGRID.NS`, and `ASIANPAINT.NS`. The list is a starting point,
not a promise that every symbol will be returned by the data provider.

The exporter writes `artifacts.pkl`, `metadata.csv`, and `manifest.json`. The
manifest records whether fundamentals are point-in-time. In the supplied
notebook they are not: the current yfinance snapshot is broadcast through the
historical panel. That is a known backtest limitation, and the UI displays it
instead of hiding it.

## 4. ML and decision analysis

### Technical pillar

The model input contains RSI, MACD histogram, Bollinger bandwidth, ATR,
stochastic %K, OBV change, Williams %R, 5-day and 21-day returns, rolling
volatility, and fractionally differentiated close. The application converts
the current cross-section into a directional technical score. Momentum and
MACD receive a modest emphasis, while volatility and ATR act as risk-aware
penalties.

### Fundamental pillar

The current notebook ratios are P/E, P/B, ROE, debt/equity, net margin,
dividend yield, and revenue growth. Lower positive P/E, P/B, and leverage are
treated as better relative evidence; higher ROE, net margin, dividend yield,
and revenue growth are treated as better relative evidence. The score is
cross-sectional and is shown on a 0–100 scale. Additional metadata such as
sector, industry, market cap, forward P/E, EPS, margins, liquidity ratios, beta,
and the provider recommendation key is displayed when available.

### Forecast and uncertainty pillar

The notebook's three LightGBM quantile models produce a lower forecast (P10),
median forecast (P50), and upper forecast (P90) for the configured horizon.
The median is the base scenario. The interval is visible as uncertainty; the
application does not turn a narrow interval into a claim of certainty.

### Portfolio-fit pillar

Trailing annual volatility, maximum drawdown, daily VaR/CVaR, momentum, HRP
correlation structure, position caps, integer share sizing, and uninvested cash
are calculated after the evidence scores. HRP is used for the initial risk
allocation, then tilted by composite evidence and capped by the profile. The
cash remainder is shown explicitly when integer share sizing or caps prevent
full deployment.

## 5. Profile logic

| Profile | What changes |
| --- | --- |
| Conservative | Higher evidence threshold, lower volatility/drawdown gates, stronger fundamental weight, lower position cap, broader diversification |
| Moderate | Balanced technical, fundamental, model, and risk weights |
| Aggressive | Higher technical/model emphasis, wider volatility/drawdown limits, higher position cap, more tolerance for mixed evidence |

Every profile can additionally specify capital, maximum holdings, minimum
forecast confidence, minimum expected return, maximum annual volatility,
maximum drawdown, maximum position weight, minimum fundamental score, sector
preference, excluded sectors, excluded tickers, and whether to show a
watchlist when no name clears all gates.

The recommendation states are `STRONG BUY`, `BUY`, `WATCH`, and `PASS`. A
watchlist row is deliberately visible when the filters are restrictive. This
prevents the product from silently turning “no suitable stock” into a false
recommendation.

## 6. Indian real-time analyzer

The real-time surface is an overlay, not an invisible retraining step. The
daily ML forecast and evidence scores remain labelled as daily model outputs.
The live quote layer updates live price, previous-close return, last tick time,
live technical confirmation, and live score. A BUY can be downgraded to a
watch state when the live technical confirmation is materially weak; the user
can see that the change came from the live overlay.

The code supports three provider states:

- `LIVE_PROVIDER=zerodha` — exchange WebSocket adapter using Kite Connect;
  requires a daily access token and an NSE instrument-token mapping.
- `LIVE_PROVIDER=yahoo` — public polling fallback; it is explicitly labelled
  delayed or availability-limited and is not exchange realtime.
- `LIVE_PROVIDER=paper` — safe default with no quote connection.

For a production Indian realtime deployment, use a licensed broker/data feed.
The application must not receive credentials from the browser. Put the
provider credentials in Coolify environment variables or secrets, restart the
service after the daily access-token rotation, and verify the UI's provider,
quote-count, and last-tick status before interpreting a live table.

## 7. Frontend surfaces

The Streamlit UI contains:

1. **Overview** — decision table, CSV download, recommendation cards, near
   misses, and the full technical/fundamental evidence ledger.
2. **Stock research** — price history, four evidence pillars, fundamental
   snapshot, risk flags, and per-stock SHAP drivers when available.
3. **Portfolio lab** — capital map, scenario range, sector exposure, HRP-sized
   positions, cash, and staged execution schedule.
4. **Validation** — held-out MAE/RMSE, directional accuracy, information
   coefficient, quantile coverage, and realized performance by original signal.
5. **India live** — provider state, session state, quote receipt, live overlay,
   and clear feed-quality messaging.
6. **Method & data** — pipeline explanation, feature coverage, artifact date,
   and the point-in-time fundamentals warning.

## 8. Deployment at `fn.iimbg.com`

The repository contains a Streamlit Dockerfile for the public UI and a second
FastAPI Dockerfile for the optional API. In Coolify:

1. Create a new application from the GitHub repository.
2. Choose the UI Dockerfile and expose port `8501`.
3. Attach a persistent/private data mount at `/app/data` and place the
   notebook-generated `artifacts.pkl` there.
4. Set `ARTIFACTS_PATH=/app/data/artifacts.pkl`.
5. For live quotes, set `LIVE_PROVIDER=zerodha` and provide secrets without
   committing them.
6. Configure the FQDN as `https://fn.iimbg.com`; Coolify should terminate TLS
   and route to port 8501.
7. Confirm `/_stcore/health`, the artifact date, and the India live feed state.

The optional FastAPI container listens on port 8000 and exposes `/health`,
`/api/v1/recommendations`, `/api/v1/stock/{ticker}`, `/api/v1/validation`,
`/api/v1/live/status`, and `/api/v1/live/analyzer`. Keep it private until
authentication, rate limiting, request auditing, and a strict CORS allowlist
are configured.

## 9. What is not yet claimed

The product is not a registered investment adviser, broker, exchange feed
licence, or automatic trading system. It does not prove future performance. A
serious production release still needs point-in-time fundamentals, survivorship
and delisting treatment, Indian transaction costs and taxes, corporate-action
handling, paper-trading soak tests, provider failover, authentication, and an
independent review of the decision policy.
