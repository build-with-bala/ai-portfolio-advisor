# AI Portfolio Advisor

*Applied AI & Machine Learning Capstone | AI and ML for Digital Business Managers*

## 1. Problem and user

- Retail investors and small wealth advisers in India pick stocks from tips and single-number targets, with no view of the range of outcomes or of how a pick fits their risk tolerance.
- Target user: an independent wealth adviser or self-directed investor who reviews a portfolio monthly.
- AI Portfolio Advisor takes a risk profile and capital, and returns a ranked book of NSE stocks marked STRONG BUY / BUY / WATCH / PASS, each with a pessimistic, median and optimistic 21-day return, the evidence behind it, and a position size.
- Working prototype: https://fn.iimbg.com  |  Code: https://github.com/build-with-bala/ai-portfolio-advisor

## 2. Business value and ROI

- Time saved (assumptions, editable in scripts/build_submission.py): 200 clients x 4 reviews a year, screening time cut from 45 to 10 minutes, adviser time valued at Rs 1,500/hour.
- That is 467 adviser hours a year, worth Rs 700,000, against an assumed running cost of Rs 120,000 a year: a return of 483% on cost.
- Risk communication: 82.4% of real 21-day outcomes on unseen data fell inside the model's P10-P90 range, against a target of 80%. A client can be shown an honest range.
- What we do not claim: extra return. In the hold-out backtest the top-5 strategy returned -4.4% a year against 3.3% for holding all stocks equally (median over 21 start days, after costs). The product is sold on speed, discipline and honest risk ranges, not on beating the market.
- Market strategy: free research tool for self-directed investors; paid seat for advisers who need client-ready evidence sheets; brokers as a channel through the existing Upstox / Zerodha live-quote adapters.

## 3. System architecture

- src/data.py: download prices (Yahoo Finance), build 19 candidate features per stock per day, select features, define time-series splits.
- src/model.py: tune and cross-validate, train three LightGBM quantile models, evaluate on a hold-out period, compute SHAP, run a backtest, export data/artifacts.pkl and reports/metrics.json.
- src/portfolio_advisor/core.py: decision engine. Combines the technical score, fundamental score, model forecast and risk fit by profile weights, sizes positions with Hierarchical Risk Parity, a position cap and whole shares.
- app.py: Streamlit interface (profile intake, ranked book, stock research with SHAP, portfolio lab, validation, model report, live NSE quotes). api_app.py: optional FastAPI service.
- Flow: prices -> features -> selection -> purged walk-forward CV -> quantile models -> artifact -> app.

## 4. Data and preprocessing

- 60 liquid NSE large caps, five per sector across 12 sectors, daily adjusted prices 2018-02-19 to 2026-10-08: 118,641 stock-days.
- Label: return over the next 21 trading days.
- Features are returns, ratios and oscillators (RSI, MACD histogram scaled by price, Bollinger bandwidth, ATR, stochastic %K, OBV change, Williams %R, momentum, volatility, distance from 52-week high, volume surge, market-relative return, market return and volatility).
- Log price is fractionally differenced with the smallest order that passes the ADF test, chosen on development data only.
- StandardScaler inside the Ridge pipeline, refitted on each training fold. Tree models split on thresholds and are unaffected by feature scale, so LightGBM gets unscaled inputs.
- Rows missing any candidate feature (indicator warm-up) are dropped.

## 5. Feature selection

- Method: greedy mutual-information relevance minus 0.5 x redundancy, run on development data only.
- Kept 10 of 19: mkt_vol_21, fracdiff_close, rel_ret_21, volume_surge, mom_126_21, Stoch_K, vol_63, ret_63, OBV_delta, Boll_BW.
- Why: many indicators measure the same thing (RSI, stochastic %K and Williams %R all track short-term position in the range). The redundancy penalty keeps one of each kind.
- Company fundamentals were removed from the model. Only today's ratios are available, so using them on past dates would leak future information. They still feed the app's separate fundamental score.

## 6. Tuning and cross-validation

- Scheme: purged walk-forward cross-validation, 5 expanding folds. Each fold trains on the past and validates on the next block, with 31 trading days removed in between because each label looks 21 days ahead.
- Search: random search over 25 settings x 5 folds x 3 quantiles = 375 model fits. Tuning metric: mean pinball loss (the loss a quantile forecast is meant to minimise).
- Chosen settings: n_estimators=100, learning_rate=0.02, num_leaves=4, min_child_samples=50, subsample=0.8, colsample_bytree=0.5, reg_lambda=0.
- CV pinball loss: tuned 0.02406 +/- 0.00529; notebook settings 0.02648; naive baseline 0.02395.
- Tuning improved on the original notebook settings by 9.2%. The search picked very small trees (4 leaves, 100 trees): with a weak signal, complex trees fit noise.
- Against the naive no-feature baseline the tuned model is 0.5% worse in cross-validation, which is inside fold-to-fold variation. We report this as it is.

## 7. Hold-out results

- Hold-out period: 2025-01-23 to 2026-09-09, 24,413 stock-days never used for selection or tuning.
- Pinball loss: LightGBM 0.01880 vs naive 0.01915. MAE: LightGBM 0.0583, Ridge 0.0580, naive 0.0584.
- Rank information coefficient (daily rank correlation of forecast and outcome): LightGBM +0.046, positive on 56% of 407 days; Ridge +0.052.
- Direction right 51.0% of the time. P10-P90 coverage 82.4% against a 80% target.
- Backtest, hold-out: strategy -4.4%/yr vs benchmark 3.3%/yr; excess -6.4% (range -16.8% to 7.1%); beat the benchmark from 24% of start days.
- Backtest, cross-validation folds: excess 1.6%/yr (range -4.3% to 16.6%).

## 8. Explainability and error analysis

- SHAP (TreeExplainer on the median model). Largest drivers: mkt_vol_21, mom_126_21, fracdiff_close, vol_63.
- Every stock page in the app shows that stock's own SHAP drivers for the current forecast.
- Calm markets: MAE 0.0517, coverage 82.4%, rank IC -0.079.
- Normal markets: MAE 0.0552, coverage 85.0%, rank IC +0.076.
- Turbulent markets: MAE 0.0680, coverage 79.8%, rank IC +0.142.
- When the market fell: MAE 0.0661, direction right 31.1%.
- When the market rose: MAE 0.0529, direction right 64.6%.

## 9. Limits and next steps

- Returns over 21 days are mostly noise; the model explains a small part of the differences between stocks.
- Universe is 60 of today's large caps (survivorship bias) and one country.
- The backtest ignores taxes, market impact and slippage beyond the flat cost assumption.
- Fundamental scores in the app use today's ratios and are not part of the validated forecast.
- Next: point-in-time fundamentals, a market-relative target, a larger universe including delisted names, and a controlled paper-trading period.

## 10. Repository and reproducibility

- Layout: src/data.py, src/model.py, app.py, requirements.txt, tests/ (17 tests), reports/, docs/.
- Reproduce: pip install -r requirements.txt; python -m src.model (trains from the committed data snapshot in about a minute); streamlit run app.py.
- Vibe Coding Log: docs/VIBE_CODING_LOG.md.
