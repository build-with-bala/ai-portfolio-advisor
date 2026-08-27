"""Streamlit frontend for the AI Portfolio Advisor."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from portfolio_advisor.core import (
    build_recommendation_snapshot,
    data_quality_report,
    execution_plan,
    load_bundle,
    profile_presets,
    stock_detail,
    validation_analysis,
)
from portfolio_advisor.live import LiveMarketService

st.set_page_config(
    page_title="Signal Desk | AI Portfolio Advisor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "bg": "#071018",
    "panel": "#0c1822",
    "panel_2": "#101f2b",
    "line": "#203442",
    "text": "#edf5f2",
    "muted": "#8fa7ab",
    "mint": "#79f2c0",
    "amber": "#ffb86b",
    "red": "#ff7474",
    "blue": "#72b7ff",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: #071018; color: #edf5f2; }
        [data-testid="stSidebar"] { background: #0a151e; border-right: 1px solid #203442; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
        .brand { letter-spacing: .18em; color: #79f2c0; font: 500 11px 'DM Mono', monospace; text-transform: uppercase; }
        .hero { padding: 1rem 0 1.4rem; border-bottom: 1px solid #203442; margin-bottom: 1.2rem; }
        .eyebrow { color: #79f2c0; font: 500 11px 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
        .hero h1 { font-size: clamp(2rem, 4vw, 4.2rem); line-height: .98; letter-spacing: -.06em; margin: .45rem 0; }
        .hero p { color: #8fa7ab; max-width: 760px; font-size: 1.02rem; }
        .section-label { color: #8fa7ab; font: 500 11px 'DM Mono', monospace; letter-spacing: .14em; text-transform: uppercase; margin: 1.2rem 0 .5rem; }
        .metric-card { background: linear-gradient(135deg, #0e1b25, #0b1720); border: 1px solid #203442; border-radius: 14px; padding: 14px 16px; min-height: 104px; }
        .metric-label { color: #8fa7ab; font: 500 10px 'DM Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
        .metric-value { color: #edf5f2; font-size: 1.55rem; font-weight: 700; margin-top: 7px; }
        .metric-note { color: #79f2c0; font-size: .78rem; margin-top: 4px; }
        .signal-card { background: #0c1822; border: 1px solid #203442; border-radius: 16px; padding: 18px; min-height: 215px; }
        .signal-ticker { color: #edf5f2; font-size: 1.5rem; font-weight: 800; letter-spacing: -.04em; }
        .signal-buy { color: #79f2c0; font: 500 11px 'DM Mono', monospace; letter-spacing: .12em; }
        .signal-watch { color: #ffb86b; font: 500 11px 'DM Mono', monospace; letter-spacing: .12em; }
        .signal-pass { color: #ff7474; font: 500 11px 'DM Mono', monospace; letter-spacing: .12em; }
        .small-copy { color: #8fa7ab; font-size: .83rem; line-height: 1.5; }
        .evidence { border-left: 2px solid #79f2c0; padding-left: 10px; color: #edf5f2; font-size: .87rem; line-height: 1.45; }
        .warning-box { background: rgba(255,184,107,.09); border: 1px solid rgba(255,184,107,.35); padding: 14px; border-radius: 12px; color: #ffdfb5; }
        .stButton > button[kind="primary"] { background: #79f2c0; color: #071018; border: 0; font-weight: 800; }
        div[data-testid="stDataFrame"] { border: 1px solid #203442; border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "—"
        return f"{number * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_number(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "—"
        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 12, "r": 12, "t": 38, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["muted"], "family": "Manrope"},
        title_font={"color": PALETTE["text"], "size": 14},
        xaxis={"gridcolor": PALETTE["line"], "zerolinecolor": PALETTE["line"]},
        yaxis={"gridcolor": PALETTE["line"], "zerolinecolor": PALETTE["line"]},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return fig


def indicator_frame(frame: pd.DataFrame | None, lookback: int = 252) -> pd.DataFrame:
    """Build the visual indicator tape used by the research and live panels."""

    if not isinstance(frame, pd.DataFrame) or "Close" not in frame:
        return pd.DataFrame()
    data = frame.copy().tail(lookback)
    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data.get("High", close), errors="coerce")
    low = pd.to_numeric(data.get("Low", close), errors="coerce")
    volume = pd.to_numeric(data.get("Volume", pd.Series(index=data.index)), errors="coerce")
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    mid = sma20
    sd = close.rolling(20).std()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False).mean() / close
    result = pd.DataFrame(index=data.index)
    result["Open"] = pd.to_numeric(data.get("Open", close), errors="coerce")
    result["High"] = high
    result["Low"] = low
    result["Close"] = close
    result["Volume"] = volume
    result["SMA20"] = sma20
    result["SMA50"] = sma50
    result["BB_upper"] = mid + 2 * sd
    result["BB_lower"] = mid - 2 * sd
    result["RSI14"] = rsi
    result["MACD"] = macd
    result["MACD_signal"] = macd.ewm(span=9, adjust=False).mean()
    result["ATR14_pct"] = atr
    result["Boll_BW"] = (4 * sd / mid).replace([np.inf, -np.inf], np.nan)
    result["ret_5"] = close.pct_change(5)
    result["ret_21"] = close.pct_change(21)
    result["vol_21"] = close.pct_change().rolling(21).std() * np.sqrt(252)
    return result.dropna(subset=["Close"])


@st.cache_resource(show_spinner=False)
def cached_bundle(path: str) -> dict:
    return load_bundle(path)


@st.cache_resource(show_spinner=False)
def cached_live_service(path: str) -> LiveMarketService:
    return LiveMarketService(load_bundle(path))


def sidebar_profile(bundle: dict) -> dict:
    metadata = bundle.get("metadata")
    sectors = []
    if isinstance(metadata, pd.DataFrame) and "sector" in metadata:
        sectors = sorted(str(item) for item in metadata["sector"].dropna().unique())
    tickers = sorted(str(item) for item in bundle["res"]["ticker"].unique())
    preset_map = {item["name"]: item for item in profile_presets()}
    with st.sidebar:
        st.markdown('<div class="brand">Signal desk / 01</div>', unsafe_allow_html=True)
        st.caption("Profile-aware research terminal")
        st.markdown("### Investor profile")
        with st.form("profile_form"):
            risk = st.selectbox("Risk posture", list(preset_map), index=1, key="profile_risk")
            preset = preset_map[risk]
            st.caption(preset["description"])
            capital = st.number_input("Investable capital", min_value=1000.0, value=100000.0, step=5000.0, key="profile_capital")
            max_holdings = st.slider("Maximum holdings", 1, min(20, max(len(tickers), 1)), min(8, max(len(tickers), 1)), key="profile_max_holdings")
            min_confidence = st.slider("Minimum forecast confidence", 0.0, 1.0, float(preset["min_confidence"]), 0.05, key="profile_min_confidence")
            min_expected_return = st.slider("Minimum expected 21-day return", -10.0, 25.0, 0.0, 0.5, key="profile_min_return") / 100
            max_volatility = st.slider("Maximum annual volatility", 10.0, 150.0, float(preset["max_annual_volatility"] * 100), 5.0, key="profile_max_volatility") / 100
            max_drawdown = st.slider("Maximum trailing drawdown", 10.0, 90.0, float(preset["max_drawdown"] * 100), 5.0, key="profile_max_drawdown") / 100
            max_position = st.slider("Maximum position weight", 5.0, 100.0, float(preset["max_position_weight"] * 100), 5.0, key="profile_max_position") / 100
            min_fundamental = st.slider("Minimum fundamental score", 0, 100, int(preset["min_fundamental_score"]), 5, key="profile_min_fundamental")
            sector_preference = st.selectbox("Sector preference", ["Any", *sectors] if sectors else ["Any"], key="profile_sector")
            exclude_sectors = st.multiselect("Exclude sectors", sectors, key="profile_exclude_sectors")
            exclude_tickers = st.multiselect("Exclude tickers", tickers, key="profile_exclude_tickers")
            include_watchlist = st.checkbox("Show watchlist when no stock clears all gates", True, key="profile_watchlist")
            submitted = st.form_submit_button("Run profile analysis", type="primary", use_container_width=True)
        if submitted or "profile" not in st.session_state:
            st.session_state["profile"] = {
                "risk": risk,
                "capital": capital,
                "max_holdings": max_holdings,
                "min_confidence": min_confidence,
                "min_expected_return": min_expected_return,
                "max_annual_volatility": max_volatility,
                "max_drawdown": max_drawdown,
                "max_position_weight": max_position,
                "min_fundamental_score": min_fundamental,
                "sector_preference": sector_preference,
                "exclude_sectors": exclude_sectors,
                "exclude_tickers": exclude_tickers,
                "include_watchlist": include_watchlist,
            }
        st.divider()
        st.caption("Research output only. No broker connection or automatic order execution is enabled.")
    return st.session_state["profile"]


def render_header(snapshot: dict) -> None:
    profile = snapshot["profile"]
    st.markdown(
        f'<div class="hero"><div class="eyebrow">AI-powered portfolio optimization / {profile["risk"]} posture</div>'
        f'<h1>Make the thesis<br>visible.</h1>'
        f'<p>Technical momentum, fundamental quality, model uncertainty, and portfolio fit in one auditable decision surface. Analysis date: <b>{snapshot["asof"].date()}</b>.</p></div>',
        unsafe_allow_html=True,
    )


def render_metrics(snapshot: dict) -> None:
    summary = snapshot["summary"]
    cols = st.columns(6)
    values = [
        ("Positions", str(summary["positions"]), f"of {snapshot['profile']['max_holdings']} max"),
        ("Base case / 21d", fmt_pct(summary["expected_return_21d"]), "model median"),
        ("Bear case / 21d", fmt_pct(summary["bear_return_21d"]), "10th percentile"),
        ("Annual volatility", fmt_pct(summary["annual_volatility"]), "trailing estimate"),
        ("Capital in market", fmt_pct(1 - summary["cash_weight"]), fmt_number(summary["invested"], 0)),
        ("Evidence quality", fmt_number(summary["avg_fundamental_score"], 0) + "/100", "fundamental pillar"),
    ]
    for column, (label, value, note) in zip(cols, values):
        with column:
            metric_card(label, value, note)


def render_insights(snapshot: dict) -> None:
    """Explain what the current profile and evidence imply in plain language."""

    analysis = snapshot["analysis"]
    picks = snapshot["picks"]
    summary = snapshot["summary"]
    st.markdown('<div class="section-label">00 / Automated insights</div>', unsafe_allow_html=True)
    if analysis.empty:
        st.info("No analysis rows are available for the selected artifact.")
        return
    buy_count = int(analysis["recommendation"].isin(["BUY", "STRONG BUY"]).sum())
    pass_count = int(analysis["recommendation"].eq("PASS").sum())
    notes: list[str] = [
        f"The model currently finds **{buy_count} actionable names** and **{pass_count} passes** across the loaded universe.",
    ]
    if picks.empty:
        notes.append("No stock cleared every profile gate. The correct action for this profile is to stay in cash or relax one constraint deliberately.")
    else:
        top = picks.iloc[0]
        notes.append(
            f"The highest-ranked fit is **{top['ticker']}** ({top['recommendation']}), with a {fmt_pct(top.get('exp_ret_21d'))} base-case 21-day forecast and {fmt_number(top.get('evidence_confidence'), 2)} evidence confidence."
        )
        if top.get("technical_score", 0) >= 60 and top.get("fundamental_score", 0) >= 60:
            notes.append("Technical momentum and fundamental quality agree on the leading pick; this is the strongest form of confirmation in the current framework.")
        elif top.get("technical_score", 0) >= 60:
            notes.append("The leading pick is technically strong but needs fundamental confirmation; treat it as a momentum-led thesis.")
        elif top.get("fundamental_score", 0) >= 60:
            notes.append("The leading pick is fundamentally strong but technical confirmation is weaker; staged entry is more appropriate than chasing price.")
        else:
            notes.append("The leading pick has mixed pillar scores; keep the position capped and review the evidence ledger before acting.")
    risk_note = f"Portfolio fit: {summary['sector_count']} sectors, {fmt_pct(summary.get('cash_weight'))} cash, and {fmt_pct(summary.get('annual_volatility'))} estimated annual volatility."
    notes.append(risk_note)
    left, right = st.columns([1.4, 1])
    with left:
        for note in notes:
            st.markdown(f'<div class="evidence" style="margin:8px 0">{note}</div>', unsafe_allow_html=True)
    with right:
        breadth = pd.DataFrame({"Decision": ["BUY", "PASS", "Other"], "Names": [buy_count, pass_count, max(len(analysis) - buy_count - pass_count, 0)]})
        fig = go.Figure(go.Bar(x=breadth["Decision"], y=breadth["Names"], marker_color=[PALETTE["mint"], PALETTE["red"], PALETTE["amber"]], text=breadth["Names"], textposition="outside"))
        fig.update_layout(title="Signal breadth", yaxis_title="Assets")
        st.plotly_chart(chart_layout(fig, 240), use_container_width=True)


def table_for(frame: pd.DataFrame, include_risk: bool = True) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = {
        "ticker": "Ticker",
        "recommendation": "Decision",
        "overall_score": "Overall",
        "technical_score": "Technical",
        "fundamental_score": "Fundamental",
        "model_score": "Model",
        "exp_ret_21d": "Base 21d",
        "bear_return_21d": "Bear 21d",
        "bull_return_21d": "Bull 21d",
        "evidence_confidence": "Evidence conf.",
        "annual_volatility": "Volatility",
        "target_weight": "Target wt.",
        "target_shares": "Shares",
        "price": "Price",
        "sector": "Sector",
    }
    selected = [column for column in columns if column in frame]
    result = frame[selected].rename(columns={column: columns[column] for column in selected}).copy()
    for column in ["Base 21d", "Bear 21d", "Bull 21d", "Volatility", "Target wt."]:
        if column in result:
            result[column] = result[column].map(fmt_pct)
    for column in ["Overall", "Technical", "Fundamental", "Model", "Evidence conf."]:
        if column in result:
            result[column] = result[column].map(lambda value, label=column: fmt_number(value, 0) if label != "Evidence conf." else fmt_number(value, 2))
    if "Price" in result:
        result["Price"] = result["Price"].map(lambda value: fmt_number(value, 2))
    if "Shares" in result:
        result["Shares"] = result["Shares"].fillna(0).astype(int)
    return result


def render_recommendations(snapshot: dict) -> None:
    picks = snapshot["picks"]
    analysis = snapshot["analysis"]
    st.markdown('<div class="section-label">01 / Recommendation book</div>', unsafe_allow_html=True)
    if picks.empty:
        st.markdown('<div class="warning-box">No position cleared the selected profile gates. The watchlist and near-miss table below show what needs confirmation.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(table_for(picks), use_container_width=True, hide_index=True)
        csv = picks.to_csv(index=False).encode("utf-8")
        st.download_button("Download recommendation book (.csv)", csv, "recommendations.csv", "text/csv")

        st.markdown('<div class="section-label">Decision cards</div>', unsafe_allow_html=True)
        for start in range(0, len(picks), 3):
            columns = st.columns(3)
            for column, (_, row) in zip(columns, picks.iloc[start:start + 3].iterrows()):
                decision = row["recommendation"]
                cls = "signal-buy" if "BUY" in decision else "signal-watch" if "WATCH" in decision else "signal-pass"
                with column:
                    st.markdown(
                        f'<div class="signal-card"><div class="{cls}">{decision}</div>'
                        f'<div class="signal-ticker">{row["ticker"]}</div>'
                        f'<div class="small-copy">{row.get("sector", "Unknown")} · {row.get("industry", "Unknown")}</div>'
                        f'<hr style="border-color:#203442">'
                        f'<div class="small-copy">Base case <b style="color:#79f2c0">{fmt_pct(row.get("exp_ret_21d"))}</b> · confidence {fmt_number(row.get("evidence_confidence"), 2)}</div>'
                        f'<div class="small-copy">Technical {fmt_number(row.get("technical_score"), 0)}/100 · Fundamental {fmt_number(row.get("fundamental_score"), 0)}/100</div>'
                        f'<div class="evidence" style="margin-top:12px">{row.get("decision_reason", "")}</div></div>',
                        unsafe_allow_html=True,
                    )

    near = analysis[~analysis["ticker"].isin(picks["ticker"] if not picks.empty else [])].sort_values("overall_score", ascending=False).head(12)
    if not near.empty:
        with st.expander("Near misses and watchlist", expanded=picks.empty):
            st.caption("These names are visible so the absence of a recommendation is inspectable, not silently hidden.")
            st.dataframe(table_for(near), use_container_width=True, hide_index=True)


def render_evidence(snapshot: dict, bundle: dict) -> None:
    picks = snapshot["picks"]
    st.markdown('<div class="section-label">02 / Evidence ledger</div>', unsafe_allow_html=True)
    if picks.empty:
        st.info("Run a less restrictive profile to inspect full evidence cards, or use the Research tab for any ticker in the universe.")
        return
    for _, row in picks.iterrows():
        with st.expander(f"{row['ticker']} · {row['recommendation']} · {row['decision_reason']}"):
            left, right = st.columns(2)
            with left:
                st.markdown("#### Technical evidence")
                technical = row.get("technical_evidence", [])
                if technical:
                    tech_frame = pd.DataFrame({"Indicator": technical, "Observed value": [row.get(item) for item in technical]})
                    st.dataframe(tech_frame, hide_index=True, use_container_width=True)
                else:
                    st.caption("Technical feature values are unavailable for this row.")
                st.metric("Technical score", f"{fmt_number(row.get('technical_score'), 0)}/100")
            with right:
                st.markdown("#### Fundamental evidence")
                fundamental = row.get("fundamental_evidence", [])
                if fundamental:
                    fund_frame = pd.DataFrame({"Ratio": fundamental, "Observed value": [row.get(item) for item in fundamental]})
                    st.dataframe(fund_frame, hide_index=True, use_container_width=True)
                else:
                    st.caption("Fundamental fields are unavailable for this row.")
                st.metric("Fundamental score", f"{fmt_number(row.get('fundamental_score'), 0)}/100")
            flags = row.get("risk_flags", [])
            if flags:
                st.warning("Risk flags: " + "; ".join(flags))
            else:
                st.success("No configured risk flag fired for this name.")


def render_portfolio_lab(snapshot: dict) -> None:
    st.markdown('<div class="section-label">03 / Portfolio lab</div>', unsafe_allow_html=True)
    picks = snapshot["picks"]
    summary = snapshot["summary"]
    if picks.empty:
        st.info("Portfolio charts appear when at least one position clears the profile gates.")
        return
    left, right = st.columns([1.1, 1])
    with left:
        labels = picks["ticker"].tolist() + (["Cash"] if summary["cash"] > 0 else [])
        values = picks["realized_weight"].fillna(0).tolist() + ([summary["cash_weight"]] if summary["cash"] > 0 else [])
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker={"colors": [PALETTE["mint"], PALETTE["blue"], PALETTE["amber"], PALETTE["red"]]}))
        fig.update_layout(title="Capital map", showlegend=True)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    with right:
        scenarios = pd.DataFrame({"Scenario": ["Bear / P10", "Base / P50", "Bull / P90"], "Return": [summary["bear_return_21d"], summary["expected_return_21d"], summary["bull_return_21d"]]})
        fig = go.Figure(go.Bar(x=scenarios["Scenario"], y=scenarios["Return"], marker_color=[PALETTE["red"], PALETTE["mint"], PALETTE["blue"]], text=[fmt_pct(x) for x in scenarios["Return"]], textposition="outside"))
        fig.update_layout(title="Portfolio scenario range", yaxis_tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    sector = picks.groupby("sector", dropna=False)["realized_weight"].sum().sort_values(ascending=True).reset_index()
    sector["sector"] = sector["sector"].fillna("Unknown")
    fig = go.Figure(go.Bar(y=sector["sector"], x=sector["realized_weight"], orientation="h", marker_color=PALETTE["blue"], text=[fmt_pct(x) for x in sector["realized_weight"]], textposition="outside"))
    fig.update_layout(title="Sector exposure", xaxis_tickformat=".0%")
    st.plotly_chart(chart_layout(fig, max(220, 60 + len(sector) * 38)), use_container_width=True)

    schedule = execution_plan(snapshot)
    if not schedule.empty:
        st.markdown("#### Execution schedule")
        st.caption("Illustrative staged orders; this does not connect to a broker or place trades.")
        st.dataframe(schedule, use_container_width=True, hide_index=True)


def render_research(bundle: dict, snapshot: dict) -> None:
    st.markdown('<div class="section-label">04 / Stock research</div>', unsafe_allow_html=True)
    tickers = sorted(str(item) for item in bundle["res"]["ticker"].unique())
    selected = st.selectbox("Choose a ticker", tickers)
    try:
        detail = stock_detail(bundle, selected, snapshot["asof"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        st.error(str(exc))
        return
    row = detail["analysis"]
    cols = st.columns(5)
    for column, (label, value) in zip(cols, [
        ("Decision", str(row.get("recommendation", "—"))),
        ("Overall", f"{fmt_number(row.get('overall_score'), 0)}/100"),
        ("Base / 21d", fmt_pct(row.get("exp_ret_21d"))),
        ("Annual vol", fmt_pct(row.get("annual_volatility"))),
        ("Max drawdown", fmt_pct(row.get("max_drawdown"))),
    ]):
        with column:
            metric_card(label, value)
    left, right = st.columns([1.5, 1])
    with left:
        indicators = indicator_frame(bundle["prices"].get(selected))
        if not indicators.empty:
            dates = indicators.index
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.48, 0.14, 0.19, 0.19], subplot_titles=(f"{selected} price + trend bands", "Volume", "RSI (14)", "MACD"))
            fig.add_trace(go.Candlestick(x=dates, open=indicators["Open"], high=indicators["High"], low=indicators["Low"], close=indicators["Close"], name="OHLC"), row=1, col=1)
            for column, color in [("SMA20", PALETTE["amber"]), ("SMA50", PALETTE["blue"]), ("BB_upper", PALETTE["muted"]), ("BB_lower", PALETTE["muted"])]:
                fig.add_trace(go.Scatter(x=dates, y=indicators[column], mode="lines", name=column, line={"color": color, "width": 1.2, "dash": "dot" if column.startswith("BB") else "solid"}), row=1, col=1)
            fig.add_trace(go.Bar(x=dates, y=indicators["Volume"], name="Volume", marker_color=PALETTE["blue"], opacity=0.55), row=2, col=1)
            fig.add_trace(go.Scatter(x=dates, y=indicators["RSI14"], mode="lines", name="RSI14", line={"color": PALETTE["mint"]}), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color=PALETTE["red"], row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=PALETTE["blue"], row=3, col=1)
            fig.add_trace(go.Scatter(x=dates, y=indicators["MACD"], mode="lines", name="MACD", line={"color": PALETTE["amber"]}), row=4, col=1)
            fig.add_trace(go.Scatter(x=dates, y=indicators["MACD_signal"], mode="lines", name="Signal", line={"color": PALETTE["blue"]}), row=4, col=1)
            fig.update_layout(height=760, xaxis_rangeslider_visible=False, showlegend=True)
            fig.update_yaxes(gridcolor=PALETTE["line"])
            st.plotly_chart(chart_layout(fig, 760), use_container_width=True)
            latest = indicators.iloc[-1]
            rsi_value = float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else None
            trend = "above both SMA20 and SMA50" if latest["Close"] > latest["SMA20"] and latest["Close"] > latest["SMA50"] else "below both SMA20 and SMA50" if latest["Close"] < latest["SMA20"] and latest["Close"] < latest["SMA50"] else "between the short- and medium-term averages"
            momentum = "overbought" if rsi_value is not None and rsi_value >= 70 else "oversold" if rsi_value is not None and rsi_value <= 30 else "in a neutral RSI zone"
            macd_state = "positive MACD crossover" if latest["MACD"] > latest["MACD_signal"] else "negative MACD crossover"
            st.markdown(f"**Technical insight:** Price is {trend}; RSI is {momentum} ({fmt_number(rsi_value, 1)}), with a {macd_state}. The 21-day realized volatility is {fmt_pct(latest['vol_21'])}.")
    with right:
        score_frame = pd.DataFrame({"Pillar": ["Technical", "Fundamental", "Model", "Portfolio fit"], "Score": [row.get("technical_score"), row.get("fundamental_score"), row.get("model_score"), row.get("portfolio_fit_score")]})
        fig = go.Figure(go.Bar(x=score_frame["Pillar"], y=score_frame["Score"], marker_color=[PALETTE["mint"], PALETTE["blue"], PALETTE["amber"], PALETTE["red"]], text=score_frame["Score"].round(0), textposition="outside"))
        fig.update_layout(title="Evidence pillars", yaxis={"range": [0, 105]})
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
    if not indicators.empty:
        st.markdown("#### Indicator readings")
        latest = indicators.iloc[-1]
        indicator_rows = pd.DataFrame([
            {"Indicator": "RSI (14)", "Value": fmt_number(latest["RSI14"], 1), "Interpretation": "Overbought" if latest["RSI14"] >= 70 else "Oversold" if latest["RSI14"] <= 30 else "Neutral"},
            {"Indicator": "MACD", "Value": fmt_number(latest["MACD"], 3), "Interpretation": "Above signal" if latest["MACD"] > latest["MACD_signal"] else "Below signal"},
            {"Indicator": "Bollinger bandwidth", "Value": fmt_pct(latest["Boll_BW"]), "Interpretation": "Expansion / higher movement" if latest["Boll_BW"] > indicators["Boll_BW"].median() else "Compression / quieter movement"},
            {"Indicator": "ATR (14)", "Value": fmt_pct(latest["ATR14_pct"]), "Interpretation": "Daily trading range"},
            {"Indicator": "Return (5d / 21d)", "Value": f"{fmt_pct(latest['ret_5'])} / {fmt_pct(latest['ret_21'])}", "Interpretation": "Recent price momentum"},
        ])
        st.dataframe(indicator_rows, use_container_width=True, hide_index=True)
    st.markdown("#### Fundamental snapshot")
    fundamentals = detail["fundamentals"]
    st.dataframe(pd.DataFrame([fundamentals]).T.rename(columns={0: "Observed value"}), use_container_width=True)
    drivers = detail["model_drivers"]
    if drivers:
        st.markdown("#### Model drivers (SHAP)")
        st.dataframe(pd.DataFrame(drivers), use_container_width=True, hide_index=True)
    else:
        st.caption("Per-stock SHAP drivers are unavailable for this artifact or the SHAP explainer could not load.")


def render_validation(bundle: dict) -> None:
    st.markdown('<div class="section-label">05 / Validation & trust</div>', unsafe_allow_html=True)
    result = validation_analysis(bundle)
    if result.get("status") != "available":
        st.info(result.get("reason", "Validation is unavailable."))
        return
    cols = st.columns(5)
    metrics = [
        ("Held-out rows", fmt_number(result.get("observations"), 0)),
        ("Directional accuracy", fmt_pct(result.get("directional_accuracy"))),
        ("Mean IC", fmt_number(result.get("mean_information_coefficient"), 3)),
        ("Quantile coverage", fmt_pct(result.get("quantile_coverage"))),
        ("MAE", fmt_pct(result.get("mae"), 2)),
    ]
    for column, (label, value) in zip(cols, metrics):
        with column:
            metric_card(label, value, "held-out test window")
    if isinstance(result.get("signal_performance"), pd.DataFrame) and not result["signal_performance"].empty:
        st.markdown("#### Realized return by original model signal")
        signal = result["signal_performance"].copy()
        signal["mean_forward_return"] = signal["mean_forward_return"].map(fmt_pct)
        signal["median_forward_return"] = signal["median_forward_return"].map(fmt_pct)
        signal["hit_rate"] = signal["hit_rate"].map(fmt_pct)
        st.dataframe(signal, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="warning-box"><b>Interpretation boundary.</b> These are held-out research diagnostics, not proof of future returns. The original notebook broadcasts a current fundamental snapshot across history; a production backtest needs point-in-time fundamentals before capital is put at risk.</div>',
        unsafe_allow_html=True,
    )


def render_live(bundle: dict, artifact_path: str, profile: dict) -> None:
    st.markdown('<div class="section-label">05 / India live analyzer</div>', unsafe_allow_html=True)
    service = cached_live_service(artifact_path)
    service.start()
    status = service.status()
    cols = st.columns(4)
    for column, (label, value, note) in zip(cols, [
        ("Provider", str(status["provider"]).upper(), status["source"]),
        ("Market", status["market_session"], "Asia/Kolkata"),
        ("Quotes received", str(status["quotes_received"]), "current process"),
        ("Last tick", str(status["last_tick"])[:19] if status["last_tick"] else "—", "provider timestamp"),
    ]):
        with column:
            metric_card(label, value, note)
    if status.get("error"):
        st.warning(f"Live provider status: {status['error']}")
    if status["provider"] not in {"zerodha", "kite", "kiteconnect", "upstox", "upstox_v3", "yahoo", "polling"}:
        st.markdown('<div class="warning-box"><b>Exchange feed is not enabled.</b> Set <code>LIVE_PROVIDER=upstox</code>, add the daily Upstox OAuth access token and ticker-to-instrument mapping in Coolify, then restart the service. The paper mode is intentionally quote-free.</div>', unsafe_allow_html=True)
        return
    if status["provider"] in {"upstox", "upstox_v3"}:
        st.info("Upstox V3 is an exchange WebSocket feed. It needs a fresh OAuth access token; NSE_EQ|ISIN keys are resolved automatically unless an explicit mapping is configured. The app never treats a missing quote as zero.")
        if status.get("subscribed_instruments"):
            st.caption(f"Live coverage: {status['subscribed_instruments']} instruments via {status.get('instrument_mapping', 'configured mapping')}. The daily model still covers the full research universe.")
        client_id = os.getenv("UPSTOX_CLIENT_ID", "")
        redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "")
        if client_id and redirect_uri:
            auth_url = "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode({
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
            })
            st.link_button("Connect Upstox", auth_url, use_container_width=False)
        st.caption("After OAuth, store the returned access token as UPSTOX_ACCESS_TOKEN in Coolify. Never place the app secret in the browser or GitHub.")
    if status["provider"] in {"yahoo", "polling"}:
        st.info("This provider is public polling and may be delayed or availability-limited. It is not presented as exchange realtime.")
    if st.button("Refresh live analyzer", type="primary"):
        st.rerun()
    if status["quotes_received"] == 0:
        st.info("Waiting for the first quote. Keep the process running during the NSE cash session.")
        return
    live_snapshot = service.analyze(profile)
    history = service.history_frame()
    if not history.empty:
        st.markdown("#### Live price tape")
        live_chart = go.Figure()
        for ticker, group in history.groupby("ticker"):
            live_chart.add_trace(go.Scatter(x=group["timestamp"], y=group["price"], mode="lines+markers", name=ticker))
        live_chart.update_layout(title="Provider quote history in this session", yaxis_title="Price", xaxis_title="IST timestamp")
        st.plotly_chart(chart_layout(live_chart, 380), use_container_width=True)
    live = live_snapshot["analysis"].copy()
    live = live.sort_values(["live_overall_score", "overall_score"], ascending=False).head(20)
    columns = [
        "ticker", "live_recommendation", "live_price", "intraday_return",
        "live_technical_confirmation", "live_overall_score", "exp_ret_21d", "sector",
    ]
    show = live[[column for column in columns if column in live]].rename(columns={
        "ticker": "Ticker", "live_recommendation": "Live decision", "live_price": "Live price",
        "intraday_return": "Intraday", "live_technical_confirmation": "Live technical",
        "live_overall_score": "Live score", "exp_ret_21d": "Daily model / 21d", "sector": "Sector",
    })
    for column in ["Intraday", "Daily model / 21d"]:
        if column in show:
            show[column] = show[column].map(fmt_pct)
    if "Live technical" in show:
        show["Live technical"] = show["Live technical"].map(lambda value: fmt_number(value, 0))
    if "Live score" in show:
        show["Live score"] = show["Live score"].map(lambda value: fmt_number(value, 0))
    if "Live price" in show:
        show["Live price"] = show["Live price"].map(lambda value: fmt_number(value, 2))
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown("#### Live insight")
    top_live = live.iloc[0]
    st.markdown(f"**{top_live['ticker']}** leads the live ranking at {fmt_number(top_live.get('live_price'), 2)} with a live score of {fmt_number(top_live.get('live_overall_score'), 0)}/100. The live overlay is only a technical confirmation; the daily model forecast remains the primary research signal.")
    st.caption("The daily ML forecast is kept separate from the live technical confirmation overlay; a tick does not silently retrain the model.")


def render_method(bundle: dict, snapshot: dict) -> None:
    st.markdown('<div class="section-label">06 / Method & data provenance</div>', unsafe_allow_html=True)
    quality = data_quality_report(bundle, snapshot["asof"])
    left, right = st.columns(2)
    with left:
        st.markdown("### Decision pipeline")
        st.markdown("""
        1. **Technical pillar** — RSI, MACD, volatility, momentum, volume, stochastic and fractionally differentiated price features.
        2. **Fundamental pillar** — valuation, profitability, leverage, dividends and revenue growth from the artifact snapshot.
        3. **Forecast pillar** — LightGBM quantile forecasts at P10, P50 and P90; the band is exposed as uncertainty.
        4. **Portfolio fit** — trailing volatility, drawdown, correlation-aware HRP weights, position caps and available cash.
        5. **Decision gate** — risk profile filters are visible and deterministic; every output carries its evidence and flags.
        """)
    with right:
        st.markdown("### Data coverage")
        for label, value in [
            ("Assets", quality.get("assets")),
            ("Held-out rows", quality.get("held_out_rows")),
            ("Selected features", quality.get("selected_features")),
            ("Fundamental snapshot assets", quality.get("fundamental_snapshot_assets")),
            ("As-of", quality.get("asof")),
        ]:
            st.write(f"**{label}:** {value if value is not None else '—'}")
        st.warning("Fundamentals are not point-in-time in the supplied notebook artifact. Treat historical performance as optimistic until that is corrected.")
    if quality.get("missing_feature_rates"):
        st.markdown("### Highest missing-feature rates")
        missing = pd.DataFrame({"Feature": list(quality["missing_feature_rates"]), "Missing rate": list(quality["missing_feature_rates"].values())})
        missing["Missing rate"] = missing["Missing rate"].map(fmt_pct)
        st.dataframe(missing, use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    artifact_path = os.getenv("ARTIFACTS_PATH", str(Path(__file__).resolve().parent / "data" / "artifacts.pkl"))
    try:
        bundle = cached_bundle(artifact_path)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        st.markdown('<div class="hero"><div class="eyebrow">AI-powered portfolio optimization</div><h1>Load the research artifact.</h1><p>The frontend is ready, but no <code>data/artifacts.pkl</code> is present yet.</p></div>', unsafe_allow_html=True)
        st.error(str(exc))
        st.markdown("""
        **Run this setup path:**

        1. Run the original notebook through Step 18 in Colab.
        2. Run the new exporter cell after Step 18.
        3. Download or copy `artifacts.pkl` into this project's `data/` folder.
        4. Restart Streamlit.
        """)
        return
    profile = sidebar_profile(bundle)
    try:
        snapshot = build_recommendation_snapshot(bundle, profile)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        st.error(f"Analysis could not be computed: {exc}")
        return
    render_header(snapshot)
    render_metrics(snapshot)
    render_insights(snapshot)
    tab_overview, tab_research, tab_portfolio, tab_validation, tab_live, tab_method = st.tabs([
        "Overview", "Stock research", "Portfolio lab", "Validation", "India live", "Method & data"
    ])
    with tab_overview:
        render_recommendations(snapshot)
        render_evidence(snapshot, bundle)
    with tab_research:
        render_research(bundle, snapshot)
    with tab_portfolio:
        render_portfolio_lab(snapshot)
    with tab_validation:
        render_validation(bundle)
    with tab_live:
        render_live(bundle, artifact_path, profile)
    with tab_method:
        render_method(bundle, snapshot)
    st.caption(f"Artifact date: {snapshot['asof'].date()} · Profile: {profile['risk']} · Educational research tool, not investment advice.")


if __name__ == "__main__":
    main()
