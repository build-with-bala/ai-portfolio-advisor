"""Grounded analyst chatbot over the portfolio artifact.

Backed by the codex-api service (`POST /v1/chat`, a sandboxed pure-model
endpoint). The chatbot is grounded on a compact table of the *entire* current
stock universe (every ticker in the artifact), passed once as the conversation's
system prompt; follow-up turns reuse the server-side conversation. It answers
from the data only — it never places trades and never gives personalised advice.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

DEFAULT_URL = "https://codex-api.ahoum.com"
# Cloudflare (error 1010) bans the default Python-urllib UA; send a normal one.
USER_AGENT = "Mozilla/5.0 (compatible; ai-portfolio-advisor/1.0; +https://fn.iimbg.com)"

SYSTEM_TEMPLATE = (
    "You are the analyst assistant inside 'AI Portfolio Advisor', a research tool for "
    "Indian (NSE) equities. Below is the model's latest analysis snapshot for {n} stocks "
    "from the Nifty-500 universe (as of {asof}). Answer ONLY from this data. If a stock "
    "is not in the table, say it is not in the current universe rather than guessing. Be "
    "concise; when you cite a stock, quote its numbers (signal, score out of 100, expected "
    "21-day return %, confidence 0-1, and relevant fundamentals). When ranking or comparing, "
    "use the table. Everything here is model research output, NOT personalised financial "
    "advice — say so plainly if the user asks what to buy.\n"
    "Table columns: ticker | sector | signal | score(/100) | ret21d(%) | conf(0-1) | "
    "P/E | ROE | D/E | vol(%annualised).\n\n"
    "DATA:\n{table}"
)


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _fmt(value: Any, nd: int = 1, scale: float = 1.0) -> str:
    f = _num(value)
    return "" if f is None else f"{f * scale:.{nd}f}"


def build_stock_context(snapshot: dict, limit: int | None = None) -> tuple[str, int]:
    """A compact pipe-delimited table of every stock in the snapshot's analysis."""
    analysis = snapshot.get("analysis")
    if not isinstance(analysis, pd.DataFrame) or analysis.empty:
        return "", 0
    df = analysis.copy()
    if "overall_score" in df.columns:
        df = df.sort_values("overall_score", ascending=False)
    if limit:
        df = df.head(limit)
    rows = ["ticker|sector|signal|score|ret21d%|conf|PE|ROE|DE|vol%"]
    for _, r in df.iterrows():
        rows.append("|".join([
            str(r.get("ticker", "")),
            (str(r.get("sector", "") or "Unknown"))[:20],
            str(r.get("recommendation", r.get("model_signal", ""))),
            _fmt(r.get("overall_score"), 0),
            _fmt(r.get("exp_ret_21d"), 1, 100.0),
            _fmt(r.get("confidence"), 2),
            _fmt(r.get("f_PE"), 1),
            _fmt(r.get("f_ROE"), 2),
            _fmt(r.get("f_DE"), 0),
            _fmt(r.get("annual_volatility"), 1, 100.0),
        ]))
    return "\n".join(rows), len(df)


def build_system_prompt(snapshot: dict, limit: int | None = None) -> str:
    table, n = build_stock_context(snapshot, limit)
    asof = snapshot.get("asof")
    asof_str = str(getattr(asof, "date", lambda: asof)()) if asof is not None else "latest"
    return SYSTEM_TEMPLATE.format(n=n, asof=asof_str, table=table)


class StockChat:
    """Thin client over codex-api `/v1/chat` with transient-failure retries."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 150) -> None:
        self.base = (base_url or os.getenv("CODEX_API_URL", DEFAULT_URL)).rstrip("/")
        self.key = api_key or os.getenv("CODEX_API_KEY", "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def ask(self, message: str, conversation_id: str | None = None, system: str | None = None, retries: int = 1) -> dict[str, Any]:
        if not self.key:
            raise RuntimeError("CODEX_API_KEY is not set; the analyst chatbot is unconfigured.")
        body: dict[str, Any] = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        elif system:
            body["system"] = system
        payload = json.dumps(body).encode("utf-8")
        last_error: str | None = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    f"{self.base}/v1/chat",
                    data=payload,
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": USER_AGENT},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("error"):
                    last_error = str(data.get("detail") or data.get("error"))
                    if data.get("error") == "codex_failed" and attempt < retries:
                        continue
                    raise RuntimeError(f"codex-api error: {data.get('error')}")
                return {"conversation_id": data.get("conversation_id"), "text": data.get("text", ""), "usage": data.get("usage")}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:300]
                if exc.code == 409 and attempt < retries:  # conversation_busy — one retry
                    last_error = detail
                    continue
                raise RuntimeError(f"codex-api HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
                if attempt < retries:
                    continue
                raise RuntimeError(f"codex-api unreachable: {exc}") from exc
        raise RuntimeError(f"codex-api failed after retries: {last_error}")
