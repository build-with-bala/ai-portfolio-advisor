"""Standalone live-market microservice.

Owns ONE upstream broker connection (Upstox V3 / Zerodha Kite / Yahoo / paper)
and serves live NSE quotes over HTTP + SSE to the research UI and any other
consumer. Splitting live data into its own process means:

  * a single broker WebSocket instead of one per Streamlit session,
  * a central OAuth token lifecycle (the client secret stays only here),
  * independent restart / scale from the research UI.

The daily research model is NOT loaded here; this service is purely the live
quote + token authority. Consumers keep the model and blend the intraday
overlay themselves (the UI does this by running with ``LIVE_PROVIDER=remote``).

Run with: uvicorn live_service:app --host 0.0.0.0 --port 8900
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from portfolio_advisor.core import load_bundle
from portfolio_advisor.live import LiveMarketService, Quote

UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
UPSTOX_DIALOG_URL = "https://api.upstox.com/v2/login/authorization/dialog"


def _cors_origins() -> list[str]:
    raw = os.getenv("LIVE_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="AI Portfolio Advisor — Live Market Service",
    version="0.1.0",
    description="Single-owner broker feed + token authority serving live NSE quotes.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Module-level singleton so the whole service shares ONE feed / broker connection.
_service: LiveMarketService | None = None


def get_service() -> LiveMarketService:
    global _service
    if _service is None:
        bundle = load_bundle()  # only used for the ticker universe
        _service = LiveMarketService(bundle)
        _service.start()
    return _service


def rebuild_service() -> LiveMarketService:
    """Recreate the feed so a freshly-set env token/provider takes effect."""
    global _service
    if _service is not None:
        try:
            _service.stop()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
    _service = None
    return get_service()


def _quote_dict(quote: Quote) -> dict[str, Any]:
    return {
        "ticker": quote.ticker,
        "price": quote.price,
        "previous_close": quote.previous_close,
        "intraday_return": quote.intraday_return,
        "volume": quote.volume,
        "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
        "source": quote.source,
    }


@app.on_event("startup")
def _startup() -> None:
    # Start the feed eagerly so quotes accumulate before the first request.
    try:
        get_service()
    except Exception:  # noqa: BLE001 - a missing artifact must not crash boot
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        service = get_service()
        status = service.status()
        return {"status": "ok", "provider": status["provider"], "quotes": status["quotes_received"]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "detail": str(exc)}


@app.get("/live/status")
def live_status() -> dict[str, Any]:
    return get_service().status()


@app.get("/live/quotes")
def live_quotes() -> dict[str, Any]:
    service = get_service()
    quotes = [_quote_dict(quote) for quote in service.store.snapshot().values()]
    return {"count": len(quotes), "quotes": quotes, "status": service.status()}


@app.get("/live/quotes/{ticker}")
def live_quote(ticker: str) -> dict[str, Any]:
    service = get_service()
    quote = service.store.snapshot().get(ticker.upper())
    if quote is None:
        raise HTTPException(status_code=404, detail=f"No live quote for {ticker}.")
    return _quote_dict(quote)


@app.get("/live/tape")
def live_tape() -> dict[str, Any]:
    frame = get_service().history_frame()
    records = json.loads(frame.to_json(orient="records", date_format="iso")) if not frame.empty else []
    return {"count": len(records), "tape": records}


@app.get("/live/stream")
async def live_stream(interval: float = 3.0) -> StreamingResponse:
    """Server-Sent Events stream of the quote snapshot every `interval` seconds."""
    service = get_service()
    interval = min(max(interval, 1.0), 30.0)

    async def event_generator() -> Any:
        while True:
            quotes = [_quote_dict(quote) for quote in service.store.snapshot().values()]
            payload = json.dumps({"quotes": quotes, "count": len(quotes)})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/live/oauth/upstox/url")
def upstox_oauth_url() -> dict[str, Any]:
    client_id = os.getenv("UPSTOX_CLIENT_ID", "")
    redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "")
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=400, detail="UPSTOX_CLIENT_ID and UPSTOX_REDIRECT_URI are not configured.")
    query = urllib.parse.urlencode(
        {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri}
    )
    return {"authorization_url": f"{UPSTOX_DIALOG_URL}?{query}"}


@app.post("/live/oauth/upstox/exchange")
def upstox_oauth_exchange(code: str = Body(..., embed=True)) -> dict[str, Any]:
    """Exchange an Upstox OAuth `code` for an access token, then (re)start the feed.

    The client secret is read only from the environment and is never returned.
    """
    client_id = os.getenv("UPSTOX_CLIENT_ID", "")
    client_secret = os.getenv("UPSTOX_CLIENT_SECRET", "")
    redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "")
    missing = [n for n, v in [("UPSTOX_CLIENT_ID", client_id), ("UPSTOX_CLIENT_SECRET", client_secret), ("UPSTOX_REDIRECT_URI", redirect_uri)] if not v]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing server config: {', '.join(missing)}")
    data = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        UPSTOX_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # surface Upstox's error without leaking the secret
        detail = exc.read().decode("utf-8", "ignore")[:500]
        raise HTTPException(status_code=502, detail=f"Upstox token exchange failed ({exc.code}): {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Upstox token exchange failed: {exc}") from exc
    token = body.get("access_token")
    if not token:
        raise HTTPException(status_code=502, detail="Upstox response did not include an access_token.")
    os.environ["UPSTOX_ACCESS_TOKEN"] = token
    os.environ.setdefault("LIVE_PROVIDER", "upstox")
    service = rebuild_service()
    return {"status": "connected", "provider": service.provider_name, "live": service.status()}


@app.post("/live/token")
def set_token(access_token: str = Body(..., embed=True), provider: str = Body("upstox", embed=True)) -> dict[str, Any]:
    """Directly set a broker access token (e.g. Upstox/Kite) and restart the feed."""
    provider = provider.lower()
    if provider in {"upstox", "upstox_v3"}:
        os.environ["UPSTOX_ACCESS_TOKEN"] = access_token
    elif provider in {"zerodha", "kite", "kiteconnect"}:
        os.environ["KITE_ACCESS_TOKEN"] = access_token
    else:
        raise HTTPException(status_code=400, detail="provider must be 'upstox' or 'kite'.")
    os.environ["LIVE_PROVIDER"] = provider
    service = rebuild_service()
    return {"status": "restarted", "provider": service.provider_name, "live": service.status()}
