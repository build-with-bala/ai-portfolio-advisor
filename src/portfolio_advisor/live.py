"""Indian-market quote adapters and the transparent intraday overlay.

The trained model remains the daily research model from the notebook. A live
quote does not silently retrain it. Instead, live prices update the displayed
price, intraday return, and a clearly-labelled technical confirmation overlay.
This keeps the distinction between a model forecast and a tick observable.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import urllib.request
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time
from threading import Event, Lock, Thread
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .core import build_recommendation_snapshot, jsonable

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class Quote:
    ticker: str
    price: float
    previous_close: float | None
    timestamp: datetime
    volume: float | None = None
    source: str = "unknown"

    @property
    def intraday_return(self) -> float | None:
        if self.previous_close is None or self.previous_close <= 0:
            return None
        return self.price / self.previous_close - 1.0


class QuoteStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._quotes: dict[str, Quote] = {}
        self._history: dict[str, deque[Quote]] = defaultdict(lambda: deque(maxlen=240))

    def update(self, quote: Quote) -> None:
        with self._lock:
            self._quotes[quote.ticker] = quote
            history = self._history[quote.ticker]
            if not history or quote.timestamp > history[-1].timestamp or quote.price != history[-1].price:
                history.append(quote)

    def snapshot(self) -> dict[str, Quote]:
        with self._lock:
            return dict(self._quotes)

    def history_frame(self) -> pd.DataFrame:
        with self._lock:
            rows = [
                {
                    "ticker": quote.ticker,
                    "timestamp": quote.timestamp,
                    "price": quote.price,
                    "intraday_return": quote.intraday_return,
                    "source": quote.source,
                }
                for history in self._history.values()
                for quote in history
            ]
        return pd.DataFrame(rows)


class BaseFeed:
    source = "unknown"

    def __init__(self, store: QuoteStore) -> None:
        self.store = store
        self.started = False
        self.last_error: str | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


class PaperFeed(BaseFeed):
    source = "paper/no live provider"


class RemotePollingFeed(BaseFeed):
    """Consume the standalone live microservice instead of a broker directly.

    The UI runs with ``LIVE_PROVIDER=remote`` and ``LIVE_SERVICE_URL`` pointing at
    the live-service (which owns the single broker WebSocket). This feed polls the
    service's ``/live/quotes`` snapshot and mirrors it into the local store, so all
    of the existing blend/analysis logic keeps working unchanged. No broker
    credentials ever touch the UI process.
    """

    source = "remote live-service"

    def __init__(self, store: QuoteStore, base_url: str, interval: int = 5) -> None:
        super().__init__(store)
        self.base_url = base_url.rstrip("/")
        self.interval = max(interval, 2)
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.upstream_source: str | None = None

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread = Thread(target=self._run, name="remote-live-poller", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.started = False

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/live/quotes", headers={"User-Agent": "ai-portfolio-advisor-ui"}
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                quotes = payload.get("quotes", payload) if isinstance(payload, Mapping) else []
                if isinstance(quotes, Mapping):
                    quotes = list(quotes.values())
                for item in quotes or []:
                    if not isinstance(item, Mapping):
                        continue
                    price = _number(item.get("price"))
                    ticker = item.get("ticker")
                    if ticker is None or price is None:
                        continue
                    self.upstream_source = item.get("source", self.upstream_source)
                    self.store.update(
                        Quote(
                            ticker=str(ticker),
                            price=price,
                            previous_close=_number(item.get("previous_close")),
                            timestamp=_tick_time(item.get("timestamp")),
                            volume=_number(item.get("volume")),
                            source=self.source,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - the UI must survive service hiccups
                self.last_error = f"live-service poll failed: {exc}"
            self.stop_event.wait(self.interval)


class YahooPollingFeed(BaseFeed):
    """Optional public polling fallback; never presented as exchange realtime."""

    source = "yahoo polling / delayed or availability-limited"

    def __init__(self, store: QuoteStore, tickers: list[str], interval: int = 60) -> None:
        super().__init__(store)
        self.tickers = tickers
        self.interval = max(interval, 30)
        self.stop_event = Event()
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread = Thread(target=self._run, name="yahoo-india-poller", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.started = False

    def _run(self) -> None:
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:
            self.last_error = f"yfinance is unavailable: {exc}"
            return
        while not self.stop_event.is_set():
            try:
                for ticker in self.tickers:
                    frame = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=True)
                    if frame.empty or "Close" not in frame:
                        continue
                    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
                    if close.empty:
                        continue
                    # Public intraday history does not reliably carry a previous
                    # exchange close, so do not manufacture an intraday return.
                    self.store.update(Quote(ticker, float(close.iloc[-1]), None, datetime.now(IST), source=self.source))
            except Exception as exc:  # noqa: BLE001 - provider failures must not kill the service
                self.last_error = str(exc)
            self.stop_event.wait(self.interval)


class ZerodhaKiteFeed(BaseFeed):
    """Zerodha Kite Connect WebSocket adapter.

    ``KITE_INSTRUMENT_TOKENS`` is a JSON mapping such as
    ``{"RELIANCE.NS": 738561, "TCS.NS": 2953217}``. Credentials are read
    only from environment variables and are never returned by ``status``.
    """

    source = "zerodha kite websocket"

    def __init__(self, store: QuoteStore, api_key: str, access_token: str, tokens: Mapping[str, int]) -> None:
        super().__init__(store)
        self.api_key = api_key
        self.access_token = access_token
        self.tokens = {str(key): int(value) for key, value in tokens.items()}
        self.reverse_tokens = {value: key for key, value in self.tokens.items()}
        self.client: Any = None
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.started:
            return
        try:
            from kiteconnect import KiteTicker  # type: ignore
        except ImportError as exc:
            self.last_error = f"kiteconnect is unavailable: {exc}"
            return
        if not self.api_key or not self.access_token or not self.tokens:
            self.last_error = "Kite provider needs KITE_API_KEY, KITE_ACCESS_TOKEN, and KITE_INSTRUMENT_TOKENS."
            return
        self.client = KiteTicker(self.api_key, self.access_token)
        self.client.on_ticks = self._on_ticks
        self.client.on_connect = self._on_connect
        self.client.on_close = self._on_close
        self.client.on_error = self._on_error
        self.started = True
        self.thread = Thread(target=self.client.connect, kwargs={"threaded": False}, name="kite-india-feed", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.started = False
        if self.client is not None:
            try:
                self.client.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must remain best-effort
                self.last_error = self.last_error or f"Kite close failed: {exc}"

    def _on_connect(self, ws: Any, response: Any) -> None:
        del response
        ws.subscribe(list(self.reverse_tokens))
        ws.set_mode(ws.MODE_QUOTE, list(self.reverse_tokens))

    def _on_ticks(self, ws: Any, ticks: list[dict[str, Any]]) -> None:
        del ws
        for tick in ticks:
            token = tick.get("instrument_token")
            ticker = self.reverse_tokens.get(token)
            price = _number(tick.get("last_price"))
            if ticker is None or price is None:
                continue
            ohlc = tick.get("ohlc") or {}
            self.store.update(
                Quote(
                    ticker=ticker,
                    price=price,
                    previous_close=_number(ohlc.get("close")),
                    timestamp=_tick_time(tick.get("exchange_timestamp") or tick.get("last_trade_time")),
                    volume=_number(tick.get("volume_traded")),
                    source=self.source,
                )
            )

    def _on_close(self, ws: Any, code: Any, reason: Any) -> None:
        del ws
        self.last_error = f"Kite WebSocket closed ({code}): {reason}"

    def _on_error(self, ws: Any, code: Any, reason: Any) -> None:
        del ws
        self.last_error = f"Kite WebSocket error ({code}): {reason}"


class UpstoxFeed(BaseFeed):
    """Upstox Market Data Feed V3 adapter.

    Upstox delivers protobuf frames, but the official Python SDK decodes them
    into dictionaries before emitting the ``message`` event.  The adapter
    accepts an explicit ``ticker -> NSE_EQ|ISIN`` mapping because a Yahoo
    symbol (for example ``RELIANCE.NS``) is not an Upstox instrument key.

    The access token is deliberately read only from the environment. Upstox
    access tokens are short-lived OAuth credentials; an app key and secret by
    themselves cannot authorize the WebSocket feed. If no explicit mapping is
    supplied, the current public Upstox NSE instrument master is used to map
    the artifact's ``.NS`` symbols to ``NSE_EQ|ISIN`` keys.
    """

    source = "upstox market data websocket v3"

    def __init__(
        self,
        store: QuoteStore,
        access_token: str,
        instrument_keys: Mapping[str, str],
        max_instruments: int = 100,
    ) -> None:
        super().__init__(store)
        self.access_token = access_token
        self.instrument_keys = {
            str(ticker): str(key)
            for ticker, key in list(instrument_keys.items())[: max(1, max_instruments)]
            if str(key).startswith("NSE_EQ|")
        }
        self.reverse_keys = {key: ticker for ticker, key in self.instrument_keys.items()}
        self.client: Any = None
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.started:
            return
        if not self.access_token:
            self.last_error = "Upstox needs UPSTOX_ACCESS_TOKEN from the OAuth token exchange."
            return
        if not self.instrument_keys:
            self.last_error = (
                "Upstox needs UPSTOX_INSTRUMENT_KEYS as JSON, for example "
                '{"RELIANCE.NS":"NSE_EQ|<ISIN>"}.'
            )
            return
        try:
            import upstox_client  # type: ignore
        except ImportError as exc:
            self.last_error = f"upstox-python-sdk is unavailable: {exc}"
            return
        try:
            configuration = upstox_client.Configuration()
            configuration.access_token = self.access_token
            self.client = upstox_client.MarketDataStreamerV3(
                upstox_client.ApiClient(configuration),
                list(self.reverse_keys),
                "ltpc",
            )
            self.client.on("open", self._on_open)
            self.client.on("message", self._on_message)
            self.client.on("error", self._on_error)
            self.client.on("close", self._on_close)
            self.started = True
            self.thread = Thread(target=self._connect, name="upstox-india-feed", daemon=True)
            self.thread.start()
        except Exception as exc:  # noqa: BLE001 - provider failures must be visible in status
            self.last_error = f"Upstox setup failed: {exc}"

    def stop(self) -> None:
        self.started = False
        if self.client is not None:
            try:
                disconnect = getattr(self.client, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception as exc:  # noqa: BLE001 - shutdown must remain best-effort
                self.last_error = self.last_error or f"Upstox disconnect failed: {exc}"

    def _connect(self) -> None:
        try:
            self.client.connect()
        except Exception as exc:  # noqa: BLE001 - keep the UI alive when feed fails
            self.started = False
            self.last_error = f"Upstox WebSocket failed: {exc}"

    def _on_open(self, *args: Any) -> None:
        del args
        self.last_error = None

    def _on_message(self, message: Any) -> None:
        payload = _as_mapping(message)
        feeds = payload.get("feeds") if isinstance(payload, Mapping) else None
        if not isinstance(feeds, Mapping):
            return
        for instrument_key, instrument_payload in feeds.items():
            ticker = self.reverse_keys.get(str(instrument_key))
            if ticker is None or not isinstance(instrument_payload, Mapping):
                continue
            ltpc = _find_mapping(instrument_payload, "ltpc") or instrument_payload
            price = _number(ltpc.get("ltp"))
            if price is None:
                continue
            timestamp = _tick_time(ltpc.get("ltt") or payload.get("currentTs"))
            previous_close = _number(ltpc.get("cp"))
            volume = _number(_find_value(instrument_payload, "vtt"))
            self.store.update(
                Quote(
                    ticker=ticker,
                    price=price,
                    previous_close=previous_close,
                    timestamp=timestamp,
                    volume=volume,
                    source=self.source,
                )
            )

    def _on_close(self, *args: Any) -> None:
        self.started = False
        self.last_error = f"Upstox WebSocket closed: {' '.join(str(arg) for arg in args if arg is not None)}".strip()

    def _on_error(self, *args: Any) -> None:
        self.last_error = f"Upstox WebSocket error: {' '.join(str(arg) for arg in args if arg is not None)}".strip()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else {}
    return {}


def upstox_nse_instrument_keys(tickers: list[str]) -> dict[str, str]:
    """Resolve Yahoo-style NSE symbols through Upstox's public NSE master."""

    desired = {str(ticker): str(ticker).removesuffix(".NS") for ticker in tickers}
    url = os.getenv(
        "UPSTOX_INSTRUMENT_MASTER_URL",
        "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ai-portfolio-advisor"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, gzip.GzipFile(
            fileobj=io.BytesIO(response.read())
        ) as compressed:
            records = json.load(compressed)
    except Exception:  # noqa: BLE001 - feed status reports the missing mapping
        return {}
    if isinstance(records, Mapping):
        records = records.get("data", [])
    if not isinstance(records, list):
        return {}
    resolved: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("segment") != "NSE_EQ":
            continue
        if record.get("instrument_type") not in {"EQ", "BE"}:
            continue
        symbol = str(record.get("trading_symbol", "")).strip()
        instrument_key = str(record.get("instrument_key", "")).strip()
        for ticker, wanted_symbol in desired.items():
            if symbol == wanted_symbol and instrument_key.startswith("NSE_EQ|"):
                resolved[ticker] = instrument_key
    return resolved


def _find_mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    candidate = value.get(key)
    if isinstance(candidate, Mapping):
        return candidate
    for child in value.values():
        found = _find_mapping(child, key)
        if found is not None:
            return found
    return None


def _find_value(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    if key in value:
        return value[key]
    for child in value.values():
        found = _find_value(child, key)
        if found is not None:
            return found
    return None


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _tick_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    try:
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1_000
        return datetime.fromtimestamp(timestamp, tz=IST)
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return datetime.now(IST)


def _session_status(now: datetime | None = None) -> str:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return "CLOSED / weekend"
    if time(9, 15) <= now.time() <= time(15, 30):
        return "OPEN / NSE cash session"
    return "CLOSED / outside NSE cash session"


class LiveMarketService:
    def __init__(self, bundle: Mapping[str, Any]) -> None:
        self.bundle = bundle
        self.store = QuoteStore()
        self.provider_name = os.getenv("LIVE_PROVIDER", "paper").lower()
        tickers = sorted(str(item) for item in bundle["res"]["ticker"].unique())
        self.feed = self._make_feed(tickers)

    def _make_feed(self, tickers: list[str]) -> BaseFeed:
        if self.provider_name in {"upstox", "upstox_v3"}:
            access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
            try:
                mapping = json.loads(os.getenv("UPSTOX_INSTRUMENT_KEYS", "{}"))
            except json.JSONDecodeError:
                mapping = {}
            if not isinstance(mapping, Mapping):
                mapping = {}
            if not mapping:
                bundle_mapping = self.bundle.get("upstox_instrument_keys", {})
                mapping = bundle_mapping if isinstance(bundle_mapping, Mapping) else {}
            if not mapping and access_token:
                mapping = upstox_nse_instrument_keys(tickers)
            mapping = {ticker: mapping[ticker] for ticker in tickers if ticker in mapping}
            return UpstoxFeed(
                self.store,
                access_token,
                mapping,
                int(os.getenv("UPSTOX_MAX_INSTRUMENTS", "100")),
            )
        if self.provider_name in {"zerodha", "kite", "kiteconnect"}:
            try:
                tokens = json.loads(os.getenv("KITE_INSTRUMENT_TOKENS", "{}"))
            except json.JSONDecodeError:
                tokens = {}
            return ZerodhaKiteFeed(self.store, os.getenv("KITE_API_KEY", ""), os.getenv("KITE_ACCESS_TOKEN", ""), tokens)
        if self.provider_name in {"remote", "microservice"}:
            base_url = os.getenv("LIVE_SERVICE_URL", "http://live:8900")
            return RemotePollingFeed(self.store, base_url, int(os.getenv("LIVE_POLL_SECONDS", "5")))
        if self.provider_name in {"yahoo", "polling"}:
            return YahooPollingFeed(self.store, tickers, int(os.getenv("LIVE_POLL_SECONDS", "60")))
        return PaperFeed(self.store)

    def start(self) -> None:
        self.feed.start()

    def stop(self) -> None:
        self.feed.stop()

    def status(self) -> dict[str, Any]:
        quotes = self.store.snapshot()
        latest = max((quote.timestamp for quote in quotes.values()), default=None)
        status = {
            "provider": self.provider_name,
            "source": self.feed.source,
            "running": self.feed.started,
            "quotes_received": len(quotes),
            "last_tick": latest,
            "market_session": _session_status(),
            "error": self.feed.last_error,
            "live_contract": "exchange websocket" if self.provider_name in {"zerodha", "kite", "kiteconnect", "upstox", "upstox_v3"} else "not exchange realtime",
        }
        if isinstance(self.feed, UpstoxFeed):
            status["subscribed_instruments"] = len(self.feed.instrument_keys)
            status["instrument_mapping"] = "explicit env/bundle" if os.getenv("UPSTOX_INSTRUMENT_KEYS") else "Upstox NSE master"
        return status

    def history_frame(self) -> pd.DataFrame:
        """Return the in-process quote tape for the live chart."""

        return self.store.history_frame()

    def analyze(self, profile: Mapping[str, Any] | None = None, asof: Any | None = None) -> dict[str, Any]:
        self.start()
        snapshot = build_recommendation_snapshot(self.bundle, profile, asof)
        quotes = self.store.snapshot()
        analysis = snapshot["analysis"].copy()
        analysis["live_price"] = analysis["ticker"].map(lambda ticker: quotes[ticker].price if ticker in quotes else None)
        analysis["intraday_return"] = analysis["ticker"].map(lambda ticker: quotes[ticker].intraday_return if ticker in quotes else None)
        analysis["live_source"] = analysis["ticker"].map(lambda ticker: quotes[ticker].source if ticker in quotes else None)
        analysis["live_last_tick"] = analysis["ticker"].map(lambda ticker: quotes[ticker].timestamp if ticker in quotes else None)
        analysis["live_technical_confirmation"] = analysis["intraday_return"].map(
            lambda value: float(50 + 50 * np.tanh(value / 0.01)) if value is not None else None
        )
        analysis["live_overall_score"] = analysis.apply(
            lambda row: float(0.85 * row["overall_score"] + 0.15 * row["live_technical_confirmation"])
            if pd.notna(row.get("live_technical_confirmation")) else row["overall_score"],
            axis=1,
        )
        analysis["live_recommendation"] = analysis.apply(_live_decision, axis=1)
        snapshot["analysis"] = analysis
        if not snapshot["picks"].empty:
            picks = snapshot["picks"].copy()
            live = analysis.set_index("ticker")
            picks["live_price"] = picks["ticker"].map(live["live_price"])
            picks["intraday_return"] = picks["ticker"].map(live["intraday_return"])
            picks["live_overall_score"] = picks["ticker"].map(live["live_overall_score"])
            picks["live_recommendation"] = picks["ticker"].map(live["live_recommendation"])
            picks["live_target_shares"] = picks.apply(
                lambda row: int((row["target_notional"] / row["live_price"]) // 1)
                if pd.notna(row.get("live_price")) and row["live_price"] > 0 else row["target_shares"], axis=1
            )
            snapshot["picks"] = picks
        snapshot["live"] = self.status()
        return snapshot


def _live_decision(row: pd.Series) -> str:
    base = str(row.get("recommendation", "WATCH"))
    confirmation = row.get("live_technical_confirmation")
    if pd.isna(confirmation):
        return base
    if "BUY" in base and confirmation < 35:
        return "WATCH / live confirmation weak"
    if base == "WATCH" and confirmation > 70 and row.get("overall_score", 0) >= 50:
        return "BUY / live confirmation strong"
    return base


def live_payload(service: LiveMarketService, profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return jsonable(service.analyze(profile))
