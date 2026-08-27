from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portfolio_advisor.live import LiveMarketService, Quote, QuoteStore, UpstoxFeed


def test_quote_intraday_return_uses_previous_close():
    quote = Quote("RELIANCE.NS", 110.0, 100.0, datetime(2026, 8, 27, tzinfo=UTC))
    assert quote.intraday_return == pytest.approx(0.1)


def test_paper_live_service_is_explicitly_not_realtime(synthetic_bundle, monkeypatch):
    monkeypatch.setenv("LIVE_PROVIDER", "paper")
    service = LiveMarketService(synthetic_bundle)
    service.start()
    status = service.status()
    assert status["provider"] == "paper"
    assert status["quotes_received"] == 0
    assert status["live_contract"] == "not exchange realtime"


def test_upstox_v3_message_is_decoded_without_network():
    store = QuoteStore()
    feed = UpstoxFeed(
        store,
        "test-token",
        {"RELIANCE.NS": "NSE_EQ|INE002A01018"},
    )
    feed._on_message({
        "feeds": {
            "NSE_EQ|INE002A01018": {
                "ltpc": {"ltp": 2924.9, "cp": 2929.65, "ltt": "1725877734631"},
            },
        },
        "currentTs": "1725877734631",
    })
    quote = store.snapshot()["RELIANCE.NS"]
    assert quote.price == pytest.approx(2924.9)
    assert quote.previous_close == pytest.approx(2929.65)
    assert quote.source == "upstox market data websocket v3"
    assert quote.intraday_return == pytest.approx(2924.9 / 2929.65 - 1)
