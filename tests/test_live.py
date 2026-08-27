from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portfolio_advisor.live import LiveMarketService, Quote


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
