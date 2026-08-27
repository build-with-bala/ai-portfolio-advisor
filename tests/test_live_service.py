from __future__ import annotations

import io
import json
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

import live_service
from portfolio_advisor.live import QuoteStore, RemotePollingFeed


@pytest.fixture
def client(synthetic_bundle, monkeypatch):
    monkeypatch.setenv("LIVE_PROVIDER", "paper")
    monkeypatch.setattr(live_service, "load_bundle", lambda *a, **k: synthetic_bundle)
    live_service._service = None  # reset singleton between tests
    with TestClient(live_service.app) as test_client:
        yield test_client
    live_service._service = None


def test_health_and_status_in_paper_mode(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["provider"] == "paper"

    status = client.get("/live/status").json()
    assert status["running"] is True
    assert status["live_contract"] == "not exchange realtime"


def test_quotes_empty_in_paper_mode(client):
    payload = client.get("/live/quotes").json()
    assert payload["count"] == 0
    assert payload["quotes"] == []


def test_oauth_url_requires_config(client):
    assert client.get("/live/oauth/upstox/url").status_code == 400


def test_set_token_rejects_unknown_provider(client):
    response = client.post("/live/token", json={"access_token": "x", "provider": "nope"})
    assert response.status_code == 400


def test_remote_polling_feed_mirrors_service_quotes(monkeypatch):
    canned = json.dumps(
        {
            "quotes": [
                {
                    "ticker": "AAA.NS",
                    "price": 123.4,
                    "previous_close": 120.0,
                    "volume": 1000,
                    "timestamp": "1725877734631",
                    "source": "upstox market data websocket v3",
                }
            ]
        }
    ).encode()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: FakeResponse(canned))

    store = QuoteStore()
    feed = RemotePollingFeed(store, "http://live:8900", interval=2)
    feed.start()
    time.sleep(0.6)
    feed.stop()

    snapshot = store.snapshot()
    assert "AAA.NS" in snapshot
    assert snapshot["AAA.NS"].price == pytest.approx(123.4)
    assert snapshot["AAA.NS"].intraday_return == pytest.approx(123.4 / 120.0 - 1)
    assert snapshot["AAA.NS"].source == "remote live-service"
