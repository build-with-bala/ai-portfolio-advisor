from __future__ import annotations

import io
import json
import urllib.request

import pytest

from portfolio_advisor.chat import StockChat, build_stock_context, build_system_prompt
from portfolio_advisor.core import build_recommendation_snapshot


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_context_covers_whole_universe(synthetic_bundle):
    snap = build_recommendation_snapshot(synthetic_bundle, {"risk": "Moderate", "max_holdings": 50})
    table, n = build_stock_context(snap)
    assert n == snap["analysis"]["ticker"].nunique()
    assert table.splitlines()[0].startswith("ticker|sector|signal")
    assert "AAA" in table  # a synthetic ticker


def test_system_prompt_embeds_table(synthetic_bundle):
    snap = build_recommendation_snapshot(synthetic_bundle, {"risk": "Moderate"})
    prompt = build_system_prompt(snap)
    assert "DATA:" in prompt and "ticker|sector" in prompt


def test_ask_success(monkeypatch):
    canned = json.dumps({"conversation_id": "c1", "text": "hi there", "model": "gpt-5"}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=150: _Resp(canned))
    result = StockChat(api_key="cxk_test").ask("hello", system="sys")
    assert result["conversation_id"] == "c1"
    assert result["text"] == "hi there"


def test_ask_retries_transient_codex_failed(monkeypatch):
    calls = {"n": 0}

    def fake(req, timeout=150):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(json.dumps({"error": "codex_failed", "detail": "transient"}).encode())
        return _Resp(json.dumps({"conversation_id": "c2", "text": "ok"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = StockChat(api_key="cxk_test").ask("hello", system="s")
    assert result["text"] == "ok"
    assert calls["n"] == 2


def test_ask_unconfigured_raises():
    with pytest.raises(RuntimeError):
        StockChat(api_key="").ask("hello")
