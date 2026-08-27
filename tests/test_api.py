from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api_app


def test_recommendations_endpoint_returns_profile_analysis(synthetic_bundle, monkeypatch):
    monkeypatch.setattr(api_app, "get_bundle", lambda: synthetic_bundle)
    client = TestClient(api_app.app)
    response = client.post(
        "/api/v1/recommendations",
        json={"risk": "Moderate", "capital": 100000, "max_holdings": 3, "min_fundamental_score": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert "picks" in body
    assert "analysis" in body
    assert "execution_plan" in body
    assert body["profile"]["risk"] == "Moderate"


def test_health_is_degraded_without_artifact(monkeypatch):
    def fail():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(api_app, "get_bundle", fail)
    response = TestClient(api_app.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
