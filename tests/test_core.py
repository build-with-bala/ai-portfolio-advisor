from __future__ import annotations

from portfolio_advisor.core import (
    build_recommendation_snapshot,
    data_quality_report,
    execution_plan,
    jsonable,
    validation_analysis,
)


def test_profile_snapshot_contains_all_decision_pillars(synthetic_bundle):
    snapshot = build_recommendation_snapshot(
        synthetic_bundle,
        {
            "risk": "Moderate",
            "capital": 100_000,
            "max_holdings": 3,
            "min_confidence": 0.2,
            "min_expected_return": 0,
            "min_fundamental_score": 0,
            "include_watchlist": False,
        },
    )
    assert len(snapshot["analysis"]) == 6
    assert len(snapshot["picks"]) == 3
    assert {"technical_score", "fundamental_score", "model_score", "overall_score"}.issubset(snapshot["analysis"].columns)
    assert snapshot["summary"]["positions"] == 3
    assert snapshot["summary"]["invested"] <= 100_000


def test_profile_filters_tickers_and_sectors(synthetic_bundle):
    snapshot = build_recommendation_snapshot(
        synthetic_bundle,
        {
            "risk": "Conservative",
            "capital": 50_000,
            "max_holdings": 5,
            "min_confidence": 0,
            "min_fundamental_score": 0,
            "exclude_tickers": ["AAA"],
            "exclude_sectors": ["Technology"],
            "include_watchlist": False,
        },
    )
    assert set(snapshot["picks"]["ticker"]).issubset({"DDD", "EEE", "FFF"})
    assert "AAA" not in set(snapshot["picks"]["ticker"])


def test_execution_plan_preserves_integer_order_quantity(synthetic_bundle):
    snapshot = build_recommendation_snapshot(
        synthetic_bundle,
        {"risk": "Aggressive", "capital": 100_000, "max_holdings": 2, "min_fundamental_score": 0},
    )
    plan = execution_plan(snapshot, steps=7)
    assert plan["shares"].ge(0).all()
    if not plan.empty:
        totals = plan.groupby("ticker")["shares"].sum().to_dict()
        expected = snapshot["picks"].set_index("ticker")["target_shares"].to_dict()
        assert totals == {key: value for key, value in expected.items() if value > 0}


def test_validation_and_quality_are_explicitly_available(synthetic_bundle):
    validation = validation_analysis(synthetic_bundle)
    quality = data_quality_report(synthetic_bundle)
    assert validation["status"] == "available"
    assert validation["observations"] == len(synthetic_bundle["res"])
    assert quality["assets"] == 6
    assert quality["fundamental_point_in_time"] is False


def test_jsonable_converts_frames_and_timestamps(synthetic_bundle):
    snapshot = build_recommendation_snapshot(synthetic_bundle, {"risk": "Moderate", "min_fundamental_score": 0})
    payload = jsonable({"asof": snapshot["asof"], "picks": snapshot["picks"].head(1)})
    assert isinstance(payload["asof"], str)
    assert isinstance(payload["picks"], list)
