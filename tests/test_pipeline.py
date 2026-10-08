"""Checks on the training pipeline's leakage controls and metrics."""
import numpy as np
import pandas as pd

from src import data, model


def test_walk_forward_folds_never_train_on_the_future():
    dates = pd.bdate_range("2018-01-01", periods=1500).to_numpy()
    folds = list(data.walk_forward_splits(dates, data.CFG))
    assert len(folds) == data.CFG["N_SPLITS"]
    for train, validate in folds:
        gap = np.searchsorted(dates, validate[0]) - np.searchsorted(dates, train[-1]) - 1
        assert train[-1] < validate[0]
        assert gap >= data.purge_gap(data.CFG) - 1


def test_holdout_is_purged_from_development():
    dates = pd.bdate_range("2018-01-01", periods=1000).to_numpy()
    dev, test = data.holdout_dates(dates, data.CFG)
    assert len(test) == 200
    assert np.searchsorted(dates, test[0]) - len(dev) == data.purge_gap(data.CFG)


def test_fundamentals_are_not_model_inputs():
    assert not [f for f in data.CANDIDATE_FEATURES if f.startswith("f_")]


def test_fractional_difference_matches_first_difference_at_d_one():
    series = pd.Series(np.arange(50, dtype=float) ** 1.5)
    out = data.frac_diff(series, 1.0, 1e-4).dropna()
    np.testing.assert_allclose(out.values, series.diff().dropna().values, atol=1e-9)


def test_pinball_loss_and_quantile_ordering():
    assert model.pinball_loss([1.0], [0.0], 0.9) == 0.9
    assert abs(model.pinball_loss([0.0], [1.0], 0.9) - 0.1) < 1e-12
    frame = pd.DataFrame({"a": [0.0, 1.0]})

    class Fake:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v)

    out = model.predict_quantiles({0.1: Fake(0.3), 0.5: Fake(0.1), 0.9: Fake(0.2)}, frame, ["a"])
    assert (out["pred_0.1"] <= out["pred_0.5"]).all() and (out["pred_0.5"] <= out["pred_0.9"]).all()


def test_saved_report_is_consistent():
    report = model.load_report()
    assert report["holdout"]["lightgbm"]["ic_days"] > 100
    assert 0.6 < report["holdout"]["lightgbm"]["coverage"] < 0.95
    assert len(report["feature_selection"]["selected"]) == data.CFG["N_FEATURES"]
