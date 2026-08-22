"""Tests for the champion/challenger gate.

A gate that only ever says yes is decoration. Most of these tests assert that promotion is
*refused*, which is the behaviour that actually protects the deployment.
"""

import pytest

from themepark.config import PipelineSettings
from themepark.promote import NoViableModelError, decide, versions_to_retire


def result(model, mae, high_wait_rmse=20.0, coverage=1.0):
    """Minimal shape of an `evaluate.evaluate_model` return value."""
    return {
        "model": model,
        "overall_mae": mae,
        "high_wait_rides_rmse": high_wait_rmse,
        "coverage": coverage,
    }


CFG = PipelineSettings()


def test_promotes_when_nothing_is_registered_yet():
    decision = decide([result("baseline", 10.0), result("prophet_fleet", 6.0)], None, CFG)
    assert decision.promoted
    assert decision.winner == "prophet_fleet"
    assert decision.incumbent is None


def test_promotes_a_clear_improvement_over_the_champion():
    decision = decide(
        [result("baseline", 10.0), result("prophet_fleet", 5.0)],
        result("xgb_global", 7.0),
        CFG,
    )
    assert decision.promoted
    assert decision.winner == "prophet_fleet"
    assert "improves MAE" in decision.reason


def test_declines_a_marginal_improvement():
    """A 1% gain is holdout noise. Churning the champion invalidates the accuracy history."""
    decision = decide(
        [result("baseline", 10.0), result("prophet_fleet", 6.94)],
        result("xgb_global", 7.0),
        CFG,
    )
    assert not decision.promoted
    assert decision.winner == "xgb_global"
    assert not decision.checks["beats_incumbent_by_margin"]
    assert "margin" in decision.reason


def test_declines_when_high_wait_rides_regress():
    """Better on average, worse where queues are long, is not better."""
    decision = decide(
        [result("baseline", 10.0), result("xgb_local_fleet", 5.0, high_wait_rmse=30.0)],
        result("prophet_fleet", 7.0, high_wait_rmse=20.0),
        CFG,
    )
    assert not decision.promoted
    assert decision.checks["beats_incumbent_by_margin"]
    assert not decision.checks["no_high_wait_regression"]
    assert "high-wait RMSE" in decision.reason


def test_low_coverage_candidate_is_excluded_even_when_most_accurate():
    """The v1 tripwire.

    A model serving only the easy 60% of rides posted a flattering MAE precisely because
    the hard rides had silently fallen out of the evaluated population.
    """
    decision = decide(
        [
            result("baseline", 10.0),
            result("xgb_local_fleet", 2.0, coverage=0.60),
            result("prophet_fleet", 6.0, coverage=1.0),
        ],
        None,
        CFG,
    )
    assert decision.winner == "prophet_fleet"


def test_fails_the_run_when_nothing_beats_the_baseline():
    """Something upstream is broken; shipping is worse than failing loudly."""
    with pytest.raises(NoViableModelError, match="baseline"):
        decide([result("baseline", 5.0), result("prophet_fleet", 8.0)], None, CFG)


def test_fails_when_every_candidate_is_below_the_coverage_floor():
    with pytest.raises(NoViableModelError, match="coverage"):
        decide(
            [result("baseline", 10.0), result("prophet_fleet", 5.0, coverage=0.5)],
            None,
            CFG,
        )


def test_baseline_must_always_be_scored():
    with pytest.raises(ValueError, match="baseline"):
        decide([result("prophet_fleet", 5.0)], None, CFG)


def test_decision_serialises_for_the_promotion_log():
    decision = decide([result("baseline", 10.0), result("prophet_fleet", 6.0)], None, CFG)
    row = decision.as_row()
    assert row["promoted"] is True
    assert row["winner"] == "prophet_fleet"
    assert row["winner_mae"] == 6.0
    assert "decided_at" in row


def test_retention_keeps_the_newest_versions():
    assert versions_to_retire([1, 2, 3, 4, 5, 6], keep=4) == [2, 1]
    assert versions_to_retire([1, 2], keep=4) == []
