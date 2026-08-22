"""The champion/challenger gate.

This is the part that makes the weekly job a deployment rather than a cron entry. Every
Sunday four candidates are retrained and re-scored on the same fresh holdout, *and so is
the incumbent champion*. Re-scoring the incumbent instead of comparing against its
recorded metrics from weeks ago is the whole point: last month's numbers were measured on
last month's data, and comparing across different holdouts would let a quiet week look
like a model improvement.

Promotion has to clear four independent checks. A gate that has never declined a
promotion is not a gate, so the decision -- and its reasoning -- is written to
`gold.promotion_log` whichever way it goes, and surfaced on the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import pipeline

log = logging.getLogger(__name__)

INCUMBENT = "__incumbent__"


@dataclass
class PromotionDecision:
    """The outcome, with enough detail to explain itself in a dashboard row."""

    promoted: bool
    winner: str
    incumbent: str | None
    reason: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_row(self) -> dict:
        return {
            "decided_at": self.decided_at,
            "promoted": self.promoted,
            "winner": self.winner,
            "incumbent": self.incumbent,
            "reason": self.reason,
            "checks_passed": sum(self.checks.values()),
            "checks_total": len(self.checks),
            **{f"check_{k}": v for k, v in self.checks.items()},
            **self.details,
        }

    def __str__(self) -> str:
        verdict = "PROMOTED" if self.promoted else "DECLINED"
        return f"[{verdict}] {self.winner} -- {self.reason}"


class NoViableModelError(RuntimeError):
    """Raised when nothing beats the historical mean. The run must fail, not ship."""


def decide(
    candidate_results: list[dict],
    incumbent_result: dict | None = None,
    settings=None,
) -> PromotionDecision:
    """Rank candidates and decide whether the best one replaces the champion.

    `candidate_results` and `incumbent_result` must all come from `evaluate_model` run
    against the *same* holdout.
    """
    cfg = settings or pipeline()

    by_name = {r["model"]: r for r in candidate_results}
    if "baseline" not in by_name:
        raise ValueError("the baseline must be scored every run; it is the floor")
    baseline = by_name["baseline"]

    # 1. Coverage. The v1 entity-resolution defect showed up as accuracy that improved
    #    while the evaluated population quietly shrank. A model that cannot serve the
    #    fleet is not eligible however good its MAE looks on the part it does serve.
    eligible = [
        r
        for r in candidate_results
        if r["model"] != "baseline" and r["coverage"] >= cfg.min_coverage
    ]
    excluded = [
        r["model"]
        for r in candidate_results
        if r["model"] != "baseline" and r["coverage"] < cfg.min_coverage
    ]
    for name in excluded:
        log.warning(
            "%s excluded: coverage %.3f below %.2f", name, by_name[name]["coverage"], cfg.min_coverage
        )

    if not eligible:
        raise NoViableModelError(
            f"no candidate met the {cfg.min_coverage:.0%} coverage floor "
            f"(excluded: {', '.join(excluded) or 'none trained'})"
        )

    best = min(eligible, key=lambda r: r["overall_mae"])

    # 2. Floor. If nothing beats a per-ride historical mean, something upstream is broken
    #    -- bad ingestion, a corrupted silver build, a feature regression. Shipping in
    #    that state is worse than shipping nothing.
    if best["overall_mae"] >= baseline["overall_mae"]:
        raise NoViableModelError(
            f"best candidate {best['model']} (MAE {best['overall_mae']:.2f}) failed to beat "
            f"the historical-mean baseline (MAE {baseline['overall_mae']:.2f}); "
            "investigate ingestion and the silver build before promoting anything"
        )

    details = {
        "winner_mae": best["overall_mae"],
        "winner_high_wait_rmse": best["high_wait_rides_rmse"],
        "winner_coverage": best["coverage"],
        "baseline_mae": baseline["overall_mae"],
    }

    # 3. First deployment: nothing to regress against.
    if incumbent_result is None:
        return PromotionDecision(
            promoted=True,
            winner=best["model"],
            incumbent=None,
            reason=(
                f"no champion registered; promoting {best['model']} "
                f"(MAE {best['overall_mae']:.2f} vs baseline {baseline['overall_mae']:.2f})"
            ),
            checks={"coverage": True, "beats_baseline": True},
            details=details,
        )

    details["incumbent_mae"] = incumbent_result["overall_mae"]
    details["incumbent_high_wait_rmse"] = incumbent_result["high_wait_rides_rmse"]

    # 4. Beat the incumbent by a real margin, and do not regress where it matters.
    #    Requiring a margin stops the champion from churning on holdout noise -- every
    #    swap invalidates the accuracy history the dashboard is built on.
    mae_target = incumbent_result["overall_mae"] * (1 - cfg.min_improvement)
    rmse_ceiling = incumbent_result["high_wait_rides_rmse"] * (1 + cfg.max_high_wait_regression)

    checks = {
        "coverage": best["coverage"] >= cfg.min_coverage,
        "beats_baseline": True,
        "beats_incumbent_by_margin": best["overall_mae"] <= mae_target,
        "no_high_wait_regression": best["high_wait_rides_rmse"] <= rmse_ceiling,
    }
    details["mae_target"] = mae_target
    details["high_wait_rmse_ceiling"] = rmse_ceiling

    if all(checks.values()):
        improvement = 1 - best["overall_mae"] / incumbent_result["overall_mae"]
        return PromotionDecision(
            promoted=True,
            winner=best["model"],
            incumbent=incumbent_result["model"],
            reason=(
                f"{best['model']} improves MAE {improvement:.1%} over champion "
                f"{incumbent_result['model']} ({incumbent_result['overall_mae']:.2f} -> "
                f"{best['overall_mae']:.2f}) with no high-wait regression"
            ),
            checks=checks,
            details=details,
        )

    failed = [name for name, ok in checks.items() if not ok]
    if "beats_incumbent_by_margin" in failed:
        why = (
            f"MAE {best['overall_mae']:.2f} did not clear the "
            f"{cfg.min_improvement:.0%} margin over champion "
            f"{incumbent_result['model']} ({incumbent_result['overall_mae']:.2f}, "
            f"target <= {mae_target:.2f})"
        )
    else:
        why = (
            f"high-wait RMSE {best['high_wait_rides_rmse']:.2f} exceeds the regression "
            f"ceiling {rmse_ceiling:.2f}"
        )

    return PromotionDecision(
        promoted=False,
        winner=incumbent_result["model"],
        incumbent=incumbent_result["model"],
        reason=f"kept champion {incumbent_result['model']}: challenger {best['model']} {why}",
        checks=checks,
        details=details,
    )


def versions_to_retire(all_versions: list[int], keep: int | None = None) -> list[int]:
    """Older model versions to delete.

    The Prophet fleet is roughly 58 MB per version and Databricks Free Edition enforces
    storage quotas; an unbounded weekly registry fills them within a couple of months.
    """
    keep = pipeline().keep_model_versions if keep is None else keep
    return sorted(all_versions, reverse=True)[keep:]
