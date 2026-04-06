"""Grader for the medium FX variance task."""

from decimal import Decimal

from domain.scenario_models import EpisodeContext

from .base import BaseGrader


class MediumGrader(BaseGrader):
    """Scores the medium task: FX variance resolution.

    Scoring breakdown:
        Evidence gathering:    15%  (fetched relevant documents)
        FX rate query:         20%  (called calculate_fx at least once)
        Adjustment accuracy:   25%  (correct amount, account, entity, reason)
        Match coverage:        20%  (correct matches)
        Elimination coverage:  10%  (eliminations completed)
        Efficiency:            10%  (steps used vs step limit)
        Penalties:
            -0.03 per invalid action (capped at -0.20)
            -0.10 if FX not queried at all (when scenario has FX disputes)
    """

    def score(self, ctx: EpisodeContext) -> float:
        return self.detailed_report(ctx)["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Evidence gathering (15%) ---
        total_docs = len(ctx.scenario.documents)
        fetched_docs = len(ctx.evidence_cache)
        evidence_ratio = min(fetched_docs, total_docs) / total_docs if total_docs > 0 else 1.0
        evidence_score = 0.15 * evidence_ratio

        # --- FX rate query (20%) ---
        fx_needed = bool(gt.required_adjustments)
        if not fx_needed:
            fx_score = 0.20  # Not applicable
        elif ctx.fx_queried:
            fx_score = 0.20  # Full marks — used treasury service
        else:
            fx_score = 0.0

        # --- Adjustment accuracy (25%) ---
        adj_score = self._score_adjustments(ctx)

        # --- Match coverage (20%) ---
        correct_matches = self._count_correct_matches(ctx)
        expected_matches = gt.total_expected_matches
        match_ratio = correct_matches / expected_matches if expected_matches > 0 else 1.0
        match_score = 0.20 * match_ratio

        # --- Elimination coverage (10%) ---
        expected_elims = gt.total_expected_eliminations
        actual_elims = len(ctx.eliminations)
        elim_ratio = min(actual_elims, expected_elims) / expected_elims if expected_elims > 0 else 1.0
        elim_score = 0.10 * elim_ratio

        # --- Efficiency (10%) ---
        step_limit = ctx.scenario.step_limit
        step_ratio = ctx.step_count / step_limit if step_limit > 0 else 1.0
        efficiency = max(0.0, 1.0 - step_ratio)
        efficiency_score = 0.10 * efficiency

        # --- Penalties ---
        penalty = min(0.20, ctx.invalid_action_count * 0.03)
        if fx_needed and not ctx.fx_queried:
            penalty += 0.10  # Did not use treasury service at all

        raw = evidence_score + fx_score + adj_score + match_score + elim_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "evidence_fetched": fetched_docs,
            "evidence_total": total_docs,
            "evidence_score": round(evidence_score, 4),
            "fx_queried": ctx.fx_queried,
            "fx_score": round(fx_score, 4),
            "adjustment_score": round(adj_score, 4),
            "correct_matches": correct_matches,
            "expected_matches": expected_matches,
            "match_score": round(match_score, 4),
            "actual_eliminations": actual_elims,
            "expected_eliminations": expected_elims,
            "elim_score": round(elim_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

    def _score_adjustments(self, ctx: EpisodeContext) -> float:
        """Score adjustments against ground truth (25% total)."""
        expected_adjs = ctx.ground_truth.required_adjustments
        if not expected_adjs:
            return 0.25  # No adjustments needed → full marks

        correct = 0
        for expected in expected_adjs:
            for actual in ctx.adjustments:
                if (
                    actual.entity_id == expected.get("entity_id")
                    and actual.reason_code == expected.get("reason_code")
                    and actual.debit_account_code == expected.get("debit_account_code")
                    and actual.credit_account_code == expected.get("credit_account_code")
                ):
                    # Check amount within tolerance (±$1 for rounding differences)
                    expected_amt = Decimal(str(expected.get("amount", "0")))
                    if abs(actual.money.amount - expected_amt) <= Decimal("1.00"):
                        correct += 1
                        break

        return 0.25 * (correct / len(expected_adjs))

    def _count_correct_matches(self, ctx: EpisodeContext) -> int:
        """Count how many of the agent's matches are in the ground truth."""
        gt_pairs: set[tuple[str, str]] = set()
        for pair in ctx.ground_truth.required_matches:
            gt_pairs.add((pair[0], pair[1]))
            gt_pairs.add((pair[1], pair[0]))

        correct = 0
        for match in ctx.matches.values():
            if (match.debit_txn_id, match.credit_txn_id) in gt_pairs:
                correct += 1
        return correct
