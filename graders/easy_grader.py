"""Grader for the easy batch matching task."""

from domain.scenario_models import EpisodeContext

from .base import BaseGrader


class EasyGrader(BaseGrader):
    """Scores the easy task: batch matching + elimination.

    Scoring breakdown:
        Match coverage:        50%  (correct matches / expected matches)
        Elimination coverage:  40%  (eliminations / expected eliminations)
        Efficiency bonus:      10%  (1 - steps_used/step_limit)
        Invalid action penalty: -0.02 per invalid action, capped at -0.20
    """

    def score(self, ctx: EpisodeContext) -> float:
        return self.detailed_report(ctx)["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Match coverage (50%) ---
        correct_matches = self._count_correct_matches(ctx)
        expected_matches = gt.total_expected_matches
        match_ratio = correct_matches / expected_matches if expected_matches > 0 else 1.0
        match_score = 0.50 * match_ratio

        # --- Elimination coverage (40%) ---
        expected_elims = gt.total_expected_eliminations
        actual_elims = len(ctx.eliminations)
        elim_ratio = min(actual_elims, expected_elims) / expected_elims if expected_elims > 0 else 1.0
        elim_score = 0.40 * elim_ratio

        # --- Efficiency bonus (10%) ---
        step_limit = ctx.scenario.step_limit
        step_ratio = ctx.step_count / step_limit if step_limit > 0 else 1.0
        efficiency = max(0.0, 1.0 - step_ratio)
        efficiency_score = 0.10 * efficiency

        # --- Invalid action penalty ---
        penalty = min(0.20, ctx.invalid_action_count * 0.02)

        raw = match_score + elim_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "correct_matches": correct_matches,
            "expected_matches": expected_matches,
            "match_ratio": round(match_ratio, 4),
            "match_score": round(match_score, 4),
            "actual_eliminations": actual_elims,
            "expected_eliminations": expected_elims,
            "elim_ratio": round(elim_ratio, 4),
            "elim_score": round(elim_score, 4),
            "steps_used": ctx.step_count,
            "step_limit": step_limit,
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

    def _count_correct_matches(self, ctx: EpisodeContext) -> int:
        """Count how many of the agent's matches are in the ground truth."""
        gt_pairs: set[tuple[str, str]] = set()
        for pair in ctx.ground_truth.required_matches:
            gt_pairs.add((pair[0], pair[1]))
            gt_pairs.add((pair[1], pair[0]))  # allow either order

        correct = 0
        for match in ctx.matches.values():
            if (match.debit_txn_id, match.credit_txn_id) in gt_pairs:
                correct += 1
        return correct
