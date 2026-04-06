"""Grader for the hard liability dispute task."""

from decimal import Decimal

from domain.scenario_models import EpisodeContext

from .base import BaseGrader


class HardGrader(BaseGrader):
    """Scores the hard task: adversarial liability dispute resolution.

    Scoring breakdown:
        Evidence gathering:    10%  (fetched shipment report + contract)
        Legal consultation:    20%  (called ask_legal_analyst)
        Liable entity correct: 25%  (posted adjustment to correct entity)
        Adjustment accuracy:   20%  (correct accounts, amount, reason_code)
        Process ordering:      15%  (evidence → legal → adjustment)
        Efficiency:            10%  (steps vs step limit)

    Penalties:
        -0.05 per invalid action (capped at -0.20)
        -0.15 if legal analyst not consulted
        -0.20 if adjustment posted to wrong entity (and no correct-entity adjustment)
    """

    def score(self, ctx: EpisodeContext) -> float:
        return self.detailed_report(ctx)["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Evidence gathering (10%) ---
        required_docs = self._required_doc_ids(ctx)
        fetched_required = len(required_docs & ctx.evidence_cache)
        evidence_ratio = fetched_required / len(required_docs) if required_docs else 1.0
        evidence_score = 0.10 * evidence_ratio

        # --- Legal consultation (20%) ---
        legal_score = 0.20 if ctx.legal_consulted else 0.0

        # --- Liable entity correctness (25%) ---
        liable_entity_correct = False
        if gt.required_liable_entity_id and ctx.adjustments:
            for adj in ctx.adjustments:
                if adj.entity_id == gt.required_liable_entity_id:
                    liable_entity_correct = True
                    break
        liable_score = 0.25 if liable_entity_correct else 0.0

        # --- Adjustment accuracy (20%) ---
        adj_score = self._score_adjustments(ctx)

        # --- Process ordering (15%) ---
        order_score = self._score_process_ordering(ctx)

        # --- Efficiency (10%) ---
        step_limit = ctx.scenario.step_limit
        step_ratio = ctx.step_count / step_limit if step_limit > 0 else 1.0
        efficiency_score = 0.10 * max(0.0, 1.0 - step_ratio)

        # --- Penalties ---
        penalty = min(0.20, ctx.invalid_action_count * 0.05)

        if not ctx.legal_consulted:
            penalty += 0.15  # Skipped legal analyst entirely

        # Wrong-entity penalty: posted to wrong entity without any correct-entity adjustment
        if gt.required_liable_entity_id and ctx.adjustments and not liable_entity_correct:
            wrong_entity_adj_exists = any(
                a.entity_id != gt.required_liable_entity_id for a in ctx.adjustments
            )
            if wrong_entity_adj_exists:
                penalty += 0.20

        raw = evidence_score + legal_score + liable_score + adj_score + order_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "evidence_fetched": fetched_required,
            "evidence_required": len(required_docs),
            "evidence_score": round(evidence_score, 4),
            "legal_consulted": ctx.legal_consulted,
            "legal_score": round(legal_score, 4),
            "liable_entity_correct": liable_entity_correct,
            "liable_score": round(liable_score, 4),
            "adjustment_score": round(adj_score, 4),
            "process_order_score": round(order_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

    def _required_doc_ids(self, ctx: EpisodeContext) -> set[str]:
        """Collect document IDs that the agent must fetch."""
        required: set[str] = set()
        if ctx.legal_truth:
            required.add(ctx.legal_truth.contract_document_id)
        for doc in ctx.documents.values():
            if getattr(doc, "document_type", None) == "shipment_report":
                required.add(doc.document_id)
        return required

    def _score_adjustments(self, ctx: EpisodeContext) -> float:
        """Score adjustments against ground truth (20% total)."""
        expected_adjs = ctx.ground_truth.required_adjustments
        if not expected_adjs:
            return 0.20  # No adjustments needed → full marks

        correct = 0
        for expected in expected_adjs:
            for actual in ctx.adjustments:
                if (
                    actual.entity_id == expected.get("entity_id")
                    and actual.debit_account_code == expected.get("debit_account_code")
                    and actual.credit_account_code == expected.get("credit_account_code")
                    and actual.reason_code == expected.get("reason_code")
                ):
                    expected_amt = Decimal(str(expected.get("amount", "0")))
                    if abs(actual.money.amount - expected_amt) <= Decimal("1.00"):
                        correct += 1
                        break

        return 0.20 * (correct / len(expected_adjs))

    def _score_process_ordering(self, ctx: EpisodeContext) -> float:
        """Score whether the agent followed the correct evidence → legal → action order."""
        audit = ctx.audit_log

        doc_events = [e for e in audit if e.action_type == "fetch_document" and e.status == "ok"]
        legal_events = [e for e in audit if e.action_type == "ask_legal_analyst" and e.status in ("ok", "rejected")]
        adj_events = [e for e in audit if e.action_type == "post_adjustment" and e.status == "ok"]

        if not legal_events:
            # No legal consultation at all — ordering penalty subsumed by legal_consulted penalty
            return 0.0

        score = 0.0

        # Evidence must come before legal consultation
        if doc_events and legal_events[0].timestamp > doc_events[0].timestamp:
            score += 0.075  # Evidence first ✓

        # Legal must come before adjustments
        if legal_events and adj_events and legal_events[0].timestamp < adj_events[0].timestamp:
            score += 0.075  # Legal before action ✓

        return round(score, 4)
