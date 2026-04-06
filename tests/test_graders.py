"""Tests for task graders.

Tests grader scoring logic against synthetically constructed EpisodeContext objects.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from domain.ledger_models import EliminationRecord, LedgerLine, MatchRecord
from domain.money import Money
from domain.scenario_models import EpisodeContext, GroundTruthChecklist, ScenarioBundle
from graders import EasyGrader, HardGrader, MediumGrader, get_grader
from graders.base import BaseGrader


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SEED_DIR = Path(__file__).resolve().parent.parent / "seed_data"


def _load_ctx(scenario_id: str = "smoke", difficulty: str = "easy") -> EpisodeContext:
    """Load a real EpisodeContext from seed data."""
    from server.environment import IntercompanyDisputeEnvironment

    env = IntercompanyDisputeEnvironment()
    env.reset(task_id="easy_batch_matching", scenario_id=scenario_id)
    return env._ctx


def _make_match(match_id: str, d_id: str, c_id: str) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        debit_txn_id=d_id,
        credit_txn_id=c_id,
        matched_at=datetime.now(timezone.utc),
    )


def _make_elim(elim_id: str, match_id: str, entity_id: str = "US_PARENT") -> EliminationRecord:
    return EliminationRecord(
        elimination_id=elim_id,
        entity_id=entity_id,
        matched_pair_id=match_id,
        eliminated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# get_grader dispatch
# ---------------------------------------------------------------------------


class TestGetGrader:
    def test_easy_returns_easy_grader(self):
        g = get_grader("easy")
        assert isinstance(g, EasyGrader)

    def test_unknown_difficulty_raises(self):
        with pytest.raises(ValueError, match="No grader implemented"):
            get_grader("bogus")

    def test_easy_grader_is_base_grader(self):
        g = get_grader("easy")
        assert isinstance(g, BaseGrader)


# ---------------------------------------------------------------------------
# EasyGrader.score — boundary conditions
# ---------------------------------------------------------------------------


class TestEasyGraderScore:
    def _perfect_ctx(self) -> EpisodeContext:
        """All 5 matches correct + 5 eliminations, 10 steps used out of 30."""
        ctx = _load_ctx(scenario_id="smoke")
        gt = ctx.ground_truth
        # Simulate a perfect agent: match all pairs, eliminate all
        for i, pair in enumerate(gt.required_matches):
            d_id, c_id = pair
            match_id = f"MATCH-{i:04d}"
            ctx.matches[match_id] = _make_match(match_id, d_id, c_id)
            elim_id = f"ELIM-{i:04d}"
            ctx.eliminations[elim_id] = _make_elim(elim_id, match_id)
        ctx.step_count = 10
        ctx.invalid_action_count = 0
        return ctx

    def test_perfect_run_scores_above_0_90(self):
        ctx = self._perfect_ctx()
        score = EasyGrader().score(ctx)
        assert score >= 0.90, f"Expected ≥0.90 for perfect run, got {score}"
        assert score <= 1.0

    def test_no_actions_scores_zero(self):
        ctx = _load_ctx(scenario_id="smoke")
        ctx.step_count = 0
        ctx.invalid_action_count = 0
        # No matches, no eliminations
        score = EasyGrader().score(ctx)
        # match_score=0, elim_score=0, efficiency=1.0 → 10% max
        assert score <= 0.10, f"Expected ≤0.10 for no actions, got {score}"

    def test_all_matches_no_eliminations(self):
        ctx = _load_ctx(scenario_id="smoke")
        gt = ctx.ground_truth
        for i, pair in enumerate(gt.required_matches):
            match_id = f"MATCH-{i:04d}"
            ctx.matches[match_id] = _make_match(match_id, pair[0], pair[1])
        ctx.step_count = 20
        ctx.invalid_action_count = 0
        score = EasyGrader().score(ctx)
        # match_score≈0.50, elim_score=0, efficiency=(1-20/30)*0.10
        assert 0.48 <= score <= 0.60, f"Expected ~0.50 for matches-only, got {score}"

    def test_half_matches_half_eliminations(self):
        ctx = _load_ctx(scenario_id="smoke")
        gt = ctx.ground_truth
        half = len(gt.required_matches) // 2
        for i, pair in enumerate(gt.required_matches[:half]):
            match_id = f"MATCH-{i:04d}"
            ctx.matches[match_id] = _make_match(match_id, pair[0], pair[1])
            elim_id = f"ELIM-{i:04d}"
            ctx.eliminations[elim_id] = _make_elim(elim_id, match_id)
        ctx.step_count = 20
        ctx.invalid_action_count = 0
        score = EasyGrader().score(ctx)
        # 2/5 pairs: match≈0.20, elim≈0.16, efficiency≈0.033 → ~0.39
        assert 0.30 <= score <= 0.50, f"Expected ~0.39 for half done, got {score}"

    def test_invalid_actions_reduce_score(self):
        ctx = self._perfect_ctx()
        ctx.invalid_action_count = 5  # 5 * 0.02 = 0.10 penalty
        score_with_invalids = EasyGrader().score(ctx)
        ctx.invalid_action_count = 0
        score_clean = EasyGrader().score(ctx)
        assert score_with_invalids < score_clean
        assert abs(score_clean - score_with_invalids - 0.10) < 0.001

    def test_penalty_capped_at_0_20(self):
        ctx = _load_ctx(scenario_id="smoke")
        ctx.invalid_action_count = 100  # 100 * 0.02 = 2.0 → capped at 0.20
        report = EasyGrader().detailed_report(ctx)
        assert report["penalty"] == 0.20

    def test_score_is_non_negative(self):
        ctx = _load_ctx(scenario_id="smoke")
        ctx.invalid_action_count = 50  # Would exceed 1.0 penalty if uncapped
        score = EasyGrader().score(ctx)
        assert score >= 0.0

    def test_score_within_unit_interval(self):
        ctx = self._perfect_ctx()
        score = EasyGrader().score(ctx)
        assert 0.0 <= score <= 1.0

    def test_wrong_matches_dont_count(self):
        """Matches with wrong pairs don't count toward score."""
        ctx = _load_ctx(scenario_id="smoke")
        gt = ctx.ground_truth
        # Match debit from pair 1 with credit from pair 2 (cross-match — wrong)
        pair_1 = gt.required_matches[0]
        pair_2 = gt.required_matches[1]
        # This would be a wrong match (d from pair1, c from pair2)
        ctx.matches["MATCH-WRONG"] = _make_match("MATCH-WRONG", pair_1[0], pair_2[1])
        ctx.step_count = 10
        ctx.invalid_action_count = 0
        score = EasyGrader().score(ctx)
        # Wrong match doesn't count → match_score = 0
        assert score <= 0.15, f"Wrong matches shouldn't count, got {score}"


# ---------------------------------------------------------------------------
# EasyGrader.detailed_report fields
# ---------------------------------------------------------------------------


class TestEasyGraderReport:
    def test_report_has_all_fields(self):
        ctx = _load_ctx(scenario_id="smoke")
        report = EasyGrader().detailed_report(ctx)
        required_keys = {
            "correct_matches", "expected_matches", "match_ratio", "match_score",
            "actual_eliminations", "expected_eliminations", "elim_ratio", "elim_score",
            "steps_used", "step_limit", "efficiency_score",
            "invalid_actions", "penalty", "raw_score", "final_score",
        }
        assert required_keys.issubset(report.keys()), \
            f"Missing keys: {required_keys - report.keys()}"

    def test_scores_sum_correctly(self):
        ctx = _load_ctx(scenario_id="smoke")
        report = EasyGrader().detailed_report(ctx)
        raw = round(
            report["match_score"] + report["elim_score"] + report["efficiency_score"], 4
        )
        assert abs(raw - report["raw_score"]) < 0.0001

    def test_final_equals_raw_minus_penalty(self):
        ctx = _load_ctx(scenario_id="smoke")
        ctx.invalid_action_count = 2
        report = EasyGrader().detailed_report(ctx)
        expected_final = round(max(0.0, min(1.0, report["raw_score"] - report["penalty"])), 4)
        assert abs(report["final_score"] - expected_final) < 0.0001


# ---------------------------------------------------------------------------
# Benchmark scenario
# ---------------------------------------------------------------------------


class TestBenchmarkScenario:
    def test_benchmark_loads(self):
        ctx = _load_ctx(scenario_id="benchmark")
        assert ctx.ground_truth.total_expected_matches == 20
        assert ctx.ground_truth.total_expected_eliminations == 20
        assert len(ctx.ledger_lines) == 40

    def test_benchmark_perfect_score(self):
        ctx = _load_ctx(scenario_id="benchmark")
        gt = ctx.ground_truth
        for i, pair in enumerate(gt.required_matches):
            match_id = f"MATCH-{i:04d}"
            ctx.matches[match_id] = _make_match(match_id, pair[0], pair[1])
            elim_id = f"ELIM-{i:04d}"
            ctx.eliminations[elim_id] = _make_elim(elim_id, match_id)
        ctx.step_count = 40
        ctx.invalid_action_count = 0
        score = EasyGrader().score(ctx)
        assert score >= 0.88, f"Perfect benchmark run should score ≥0.88, got {score}"


# ---------------------------------------------------------------------------
# Integration: terminal score through environment
# ---------------------------------------------------------------------------


class TestTerminalScoreIntegration:
    def test_easy_grader_called_on_done(self):
        from openenv.core.env_server.mcp_types import CallToolAction
        from server.environment import IntercompanyDisputeEnvironment

        env = IntercompanyDisputeEnvironment()
        env.reset(task_id="easy_batch_matching", scenario_id="smoke")

        # Force episode end by exhausting steps
        ctx = env._ctx
        ctx.step_count = ctx.scenario.step_limit - 1
        obs = env.step(CallToolAction(tool_name="query_open_items", arguments={}))

        assert obs.done is True
        assert "terminal_task_score" in (obs.metadata or {})
        score = obs.metadata["terminal_task_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# MediumGrader tests
# ---------------------------------------------------------------------------


def _load_medium_ctx(scenario_id: str = "smoke") -> EpisodeContext:
    from server.environment import IntercompanyDisputeEnvironment
    env = IntercompanyDisputeEnvironment()
    env.reset(task_id="medium_fx_variance", scenario_id=scenario_id)
    return env._ctx


class TestGetGraderMedium:
    def test_medium_returns_medium_grader(self):
        g = get_grader("medium")
        assert isinstance(g, MediumGrader)


class TestMediumGraderScore:
    def _perfect_ctx(self) -> EpisodeContext:
        """Simulate a perfect agent: evidence + FX + correct adjustments + all matches/elims."""
        ctx = _load_medium_ctx("smoke")
        gt = ctx.ground_truth
        # Fetch all docs (evidence)
        for doc_id in ctx.documents:
            ctx.evidence_cache.add(doc_id)
        ctx.fx_queried = True

        # Post correct adjustments
        from datetime import datetime, timezone
        from domain.ledger_models import JournalEntry
        from domain.money import Money
        from decimal import Decimal

        for i, expected_adj in enumerate(gt.required_adjustments):
            entry_id = f"ADJ-PERF-{i:03d}"
            money = Money(amount=Decimal(str(expected_adj["amount"])), currency=expected_adj["currency"])
            journal = JournalEntry(
                entry_id=entry_id,
                entity_id=expected_adj["entity_id"],
                debit_account_code=expected_adj["debit_account_code"],
                credit_account_code=expected_adj["credit_account_code"],
                money=money,
                reason_code=expected_adj["reason_code"],
                evidence_refs=[],
                posted_at=datetime.now(timezone.utc),
            )
            ctx.adjustments.append(journal)

        # Match all pairs
        for i, pair in enumerate(gt.required_matches):
            match_id = f"MATCH-{i:04d}"
            ctx.matches[match_id] = _make_match(match_id, pair[0], pair[1])
            elim_id = f"ELIM-{i:04d}"
            ctx.eliminations[elim_id] = _make_elim(elim_id, match_id)

        ctx.step_count = 30
        ctx.invalid_action_count = 0
        return ctx

    def test_perfect_run_scores_above_0_88(self):
        ctx = self._perfect_ctx()
        score = MediumGrader().score(ctx)
        assert score >= 0.88, f"Expected ≥0.88 for perfect medium run, got {score}"
        assert score <= 1.0

    def test_no_actions_scores_low(self):
        ctx = _load_medium_ctx("smoke")
        ctx.step_count = 0
        ctx.invalid_action_count = 0
        score = MediumGrader().score(ctx)
        # No evidence, no FX, no adjustments, no matches, no elims → 0 + efficiency
        # Evidence=0, FX=0 (but penalty +0.10), adj=0, match=0, elim=0, efficiency~0.10
        assert score <= 0.10, f"Expected ≤0.10 for no actions, got {score}"

    def test_no_fx_query_penalized(self):
        ctx = _load_medium_ctx("smoke")
        ctx.fx_queried = False
        report = MediumGrader().detailed_report(ctx)
        # Penalty should include +0.10 for not querying FX
        assert report["penalty"] >= 0.10

    def test_fx_queried_no_penalty(self):
        ctx = _load_medium_ctx("smoke")
        ctx.fx_queried = True
        ctx.invalid_action_count = 0
        report = MediumGrader().detailed_report(ctx)
        assert report["fx_score"] == 0.20

    def test_correct_adjustment_gets_credit(self):
        """A single correct adjustment should score 1/3 of 25%."""
        ctx = _load_medium_ctx("smoke")
        gt = ctx.ground_truth
        from datetime import datetime, timezone
        from domain.ledger_models import JournalEntry
        from domain.money import Money
        from decimal import Decimal

        expected_adj = gt.required_adjustments[0]
        money = Money(amount=Decimal(str(expected_adj["amount"])), currency=expected_adj["currency"])
        journal = JournalEntry(
            entry_id="ADJ-TEST-001",
            entity_id=expected_adj["entity_id"],
            debit_account_code=expected_adj["debit_account_code"],
            credit_account_code=expected_adj["credit_account_code"],
            money=money,
            reason_code=expected_adj["reason_code"],
            evidence_refs=[],
            posted_at=datetime.now(timezone.utc),
        )
        ctx.adjustments.append(journal)
        report = MediumGrader().detailed_report(ctx)
        n = gt.total_expected_adjustments  # 3 for smoke
        expected_score = round(0.25 * (1 / n), 4)
        assert abs(report["adjustment_score"] - expected_score) < 0.01

    def test_score_within_unit_interval(self):
        ctx = self._perfect_ctx()
        score = MediumGrader().score(ctx)
        assert 0.0 <= score <= 1.0

    def test_medium_benchmark_loads(self):
        ctx = _load_medium_ctx("benchmark")
        assert ctx.ground_truth.total_expected_matches == 10
        assert len(ctx.ledger_lines) == 20

    def test_report_has_required_fields(self):
        ctx = _load_medium_ctx("smoke")
        report = MediumGrader().detailed_report(ctx)
        required = {
            "evidence_fetched", "evidence_total", "evidence_score",
            "fx_queried", "fx_score", "adjustment_score",
            "correct_matches", "expected_matches", "match_score",
            "actual_eliminations", "expected_eliminations", "elim_score",
            "efficiency_score", "invalid_actions", "penalty", "raw_score", "final_score",
        }
        assert required.issubset(report.keys())


# ---------------------------------------------------------------------------
# HardGrader tests
# ---------------------------------------------------------------------------


def _load_hard_ctx(scenario_id: str = "smoke") -> EpisodeContext:
    from server.environment import IntercompanyDisputeEnvironment
    env = IntercompanyDisputeEnvironment()
    env.reset(task_id="hard_liability_dispute", scenario_id=scenario_id)
    return env._ctx


def _post_adj_to_ctx(ctx: EpisodeContext, entity_id: str, debit: str, credit: str,
                     amount: str, currency: str, reason: str) -> None:
    from datetime import datetime, timezone
    from decimal import Decimal
    from domain.ledger_models import JournalEntry
    from domain.money import Money
    money = Money(amount=Decimal(amount), currency=currency)
    journal = JournalEntry(
        entry_id=f"ADJ-{len(ctx.adjustments):04d}",
        entity_id=entity_id,
        debit_account_code=debit,
        credit_account_code=credit,
        money=money,
        reason_code=reason,
        evidence_refs=[],
        posted_at=datetime.now(timezone.utc),
    )
    ctx.adjustments.append(journal)


class TestGetGraderHard:
    def test_hard_returns_hard_grader(self):
        g = get_grader("hard")
        assert isinstance(g, HardGrader)


class TestHardGraderScore:
    def _perfect_ctx(self) -> EpisodeContext:
        """Simulate perfect hard agent: fetch docs, consult legal, correct adjustment."""
        from datetime import datetime, timezone
        from domain.ledger_models import AuditEvent
        ctx = _load_hard_ctx("smoke")
        now = datetime.now(timezone.utc)

        # Fetch all docs
        for doc_id in ctx.documents:
            ctx.evidence_cache.add(doc_id)

        # Log doc fetch events (timestamp ordering matters)
        for doc_id in list(ctx.documents.keys())[:2]:
            ctx.audit_log.append(AuditEvent(
                timestamp=now,
                actor="orchestrator",
                action_type="fetch_document",
                status="ok",
                detail=doc_id,
            ))

        # Consult legal
        ctx.legal_consulted = True
        import time; time.sleep(0.001)
        from datetime import datetime, timezone
        ctx.audit_log.append(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor="orchestrator",
            action_type="ask_legal_analyst",
            status="ok",
            detail="DOC-H-002",
        ))

        # Post correct adjustment
        import time; time.sleep(0.001)
        _post_adj_to_ctx(ctx, "DE_SUB", "5100", "1300", "50000.00", "EUR", "inventory_loss")
        ctx.audit_log.append(AuditEvent(
            timestamp=datetime.now(timezone.utc),
            actor="orchestrator",
            action_type="post_adjustment",
            status="ok",
            detail="DE_SUB",
        ))

        ctx.step_count = 10
        ctx.invalid_action_count = 0
        return ctx

    def test_perfect_run_scores_above_0_80(self):
        ctx = self._perfect_ctx()
        score = HardGrader().score(ctx)
        assert score >= 0.80, f"Expected ≥0.80 for perfect hard run, got {score}"
        assert score <= 1.0

    def test_no_legal_consultation_heavy_penalty(self):
        ctx = _load_hard_ctx("smoke")
        ctx.legal_consulted = False
        # Post correct adjustment anyway (guessed)
        _post_adj_to_ctx(ctx, "DE_SUB", "5100", "1300", "50000.00", "EUR", "inventory_loss")
        ctx.step_count = 10
        report = HardGrader().detailed_report(ctx)
        # Penalty of 0.15 for skipping legal
        assert report["penalty"] >= 0.15

    def test_wrong_entity_penalized(self):
        ctx = _load_hard_ctx("smoke")
        ctx.legal_consulted = True
        # Post to wrong entity (UK_SUB instead of DE_SUB)
        _post_adj_to_ctx(ctx, "UK_SUB", "5100", "1300", "50000.00", "EUR", "inventory_loss")
        ctx.step_count = 10
        report = HardGrader().detailed_report(ctx)
        assert not report["liable_entity_correct"]
        # Wrong entity penalty
        assert report["penalty"] >= 0.20

    def test_correct_entity_gets_liable_score(self):
        ctx = _load_hard_ctx("smoke")
        ctx.legal_consulted = True
        _post_adj_to_ctx(ctx, "DE_SUB", "5100", "1300", "50000.00", "EUR", "inventory_loss")
        report = HardGrader().detailed_report(ctx)
        assert report["liable_entity_correct"]
        assert report["liable_score"] == 0.25

    def test_correct_adjustment_scores(self):
        ctx = _load_hard_ctx("smoke")
        ctx.legal_consulted = True
        _post_adj_to_ctx(ctx, "DE_SUB", "5100", "1300", "50000.00", "EUR", "inventory_loss")
        report = HardGrader().detailed_report(ctx)
        assert report["adjustment_score"] == 0.20

    def test_no_legal_no_evidence_scores_zero(self):
        ctx = _load_hard_ctx("smoke")
        ctx.step_count = 0
        ctx.invalid_action_count = 0
        score = HardGrader().score(ctx)
        # evidence=0, legal=0 (-0.15 penalty), adj=0, order=0, efficiency~0.10
        assert score <= 0.05, f"Expected ≤0.05 with no actions, got {score}"

    def test_score_within_unit_interval(self):
        ctx = self._perfect_ctx()
        score = HardGrader().score(ctx)
        assert 0.0 <= score <= 1.0

    def test_hard_benchmark_loads(self):
        ctx = _load_hard_ctx("benchmark")
        assert ctx.ground_truth.total_expected_adjustments == 3
        assert len(ctx.documents) == 6

    def test_report_has_required_fields(self):
        ctx = _load_hard_ctx("smoke")
        report = HardGrader().detailed_report(ctx)
        required = {
            "evidence_fetched", "evidence_required", "evidence_score",
            "legal_consulted", "legal_score",
            "liable_entity_correct", "liable_score",
            "adjustment_score", "process_order_score",
            "efficiency_score", "invalid_actions", "penalty", "raw_score", "final_score",
        }
        assert required.issubset(report.keys())
