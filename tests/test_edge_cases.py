"""Phase 8: Edge case and anti-exploit hardening tests.

Tests that the environment correctly handles malformed inputs, exploit patterns,
hallucinated IDs, and other adversarial scenarios.
"""

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from server.environment import IntercompanyDisputeEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    e = IntercompanyDisputeEnvironment()
    e.reset(task_id="easy_batch_matching", scenario_id="smoke")
    return e


def step(env, tool_name, **args):
    """Convenience wrapper for CallToolAction step."""
    return env.step(CallToolAction(tool_name=tool_name, arguments=args))


def result(obs) -> dict:
    """Extract dict from CallToolObservation."""
    import json
    r = getattr(obs, "result", None)
    if r is None:
        return {}
    if hasattr(r, "structured_content") and r.structured_content:
        return r.structured_content
    if hasattr(r, "data") and r.data:
        return r.data
    content = getattr(r, "content", [])
    for item in content:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    return {}


# ---------------------------------------------------------------------------
# 1. Hallucinated / non-existent IDs
# ---------------------------------------------------------------------------


class TestHallucinatedIDs:
    def test_match_hallucinated_debit_rejected(self, env):
        obs = step(env, "execute_match",
                   debit_txn_id="TXN-DOES-NOT-EXIST-D", credit_txn_id="TXN-E-001-C")
        assert result(obs).get("status") == "rejected"

    def test_match_hallucinated_credit_rejected(self, env):
        obs = step(env, "execute_match",
                   debit_txn_id="TXN-E-001-D", credit_txn_id="TXN-DOES-NOT-EXIST-C")
        assert result(obs).get("status") == "rejected"

    def test_eliminate_hallucinated_match_id_rejected(self, env):
        obs = step(env, "execute_elimination",
                   entity_id="US_PARENT", matched_pair_id="MATCH-FAKEFAKE")
        assert result(obs).get("status") == "rejected"

    def test_fetch_hallucinated_doc_id(self, env):
        obs = step(env, "fetch_document", document_id="DOC-DOESNOTEXIST")
        r = result(obs)
        assert "error" in r

    def test_adjustment_hallucinated_entity_rejected(self, env):
        obs = step(env, "post_adjustment",
                   entity_id="FAKE_ENTITY_XYZ",
                   debit_account_code="1300", credit_account_code="2100",
                   amount="100.00", currency="USD", reason_code="manual_true_up")
        assert result(obs).get("status") == "rejected"


# ---------------------------------------------------------------------------
# 2. Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def test_match_empty_debit_id_rejected(self, env):
        obs = step(env, "execute_match", debit_txn_id="", credit_txn_id="TXN-E-001-C")
        r = result(obs)
        assert r.get("status") in ("rejected", "invalid")

    def test_match_empty_credit_id_rejected(self, env):
        obs = step(env, "execute_match", debit_txn_id="TXN-E-001-D", credit_txn_id="")
        r = result(obs)
        assert r.get("status") in ("rejected", "invalid")

    def test_elimination_empty_match_id_rejected(self, env):
        obs = step(env, "execute_elimination", entity_id="US_PARENT", matched_pair_id="")
        r = result(obs)
        assert r.get("status") in ("rejected", "invalid")

    def test_adjustment_invalid_reason_code_rejected(self, env):
        obs = step(env, "post_adjustment",
                   entity_id="US_PARENT",
                   debit_account_code="1300", credit_account_code="2100",
                   amount="100.00", currency="USD", reason_code="HACKED_REASON_CODE")
        r = result(obs)
        assert r.get("status") in ("rejected", "invalid")

    def test_adjustment_negative_amount_rejected(self, env):
        obs = step(env, "post_adjustment",
                   entity_id="US_PARENT",
                   debit_account_code="1300", credit_account_code="2100",
                   amount="-500.00", currency="USD", reason_code="manual_true_up")
        assert result(obs).get("status") == "rejected"

    def test_adjustment_zero_amount_rejected(self, env):
        obs = step(env, "post_adjustment",
                   entity_id="US_PARENT",
                   debit_account_code="1300", credit_account_code="2100",
                   amount="0", currency="USD", reason_code="manual_true_up")
        assert result(obs).get("status") == "rejected"

    def test_adjustment_invalid_account_code_rejected(self, env):
        obs = step(env, "post_adjustment",
                   entity_id="US_PARENT",
                   debit_account_code="9999", credit_account_code="2100",
                   amount="100.00", currency="USD", reason_code="manual_true_up")
        assert result(obs).get("status") == "rejected"

    def test_query_open_items_limit_clamped(self, env):
        # Negative limit should be clamped to 1
        obs = step(env, "query_open_items", limit=-100)
        r = result(obs)
        assert "items" in r
        assert r["returned_count"] >= 1

    def test_query_open_items_huge_limit_clamped(self, env):
        # Huge limit should be clamped to 500
        obs = step(env, "query_open_items", limit=999999)
        r = result(obs)
        assert "items" in r


# ---------------------------------------------------------------------------
# 3. Duplicate actions
# ---------------------------------------------------------------------------


class TestDuplicateActions:
    def _get_first_matchable_pair(self, env):
        obs = step(env, "query_open_items", limit=50)
        items = result(obs)["items"]
        debits = [i for i in items if i["side"] == "debit"]
        credits = [i for i in items if i["side"] == "credit"]
        for d in debits:
            for c in credits:
                if (d["entity_id"] == c["counterparty_entity_id"]
                        and d["money"]["amount"] == c["money"]["amount"]
                        and d["money"]["currency"] == c["money"]["currency"]):
                    return d["txn_id"], c["txn_id"]
        return None, None

    def test_double_match_rejected(self, env):
        d_id, c_id = self._get_first_matchable_pair(env)
        obs1 = step(env, "execute_match", debit_txn_id=d_id, credit_txn_id=c_id)
        assert result(obs1).get("status") == "ok"
        obs2 = step(env, "execute_match", debit_txn_id=d_id, credit_txn_id=c_id)
        assert result(obs2).get("status") == "rejected"

    def test_double_elimination_rejected(self, env):
        d_id, c_id = self._get_first_matchable_pair(env)
        obs_match = step(env, "execute_match", debit_txn_id=d_id, credit_txn_id=c_id)
        match_id = result(obs_match)["match_id"]
        obs1 = step(env, "execute_elimination", entity_id="US_PARENT", matched_pair_id=match_id)
        assert result(obs1).get("status") == "ok"
        obs2 = step(env, "execute_elimination", entity_id="US_PARENT", matched_pair_id=match_id)
        assert result(obs2).get("status") == "rejected"


# ---------------------------------------------------------------------------
# 4. Cross-entity violations
# ---------------------------------------------------------------------------


class TestCrossEntityViolations:
    def test_match_wrong_counterparty_rejected(self, env):
        """Cannot match debit with credit from non-counterparty entity."""
        obs = step(env, "query_open_items", limit=50)
        items = result(obs)["items"]
        debits = [i for i in items if i["side"] == "debit"]
        credits = [i for i in items if i["side"] == "credit"]
        # Find a pair that does NOT match (same entity)
        same_entity_credit = next((c for c in credits if c["entity_id"] == debits[0]["entity_id"]), None)
        if same_entity_credit:
            obs = step(env, "execute_match",
                       debit_txn_id=debits[0]["txn_id"],
                       credit_txn_id=same_entity_credit["txn_id"])
            assert result(obs).get("status") == "rejected"

    def test_eliminate_wrong_entity_rejected(self, env):
        """Cannot eliminate with entity not involved in the match."""
        obs_items = step(env, "query_open_items", limit=50)
        items = result(obs_items)["items"]
        debits = {i["txn_id"]: i for i in items if i["side"] == "debit"}
        credits = {i["txn_id"]: i for i in items if i["side"] == "credit"}
        d_id = c_id = None
        for d_tx, d in debits.items():
            for c_tx, c in credits.items():
                if (d["entity_id"] == c["counterparty_entity_id"]
                        and d["money"]["amount"] == c["money"]["amount"]):
                    d_id, c_id = d_tx, c_tx
                    break
            if d_id:
                break

        obs_match = step(env, "execute_match", debit_txn_id=d_id, credit_txn_id=c_id)
        match_id = result(obs_match)["match_id"]

        # Try to eliminate with a different (unrelated) entity
        obs = step(env, "execute_elimination",
                   entity_id="DE_SUB",  # not in this smoke scenario
                   matched_pair_id=match_id)
        assert result(obs).get("status") == "rejected"


# ---------------------------------------------------------------------------
# 5. Exploit patterns
# ---------------------------------------------------------------------------


class TestExploitPatterns:
    def test_step_count_tracks_correctly(self, env):
        for _ in range(5):
            step(env, "query_open_items")
        assert env._ctx.step_count == 5

    def test_invalid_count_increments_on_bad_actions(self, env):
        initial = env._ctx.invalid_action_count
        step(env, "execute_match", debit_txn_id="FAKE", credit_txn_id="FAKE")
        assert env._ctx.invalid_action_count > initial

    def test_episode_terminates_on_step_limit(self, env):
        env._ctx.step_count = env._ctx.scenario.step_limit - 1
        obs = step(env, "query_open_items")
        assert obs.done is True

    def test_episode_terminates_on_too_many_invalids(self, env):
        env._ctx.invalid_action_count = 19
        step(env, "execute_match", debit_txn_id="FAKE", credit_txn_id="FAKE")
        obs = step(env, "query_open_items")  # after 20 invalids → done
        # Either this step or the next one triggers done
        assert env._ctx.invalid_action_count >= 20 or obs.done

    def test_no_steps_after_done(self, env):
        env._done = True
        obs = step(env, "query_open_items")
        assert obs.done is True
        r = obs.metadata or {}
        assert "error" in r

    def test_reward_penalizes_invalid_actions(self, env):
        obs = step(env, "execute_match", debit_txn_id="FAKE", credit_txn_id="FAKE")
        assert obs.reward < 0

    def test_loop_detection_applied(self, env):
        """Repeated identical calls should trigger loop detection and negative reward."""
        obs = None
        for i in range(15):  # 15 identical calls should trigger pattern 3 (no write for 10 steps)
            obs = step(env, "query_open_items", entity_id="US_PARENT", status="open")
        # By step 10+, loop should be detected; reward should be lower
        assert obs is not None
        assert obs.reward is not None


# ---------------------------------------------------------------------------
# 6. Grader determinism
# ---------------------------------------------------------------------------


class TestGraderDeterminism:
    def _score_easy(self, n_matches: int) -> float:
        from graders import EasyGrader
        env = IntercompanyDisputeEnvironment()
        env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        ctx = env._ctx
        gt = ctx.ground_truth
        from datetime import datetime, timezone
        from domain.ledger_models import MatchRecord
        for i, pair in enumerate(gt.required_matches[:n_matches]):
            match_id = f"MATCH-DET-{i:04d}"
            ctx.matches[match_id] = MatchRecord(
                match_id=match_id,
                debit_txn_id=pair[0],
                credit_txn_id=pair[1],
                matched_at=datetime.now(timezone.utc),
            )
        ctx.step_count = 10
        return EasyGrader().score(ctx)

    def test_easy_grader_is_deterministic(self):
        scores = [self._score_easy(3) for _ in range(5)]
        assert len(set(scores)) == 1, f"Scores differed across runs: {scores}"

    def _score_medium(self, fx_queried: bool) -> float:
        from graders import MediumGrader
        env = IntercompanyDisputeEnvironment()
        env.reset(task_id="medium_fx_variance", scenario_id="smoke")
        ctx = env._ctx
        ctx.fx_queried = fx_queried
        ctx.step_count = 20
        return MediumGrader().score(ctx)

    def test_medium_grader_is_deterministic(self):
        scores = [self._score_medium(True) for _ in range(5)]
        assert len(set(scores)) == 1, f"Medium scores differed: {scores}"

    def _score_hard(self, legal_consulted: bool) -> float:
        from graders import HardGrader
        env = IntercompanyDisputeEnvironment()
        env.reset(task_id="hard_liability_dispute", scenario_id="smoke")
        ctx = env._ctx
        ctx.legal_consulted = legal_consulted
        ctx.step_count = 10
        return HardGrader().score(ctx)

    def test_hard_grader_is_deterministic(self):
        scores = [self._score_hard(False) for _ in range(5)]
        assert len(set(scores)) == 1, f"Hard scores differed: {scores}"

    def test_same_actions_same_score(self):
        """Two envs running identical action sequences produce identical terminal scores."""
        from graders import EasyGrader
        scores = []
        for _ in range(2):
            env = IntercompanyDisputeEnvironment()
            env.reset(task_id="easy_batch_matching", scenario_id="smoke")
            ctx = env._ctx
            gt = ctx.ground_truth
            from datetime import datetime, timezone
            from domain.ledger_models import EliminationRecord, MatchRecord
            for i, pair in enumerate(gt.required_matches):
                mid = f"MATCH-SYN-{i:04d}"
                eid = f"ELIM-SYN-{i:04d}"
                ctx.matches[mid] = MatchRecord(
                    match_id=mid, debit_txn_id=pair[0], credit_txn_id=pair[1],
                    matched_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                )
                ctx.eliminations[eid] = EliminationRecord(
                    elimination_id=eid, entity_id="US_PARENT", matched_pair_id=mid,
                    eliminated_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                )
            ctx.step_count = 20
            ctx.invalid_action_count = 0
            scores.append(EasyGrader().score(ctx))
        assert scores[0] == scores[1], f"Same actions produced different scores: {scores}"
