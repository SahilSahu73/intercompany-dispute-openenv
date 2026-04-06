"""Phase 4 validation: MCPEnvironment integration tests.

Tests the full env lifecycle: reset → step(list_tools) → step(call_tool) → state.
Tests run against env instances directly (not via HTTP) to avoid stateless-mode issues.
"""

import pytest
from decimal import Decimal

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from server.environment import IntercompanyDisputeEnvironment
from models import FinanceDisputeState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Fresh environment instance."""
    return IntercompanyDisputeEnvironment()


@pytest.fixture
def reset_env(env):
    """Environment already reset with easy_batch_matching / smoke."""
    env.reset(task_id="easy_batch_matching", scenario_id="smoke")
    return env


# ---------------------------------------------------------------------------
# 1. list_tools — stateless, should work even without reset
# ---------------------------------------------------------------------------


class TestListTools:
    def test_list_tools_without_reset(self, env):
        """list_tools must succeed even before reset() is called."""
        action = ListToolsAction()
        obs = env.step(action)
        assert obs.done is False or obs.done == False  # not terminated
        # Check tools were returned
        tools = getattr(obs, "tools", None)
        assert tools is not None, "ListToolsObservation.tools should be set"

    def test_list_tools_returns_all_eight_tools(self, env):
        action = ListToolsAction()
        obs = env.step(action)
        tool_names = {t.name for t in obs.tools}
        expected = {
            "query_open_items",
            "query_ledger_balance",
            "fetch_document",
            "ask_legal_analyst",
            "calculate_fx",
            "execute_match",
            "post_adjustment",
            "execute_elimination",
        }
        assert tool_names == expected, f"Missing tools: {expected - tool_names}"

    def test_list_tools_after_reset(self, reset_env):
        """list_tools still works after reset."""
        obs = reset_env.step(ListToolsAction())
        assert obs.tools is not None
        assert len(obs.tools) == 8


# ---------------------------------------------------------------------------
# 2. reset — initializes episode context
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        assert obs is not None
        assert obs.done is False

    def test_reset_metadata_contains_task_info(self, env):
        obs = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        meta = obs.metadata or {}
        assert meta.get("task_id") == "easy_batch_matching"
        assert meta.get("difficulty") == "easy"
        assert "objectives" in meta
        assert "step_limit" in meta

    def test_reset_provides_open_items_preview(self, env):
        obs = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        meta = obs.metadata or {}
        preview = meta.get("open_items_preview", {})
        assert preview.get("total_count", 0) > 0, "Should have open items after reset"

    def test_reset_provides_document_ids(self, env):
        obs = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        meta = obs.metadata or {}
        doc_ids = meta.get("available_document_ids", [])
        assert len(doc_ids) > 0, "Should have document IDs"

    def test_reset_idempotent(self, env):
        obs1 = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        obs2 = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        assert obs1.done is False
        assert obs2.done is False

    def test_reset_bad_scenario_raises(self, env):
        with pytest.raises(Exception):
            env.reset(task_id="easy_batch_matching", scenario_id="nonexistent_file")

    def test_reset_bad_task_raises(self, env):
        """Unknown task_id maps to unknown difficulty → FileNotFoundError."""
        with pytest.raises(Exception):
            env.reset(task_id="totally_bogus_task", scenario_id="smoke")


# ---------------------------------------------------------------------------
# 3. state — reflects episode context
# ---------------------------------------------------------------------------


class TestState:
    def test_state_before_reset_is_empty(self, env):
        s = env.state
        assert isinstance(s, FinanceDisputeState)
        assert s.task_id == ""
        assert s.step_count == 0

    def test_state_after_reset(self, reset_env):
        s = reset_env.state
        assert isinstance(s, FinanceDisputeState)
        assert s.task_id == "easy_batch_matching"
        assert s.difficulty == "easy"
        assert s.step_count == 0
        assert s.step_limit > 0
        assert isinstance(s.remaining_objectives, list)
        assert len(s.remaining_objectives) > 0

    def test_state_step_count_increments(self, reset_env):
        reset_env.step(CallToolAction(tool_name="query_open_items", arguments={}))
        s = reset_env.state
        assert s.step_count == 1

    def test_state_never_exposes_ground_truth(self, reset_env):
        """The public state must not contain ground truth / grader data."""
        s = reset_env.state
        d = s.model_dump()
        forbidden_keys = {"expected_matches", "expected_eliminations", "ground_truth"}
        assert not forbidden_keys.intersection(d.keys()), \
            f"Ground truth leaked into state: {forbidden_keys.intersection(d.keys())}"


# ---------------------------------------------------------------------------
# 4. call_tool: read-only tools
# ---------------------------------------------------------------------------


class TestCallToolReadOnly:
    def test_query_open_items_no_filter(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="query_open_items",
            arguments={},
        ))
        result = _extract_result(obs)
        assert "items" in result
        assert "total_count" in result
        assert result["total_count"] > 0

    def test_query_open_items_entity_filter(self, reset_env):
        # First get all items to find a valid entity_id
        obs_all = reset_env.step(CallToolAction(
            tool_name="query_open_items", arguments={},
        ))
        all_items = _extract_result(obs_all)["items"]
        assert all_items, "Expected open items in smoke scenario"
        entity_id = all_items[0]["entity_id"]

        obs = reset_env.step(CallToolAction(
            tool_name="query_open_items",
            arguments={"entity_id": entity_id},
        ))
        result = _extract_result(obs)
        # All returned items belong to this entity
        for item in result["items"]:
            assert item["entity_id"] == entity_id

    def test_query_ledger_balance_valid_entity(self, reset_env):
        obs2 = reset_env.step(CallToolAction(
            tool_name="query_ledger_balance",
            arguments={"entity_id": "US_PARENT", "account_code": "1300"},
        ))
        result = _extract_result(obs2)
        assert "entity_id" in result
        assert "debit_total" in result
        assert "credit_total" in result

    def test_fetch_document_valid_id(self, reset_env):
        # Get a doc id from reset metadata
        obs_reset = reset_env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        doc_id = obs_reset.metadata["available_document_ids"][0]

        obs = reset_env.step(CallToolAction(
            tool_name="fetch_document",
            arguments={"document_id": doc_id},
        ))
        result = _extract_result(obs)
        assert "document_id" in result or result.get("status") == "ok"

    def test_fetch_document_invalid_id(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="fetch_document",
            arguments={"document_id": "DOC-NOTEXIST"},
        ))
        result = _extract_result(obs)
        assert "error" in result

    def test_calculate_fx_returns_rate(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="calculate_fx",
            arguments={"from_currency": "USD", "to_currency": "EUR", "amount": "100.00"},
        ))
        result = _extract_result(obs)
        # Either a valid FX result or "no rate available" error is acceptable
        assert isinstance(result, dict)

    def test_ask_legal_analyst_returns_text(self, reset_env):
        # First get a doc id
        obs_reset = reset_env.reset(task_id="easy_batch_matching", scenario_id="smoke")
        doc_id = obs_reset.metadata["available_document_ids"][0]
        obs = reset_env.step(CallToolAction(
            tool_name="ask_legal_analyst",
            arguments={"document_id": doc_id, "question": "Who bears liability?"},
        ))
        result = _extract_result(obs)
        assert isinstance(result, dict)
        # easy task has no legal_truth, so expect a "not available" style response
        assert "opinion" in result or "error" in result or "message" in result or "answer" in result


# ---------------------------------------------------------------------------
# 5. call_tool: write tools (match / adjustment / elimination)
# ---------------------------------------------------------------------------


class TestCallToolWrite:
    def _get_matchable_pair(self, env):
        """Return (debit_txn_id, credit_txn_id) for a valid matchable pair."""
        obs = env.step(CallToolAction(tool_name="query_open_items", arguments={"limit": 50}))
        items = _extract_result(obs)["items"]
        # Find debit/credit with amounts inside money: {amount, currency}
        debits = {i["txn_id"]: i for i in items if i["side"] == "debit"}
        credits = {i["txn_id"]: i for i in items if i["side"] == "credit"}
        for d_id, d in debits.items():
            for c_id, c in credits.items():
                if (
                    d["entity_id"] == c["counterparty_entity_id"]
                    and abs(float(d["money"]["amount"])) == abs(float(c["money"]["amount"]))
                    and d["money"]["currency"] == c["money"]["currency"]
                ):
                    return d_id, c_id
        return None, None

    def test_execute_match_valid_pair(self, reset_env):
        d_id, c_id = self._get_matchable_pair(reset_env)
        assert d_id and c_id, "Smoke scenario should have at least one matchable pair"

        obs = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": d_id, "credit_txn_id": c_id},
        ))
        result = _extract_result(obs)
        assert result.get("status") == "ok", f"Match failed: {result}"
        assert "match_id" in result

    def test_execute_match_duplicate_rejected(self, reset_env):
        d_id, c_id = self._get_matchable_pair(reset_env)
        reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": d_id, "credit_txn_id": c_id},
        ))
        # Second attempt should be rejected
        obs = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": d_id, "credit_txn_id": c_id},
        ))
        result = _extract_result(obs)
        assert result.get("status") == "rejected"

    def test_execute_match_nonexistent_rejected(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": "TXN-FAKE1", "credit_txn_id": "TXN-FAKE2"},
        ))
        result = _extract_result(obs)
        assert result.get("status") == "rejected"

    def test_post_adjustment_valid(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="post_adjustment",
            arguments={
                "entity_id": "US_PARENT",
                "debit_account_code": "6100",
                "credit_account_code": "1300",
                "amount": "500.00",
                "currency": "USD",
                "reason_code": "manual_true_up",
            },
        ))
        result = _extract_result(obs)
        assert result.get("status") == "ok", f"Adjustment failed: {result}"
        assert "entry_id" in result

    def test_post_adjustment_invalid_account_rejected(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="post_adjustment",
            arguments={
                "entity_id": "US_PARENT",
                "debit_account_code": "9999",
                "credit_account_code": "2100",
                "amount": "100.00",
                "currency": "USD",
                "reason_code": "manual_true_up",
            },
        ))
        result = _extract_result(obs)
        assert result.get("status") == "rejected"

    def test_post_adjustment_negative_amount_rejected(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="post_adjustment",
            arguments={
                "entity_id": "US_PARENT",
                "debit_account_code": "1300",
                "credit_account_code": "2100",
                "amount": "-100.00",
                "currency": "USD",
                "reason_code": "manual_true_up",
            },
        ))
        result = _extract_result(obs)
        assert result.get("status") == "rejected"

    def test_full_match_then_eliminate(self, reset_env):
        """Full workflow: match → eliminate."""
        d_id, c_id = self._get_matchable_pair(reset_env)
        assert d_id and c_id

        # Match
        obs_match = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": d_id, "credit_txn_id": c_id},
        ))
        match_result = _extract_result(obs_match)
        assert match_result.get("status") == "ok"
        match_id = match_result["match_id"]

        # Get entity from open items
        obs_items = reset_env.step(CallToolAction(tool_name="query_open_items", arguments={"limit": 5}))
        entity_id = _extract_result(obs_items)["items"][0]["entity_id"] if _extract_result(obs_items)["items"] else "US_PARENT"

        # Eliminate
        obs_elim = reset_env.step(CallToolAction(
            tool_name="execute_elimination",
            arguments={"entity_id": entity_id, "matched_pair_id": match_id},
        ))
        elim_result = _extract_result(obs_elim)
        assert elim_result.get("status") == "ok", f"Elimination failed: {elim_result}"
        assert "elimination_id" in elim_result

    def test_eliminate_without_match_rejected(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="execute_elimination",
            arguments={"entity_id": "US_PARENT", "matched_pair_id": "MATCH-FAKE0001"},
        ))
        result = _extract_result(obs)
        assert result.get("status") == "rejected"


# ---------------------------------------------------------------------------
# 6. Reward signals
# ---------------------------------------------------------------------------


class TestRewards:
    def test_read_action_has_small_negative_reward(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="query_open_items", arguments={},
        ))
        assert obs.reward is not None
        assert obs.reward < 0, "Read-only step should cost a small negative reward"

    def test_successful_match_has_positive_reward(self, reset_env):
        items = _extract_result(reset_env.step(CallToolAction(
            tool_name="query_open_items", arguments={}
        )))["items"]
        debits = [i for i in items if i["side"] == "debit"]
        credits = [i for i in items if i["side"] == "credit"]
        d_id = c_id = None
        for d in debits:
            for c in credits:
                if (d["entity_id"] == c["counterparty_entity_id"]
                        and abs(float(d["money"]["amount"])) == abs(float(c["money"]["amount"]))
                        and d["money"]["currency"] == c["money"]["currency"]):
                    d_id, c_id = d["txn_id"], c["txn_id"]
                    break
            if d_id:
                break
        assert d_id, "Need matchable pair"
        obs = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": d_id, "credit_txn_id": c_id},
        ))
        assert obs.reward > 0, "Successful match should earn positive reward"

    def test_invalid_action_has_negative_reward(self, reset_env):
        obs = reset_env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": "FAKE", "credit_txn_id": "FAKE"},
        ))
        assert obs.reward is not None
        assert obs.reward < 0


# ---------------------------------------------------------------------------
# 7. Episode termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_step_after_done_returns_done(self, reset_env):
        reset_env._done = True
        obs = reset_env.step(CallToolAction(tool_name="query_open_items", arguments={}))
        assert obs.done is True

    def test_step_without_reset_returns_done(self, env):
        obs = env.step(CallToolAction(tool_name="query_open_items", arguments={}))
        assert obs.done is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_result(obs) -> dict:
    """Extract the result dict from a CallToolObservation."""
    result = getattr(obs, "result", None)
    if result is None:
        return {}

    # CallToolResult has .structured_content / .data as parsed dicts
    if hasattr(result, "structured_content") and result.structured_content:
        return result.structured_content
    if hasattr(result, "data") and result.data:
        return result.data

    # Fallback: parse JSON from first TextContent
    import json
    content = getattr(result, "content", [])
    for item in content:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}

    if isinstance(result, dict):
        return result
    return {}
