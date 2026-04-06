# Phase 4: MCPEnvironment Core — Tools, Wiring, and Reward Engine

## Goal
Wire the domain models and service layer into the actual OpenEnv MCPEnvironment class. Register all 8 tools as `@mcp.tool()` decorators, implement `reset()` / `step()` / `state`, build the reward engine, and make the typed client work end-to-end.

## Expected Outcome
- Full `IntercompanyDisputeEnvironment` that extends `MCPEnvironment`
- 8 MCP tools discoverable via `ListToolsAction`, callable via `CallToolAction`
- `reset(task_id=..., seed=...)` loads a scenario and returns initial observation
- Dense step reward computed after every action
- `state` returns public-only `FinanceDisputeState`
- Typed `IntercompanyDisputeClient` can run a full episode over WebSocket
- One end-to-end smoke test passes locally

## Architecture: How the MCPEnvironment Wiring Works

```
IntercompanyDisputeEnvironment(MCPEnvironment)
    .__init__():
        mcp = FastMCP("intercompany_dispute_env")

        @mcp.tool()   ← 5 read-only tools
        def query_open_items(...): ...
        def query_ledger_balance(...): ...
        def fetch_document(...): ...
        def ask_legal_analyst(...): ...
        def calculate_fx(...): ...

        @mcp.tool()   ← 3 write tools
        def execute_match(...): ...
        def post_adjustment(...): ...
        def execute_elimination(...): ...

        super().__init__(mcp)   ← Registers tools with MCPEnvironment

    .reset():  loads scenario → creates EpisodeContext → returns Observation
    .step():   inherited from MCPEnvironment, auto-routes:
               ListToolsAction  → _handle_list_tools() [built-in]
               CallToolAction   → _handle_call_tool()  [built-in, calls our @mcp.tool functions]
               other            → _step_impl()         [our fallback]
    .state:    returns FinanceDisputeState (public only)
```

**Key insight**: The `@mcp.tool()` closures capture `self`, so they can access `self._ctx` (the episode context). The MCPEnvironment base class handles all the `ListToolsAction`/`CallToolAction` routing automatically — we never write a manual dispatch table.

## File-by-File Implementation

### 1. `server/environment.py` — Full Implementation (replaces Phase 1 stub)

```python
"""
Intercompany Dispute Resolution Environment.

This is the core OpenEnv environment. It extends MCPEnvironment to expose
financial tools via MCP, while maintaining internal episode state for
grading and reward computation.
"""

import json
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from domain import (
    EpisodeContext, ScenarioBundle, GroundTruthChecklist,
    LedgerLine, Document, Invoice, Contract, ShipmentReport,
    FxRate, LegalTruth, Money,
    VALID_ACCOUNT_CODES,
)
from services import ledger_service, document_service, treasury_service, legal_service, matching_service, audit_service
from models import FinanceDisputeState

# Resolve seed_data directory relative to project root
SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"


class IntercompanyDisputeEnvironment(MCPEnvironment):
    """Intercompany financial dispute resolution environment.

    The agent is an Enterprise Consolidation Orchestrator that must
    resolve intercompany accounting disputes using MCP tool calls.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("intercompany_dispute_env")

        # ── Read-only tools ──────────────────────────────────────

        @mcp.tool()
        def query_open_items(
            entity_id: str = "",
            counterparty_entity_id: str = "",
            status: str = "open",
            limit: int = 100,
        ) -> dict:
            """Query open intercompany transaction items.

            Filter by entity, counterparty, and status. Returns a list of
            open ledger items that need to be matched and eliminated.

            Args:
                entity_id: Filter by booking entity (e.g. "US_PARENT"). Empty = all.
                counterparty_entity_id: Filter by counterparty. Empty = all.
                status: Filter by status: "open", "matched", "adjusted", "eliminated".
                limit: Max items to return (1-1000).
            """
            audit_service.record_event(self._ctx, "orchestrator", "query_open_items", "ok",
                                       detail=f"entity={entity_id}")
            return ledger_service.query_open_items(
                self._ctx,
                entity_id=entity_id or None,
                counterparty_entity_id=counterparty_entity_id or None,
                status=status,
                limit=limit,
            )

        @mcp.tool()
        def query_ledger_balance(entity_id: str, account_code: str) -> dict:
            """Query the net balance for an entity's account.

            Returns debit total, credit total, and net balance for a
            specific account code within an entity.

            Args:
                entity_id: The entity to query (e.g. "US_PARENT").
                account_code: The account code (e.g. "1300" for IC Receivable).
            """
            audit_service.record_event(self._ctx, "orchestrator", "query_ledger_balance", "ok",
                                       detail=f"{entity_id}:{account_code}")
            return ledger_service.query_ledger_balance(self._ctx, entity_id, account_code)

        @mcp.tool()
        def fetch_document(document_id: str) -> dict:
            """Retrieve the full text of a document (invoice, contract, email, shipment report).

            Returns the document body, type, and related entities.
            This also registers the document as evidence you have gathered.

            Args:
                document_id: The document ID to fetch (e.g. "DOC-E-001").
            """
            result = document_service.fetch_document(self._ctx, document_id)
            status = "ok" if "error" not in result else "rejected"
            audit_service.record_event(self._ctx, "orchestrator", "fetch_document", status,
                                       detail=document_id)
            return result

        @mcp.tool()
        def ask_legal_analyst(document_id: str, question: str) -> dict:
            """Consult the Legal & Compliance sub-agent about a contract.

            Pass a contract document_id and your question. The legal analyst
            will interpret the contract terms (e.g. Incoterms like CIF/FOB)
            and return a structured liability determination.

            Only works with contract-type documents. Required before posting
            liability adjustments in hard-difficulty tasks.

            Args:
                document_id: A contract document ID.
                question: Your legal question (e.g. "Who is liable for damaged goods?").
            """
            result = legal_service.ask_legal_analyst(self._ctx, document_id, question)
            status = "ok" if "error" not in result else "rejected"
            audit_service.record_event(self._ctx, "orchestrator", "ask_legal_analyst", status,
                                       detail=document_id)
            return result

        @mcp.tool()
        def calculate_fx(
            source_currency: str,
            target_currency: str,
            amount: str,
            conversion_date: str,
        ) -> dict:
            """Convert a monetary amount between currencies using historical FX rates.

            Uses the Treasury service's historical rate table. Rates are
            deterministic and scenario-specific. You must use this tool
            rather than guessing exchange rates.

            Args:
                source_currency: Source currency code (USD, GBP, EUR).
                target_currency: Target currency code (USD, GBP, EUR).
                amount: The amount to convert (as a string, e.g. "10000.00").
                conversion_date: The date for the FX rate (ISO format, e.g. "2024-01-15").
            """
            result = treasury_service.calculate_fx(
                self._ctx, source_currency, target_currency, amount, conversion_date,
            )
            status = "ok" if "error" not in result else "rejected"
            audit_service.record_event(self._ctx, "orchestrator", "calculate_fx", status,
                                       detail=f"{source_currency}->{target_currency}")
            return result

        # ── Write tools ──────────────────────────────────────────

        @mcp.tool()
        def execute_match(debit_txn_id: str, credit_txn_id: str) -> dict:
            """Match a debit transaction with a credit transaction.

            Links an intercompany receivable (debit) with its corresponding
            payable (credit). Both must be open, have matching amounts, and
            be intercompany counterparties.

            Args:
                debit_txn_id: The debit transaction ID.
                credit_txn_id: The credit transaction ID.
            """
            result = matching_service.execute_match(self._ctx, debit_txn_id, credit_txn_id)
            status = result.get("status", "rejected")
            audit_service.record_event(self._ctx, "orchestrator", "execute_match", status,
                                       detail=f"{debit_txn_id}<->{credit_txn_id}",
                                       reference_id=result.get("match_id"))
            if status != "ok":
                self._ctx.invalid_action_count += 1
            return result

        @mcp.tool()
        def post_adjustment(
            entity_id: str,
            debit_account_code: str,
            credit_account_code: str,
            amount: str,
            currency: str,
            reason_code: str,
            evidence_refs: str = "",
        ) -> dict:
            """Post an adjustment journal entry to correct a discrepancy.

            Creates a balanced debit/credit entry. Use this to correct
            FX variances, recognize liabilities, or record inventory losses.

            Args:
                entity_id: The entity to post the adjustment for.
                debit_account_code: Account to debit (e.g. "6100" for FX Gain/Loss).
                credit_account_code: Account to credit (e.g. "1300" for IC Receivable).
                amount: Adjustment amount as string (e.g. "150.00"). Must be positive.
                currency: Currency code (USD, GBP, EUR).
                reason_code: One of: fx_variance, liability_recognition, inventory_loss, manual_true_up.
                evidence_refs: Comma-separated document IDs as evidence (e.g. "DOC-001,DOC-002").
            """
            refs = [r.strip() for r in evidence_refs.split(",") if r.strip()] if evidence_refs else []
            result = matching_service.post_adjustment(
                self._ctx, entity_id, debit_account_code, credit_account_code,
                amount, currency, reason_code, refs,
            )
            status = result.get("status", "rejected")
            audit_service.record_event(self._ctx, "orchestrator", "post_adjustment", status,
                                       detail=f"{entity_id} {reason_code} {amount} {currency}",
                                       reference_id=result.get("entry_id"))
            if status != "ok":
                self._ctx.invalid_action_count += 1
            return result

        @mcp.tool()
        def execute_elimination(entity_id: str, matched_pair_id: str) -> dict:
            """Eliminate a matched intercompany pair from consolidation.

            After matching, you must eliminate the pair to complete the
            reconciliation. The entity must be involved in the match.

            Args:
                entity_id: The entity performing the elimination.
                matched_pair_id: The match ID returned by execute_match.
            """
            result = matching_service.execute_elimination(self._ctx, entity_id, matched_pair_id)
            status = result.get("status", "rejected")
            audit_service.record_event(self._ctx, "orchestrator", "execute_elimination", status,
                                       detail=f"{entity_id}:{matched_pair_id}",
                                       reference_id=result.get("elimination_id"))
            if status != "ok":
                self._ctx.invalid_action_count += 1
            return result

        # ── Init MCPEnvironment ──────────────────────────────────
        super().__init__(mcp)

        # Episode state (set properly on reset)
        self._ctx: EpisodeContext | None = None
        self._episode_id: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0

    # ── Scenario Loading ─────────────────────────────────────────

    def _load_scenario(self, task_id: str, scenario_id: str | None = None) -> ScenarioBundle:
        """Load a scenario bundle from seed_data/{difficulty}/."""
        # Map task_id to difficulty
        difficulty_map = {
            "easy_batch_matching": "easy",
            "medium_fx_variance": "medium",
            "hard_liability_dispute": "hard",
        }
        difficulty = difficulty_map.get(task_id, task_id.split("_")[0])

        scenario_dir = SEED_DATA_DIR / difficulty
        if scenario_id:
            scenario_file = scenario_dir / f"{scenario_id}.json"
        else:
            # Load the first .json file in the directory
            json_files = sorted(scenario_dir.glob("*.json"))
            if not json_files:
                raise ValueError(f"No scenario files in {scenario_dir}")
            scenario_file = json_files[0]

        with open(scenario_file) as f:
            raw = json.load(f)

        return ScenarioBundle.model_validate(raw)

    def _init_episode_context(self, scenario: ScenarioBundle) -> EpisodeContext:
        """Build EpisodeContext from a loaded ScenarioBundle."""
        ground_truth = GroundTruthChecklist.model_validate(scenario.ground_truth)

        # Parse ledger lines
        ledger_lines = {}
        for raw_line in scenario.ledger_lines:
            line = LedgerLine.model_validate(raw_line)
            ledger_lines[line.txn_id] = line

        # Parse documents (dispatch by type)
        documents = {}
        for raw_doc in scenario.documents:
            doc_type = raw_doc.get("document_type", "invoice")
            if doc_type == "contract":
                doc = Contract.model_validate(raw_doc)
            elif doc_type == "shipment_report":
                doc = ShipmentReport.model_validate(raw_doc)
            elif doc_type == "invoice":
                doc = Invoice.model_validate(raw_doc)
            else:
                doc = Document.model_validate(raw_doc)
            documents[doc.document_id] = doc

        # Parse FX rates
        fx_rates = [FxRate.model_validate(r) for r in scenario.fx_rates]

        # Parse legal truth
        legal_truth = None
        if scenario.legal_truth:
            legal_truth = LegalTruth.model_validate(scenario.legal_truth)

        return EpisodeContext(
            scenario=scenario,
            ground_truth=ground_truth,
            ledger_lines=ledger_lines,
            documents=documents,
            fx_rates=fx_rates,
            legal_truth=legal_truth,
        )

    # ── OpenEnv Interface ────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode.

        Kwargs:
            task_id: str — one of "easy_batch_matching", "medium_fx_variance", "hard_liability_dispute"
            scenario_id: str | None — specific scenario file name (without .json)
        """
        task_id = kwargs.get("task_id", "easy_batch_matching")
        scenario_id = kwargs.get("scenario_id", None)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._done = False
        self._cumulative_reward = 0.0

        # Load and initialize
        scenario = self._load_scenario(task_id, scenario_id)
        self._ctx = self._init_episode_context(scenario)

        audit_service.record_event(
            self._ctx, "environment", "reset", "ok",
            detail=f"task={task_id} scenario={scenario.scenario_id}",
        )

        # Build initial observation
        open_items = ledger_service.query_open_items(self._ctx, limit=20)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task_id": task_id,
                "scenario_id": scenario.scenario_id,
                "difficulty": scenario.difficulty,
                "description": scenario.description,
                "objectives": scenario.objectives,
                "step_limit": scenario.step_limit,
                "open_items_preview": open_items,
                "available_document_ids": [doc_id for doc_id in self._ctx.documents.keys()],
                "message": "Episode started. Use list_tools to see available actions, then query_open_items to see transactions.",
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (fallback)."""
        return Observation(
            done=False,
            reward=-0.01,
            metadata={"error": "Unknown action type. Use CallToolAction or ListToolsAction."},
        )

    @property
    def state(self) -> State:
        """Return public episode state. NO grader truth."""
        if not self._ctx:
            return FinanceDisputeState(episode_id=self._episode_id, step_count=0)

        return FinanceDisputeState(
            episode_id=self._episode_id,
            step_count=self._ctx.step_count,
            task_id=self._ctx.scenario.task_id,
            scenario_id=self._ctx.scenario.scenario_id,
            difficulty=self._ctx.scenario.difficulty,
            step_limit=self._ctx.scenario.step_limit,
            completed_objectives=self._ctx.completed_objectives,
            remaining_objectives=[
                obj for obj in self._ctx.scenario.objectives
                if obj not in self._ctx.completed_objectives
            ],
            violations_count=self._ctx.invalid_action_count,
        )

    # ── Reward + Done overrides ──────────────────────────────────
    #
    # MCPEnvironment.step() calls _handle_call_tool() which returns
    # CallToolObservation with done=False, reward=None by default.
    # We override step() to inject our reward and done logic AFTER
    # the tool call completes.

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute action, then inject reward and done flag."""
        if self._done:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode already finished. Call reset()."},
            )

        if not self._ctx:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "No active episode. Call reset() first."},
            )

        self._ctx.step_count += 1

        # Delegate to MCPEnvironment (handles ListTools/CallTool routing)
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Compute reward
        reward = self._compute_step_reward(action, obs)
        self._cumulative_reward += reward

        # Check done conditions
        done = self._check_done()
        self._done = done

        # Inject reward and done into the observation
        obs.reward = reward
        obs.done = done

        # If done, compute terminal score and add to metadata
        if done:
            terminal_score = self._compute_terminal_score()
            obs.metadata = obs.metadata or {}
            obs.metadata["terminal_task_score"] = terminal_score
            obs.metadata["cumulative_reward"] = self._cumulative_reward
            obs.metadata["steps_used"] = self._ctx.step_count
            obs.metadata["step_limit"] = self._ctx.scenario.step_limit

        return obs

    def _compute_step_reward(self, action: Action, obs: Observation) -> float:
        """Dense per-step reward.

        Components:
            +0.10  for a successful match
            +0.15  for a successful elimination
            +0.05  for a successful adjustment
            +0.02  for fetching a document (evidence gathering)
            +0.02  for querying FX rate
            +0.02  for consulting legal analyst
            -0.01  per step (efficiency penalty)
            -0.05  for invalid/rejected actions
            -0.10  for detected loops
        """
        reward = -0.01   # base step cost

        # Determine what happened from the observation
        metadata = obs.metadata or {}
        tool_name = getattr(obs, "tool_name", None)

        # Check if it's a CallToolObservation with a result
        result = getattr(obs, "result", None)
        error = getattr(obs, "error", None)

        if error:
            reward -= 0.05
            return reward

        # Extract result data
        result_data = None
        if result is not None:
            # FastMCP wraps results — try to get the actual data
            if hasattr(result, "data"):
                result_data = result.data
            elif hasattr(result, "structured_content"):
                result_data = getattr(result, "structured_content", {}).get("result")
            elif isinstance(result, dict):
                result_data = result

        if tool_name and result_data and isinstance(result_data, dict):
            status = result_data.get("status", "")

            if status == "rejected" or "error" in result_data:
                reward -= 0.05
            elif tool_name == "execute_match" and status == "ok":
                reward += 0.10
            elif tool_name == "execute_elimination" and status == "ok":
                reward += 0.15
            elif tool_name == "post_adjustment" and status == "ok":
                reward += 0.05
            elif tool_name == "fetch_document" and "error" not in result_data:
                reward += 0.02
            elif tool_name == "calculate_fx" and "error" not in result_data:
                reward += 0.02
            elif tool_name == "ask_legal_analyst" and "error" not in result_data:
                reward += 0.02

        # Loop detection penalty
        if audit_service.detect_loops(self._ctx):
            reward -= 0.10

        return round(reward, 4)

    def _check_done(self) -> bool:
        """Check if the episode should terminate."""
        ctx = self._ctx

        # Step limit exceeded
        if ctx.step_count >= ctx.scenario.step_limit:
            return True

        # All objectives completed (check matches + eliminations vs ground truth)
        gt = ctx.ground_truth
        matches_done = len(ctx.matches) >= gt.total_expected_matches
        elims_done = len(ctx.eliminations) >= gt.total_expected_eliminations
        adjs_done = len(ctx.adjustments) >= gt.total_expected_adjustments

        if matches_done and elims_done and adjs_done:
            return True

        return False

    def _compute_terminal_score(self) -> float:
        """Compute final normalized score in [0.0, 1.0].

        This is the benchmark grade — separate from step reward shaping.
        Computed by comparing episode state against ground truth checklist.
        Actual scoring logic is delegated to graders (Phases 5-7).
        This is a fallback generic scorer.
        """
        ctx = self._ctx
        gt = ctx.ground_truth
        score = 0.0
        total_weight = 0.0

        # Match coverage (40% weight)
        if gt.total_expected_matches > 0:
            correct_matches = self._count_correct_matches()
            match_score = correct_matches / gt.total_expected_matches
            score += 0.40 * match_score
            total_weight += 0.40

        # Elimination coverage (30% weight)
        if gt.total_expected_eliminations > 0:
            elim_score = min(len(ctx.eliminations), gt.total_expected_eliminations) / gt.total_expected_eliminations
            score += 0.30 * elim_score
            total_weight += 0.30

        # Adjustment accuracy (20% weight)
        if gt.total_expected_adjustments > 0:
            adj_score = min(len(ctx.adjustments), gt.total_expected_adjustments) / gt.total_expected_adjustments
            score += 0.20 * adj_score
            total_weight += 0.20

        # Efficiency (10% weight) — fewer steps is better
        step_ratio = ctx.step_count / ctx.scenario.step_limit
        efficiency = max(0.0, 1.0 - step_ratio)
        score += 0.10 * efficiency
        total_weight += 0.10

        # Invalid action penalty
        penalty = min(0.2, ctx.invalid_action_count * 0.02)
        score = max(0.0, score - penalty)

        return round(min(1.0, max(0.0, score)), 4)

    def _count_correct_matches(self) -> int:
        """Count how many matches align with ground truth required_matches."""
        gt_set = set()
        for pair in self._ctx.ground_truth.required_matches:
            gt_set.add((pair[0], pair[1]))
            gt_set.add((pair[1], pair[0]))  # Order shouldn't matter

        correct = 0
        for match in self._ctx.matches.values():
            pair = (match.debit_txn_id, match.credit_txn_id)
            if pair in gt_set:
                correct += 1
        return correct
```

### 2. Updated `models.py`

No changes needed from Phase 1 — `FinanceDisputeState` is already defined. The MCP action/observation types are handled by the framework.

### 3. Updated `client.py`

The client from Phase 1 is sufficient. The MCP tool calls are standard `CallToolAction`/`CallToolObservation` — no custom parsing needed.

### 4. Updated `server/app.py`

```python
"""FastAPI application for the Intercompany Dispute Environment."""

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

from server.environment import IntercompanyDisputeEnvironment

# action_cls=Action base type — MCPEnvironment's deserialize_action()
# auto-routes "call_tool"/"list_tools" type discriminators to MCP classes.
app = create_app(
    env=IntercompanyDisputeEnvironment,
    action_cls=Action,
    observation_cls=Observation,
    env_name="intercompany_dispute_env",
)


def main():
    """CLI entry point: `uv run --project . server`"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
```

**Note on imports**: Since the repo root is the package root, imports use `server.environment`, `domain.*`, `services.*` etc. If import issues arise, verify `[tool.setuptools.packages.find]` in pyproject.toml includes all packages.

## MCP Tool ↔ Service Mapping

| MCP Tool Name          | Service Function                          | Read/Write |
|------------------------|-------------------------------------------|------------|
| `query_open_items`     | `ledger_service.query_open_items()`       | Read       |
| `query_ledger_balance` | `ledger_service.query_ledger_balance()`   | Read       |
| `fetch_document`       | `document_service.fetch_document()`       | Read (+evidence tracking) |
| `ask_legal_analyst`    | `legal_service.ask_legal_analyst()`       | Read (+legal tracking)    |
| `calculate_fx`         | `treasury_service.calculate_fx()`         | Read (+FX tracking)       |
| `execute_match`        | `matching_service.execute_match()`        | Write      |
| `post_adjustment`      | `matching_service.post_adjustment()`      | Write      |
| `execute_elimination`  | `matching_service.execute_elimination()`  | Write      |

## Tool Argument Types — MCP Limitation

**Critical**: MCP tool arguments are passed as strings/primitives via JSON. Complex types like `list[str]` must be serialized as comma-separated strings. This is why `post_adjustment.evidence_refs` accepts `str` (comma-separated) and splits internally, rather than `list[str]`.

## Reward Design Summary

**Step Reward** (dense, per action):
| Event                   | Reward  |
|------------------------|---------|
| Base step cost          | -0.01   |
| Successful match        | +0.10   |
| Successful elimination  | +0.15   |
| Successful adjustment   | +0.05   |
| Evidence gathering      | +0.02   |
| FX query                | +0.02   |
| Legal consultation      | +0.02   |
| Invalid/rejected action | -0.05   |
| Loop detected           | -0.10   |

**Terminal Score** (normalized 0.0–1.0):
| Component              | Weight  |
|-----------------------|---------|
| Match coverage         | 40%     |
| Elimination coverage   | 30%     |
| Adjustment accuracy    | 20%     |
| Step efficiency        | 10%     |
| Invalid action penalty | -0.02/each, max -0.20 |

## Validation Gate

```bash
# 1. Start server
uv run --project . server &
sleep 3

# 2. List tools
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "list_tools"}}' | python -m json.tool

# 3. Reset with easy task
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_batch_matching"}' | python -m json.tool

# 4. Call a tool
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "call_tool", "tool_name": "query_open_items", "arguments": {"entity_id": "US_PARENT"}}}' | python -m json.tool

# 5. Check state
curl -s http://localhost:8000/state | python -m json.tool

# 6. Kill server
kill %1

# 7. Run tests
uv run python -m pytest tests/test_environment.py -v
```

## Common Pitfalls

- **Tool closures capture `self`**: The `@mcp.tool()` functions defined inside `__init__` capture `self` via closure. This is correct — each `IntercompanyDisputeEnvironment` instance gets its own set of tool functions pointing to its own `self._ctx`.
- **`step()` override**: We override `MCPEnvironment.step()` to inject reward/done AFTER the parent routes the action. Call `super().step()` first, then post-process.
- **Observation mutability**: `CallToolObservation.reward` and `.done` are mutable fields. We set them after `super().step()` returns.
- **No `_step_impl` dispatching**: Since ALL our tools are registered via `@mcp.tool()`, the `_step_impl` fallback should rarely fire. It's only for invalid action types.
