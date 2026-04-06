"""
Intercompany Dispute Resolution Environment.

Full MCPEnvironment implementation with 8 MCP tools, deterministic
service routing, dense step rewards, and grader-ready terminal scoring.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.env_server.types import Action, Observation, State

from domain import (
    VALID_ACCOUNT_CODES,
    Contract,
    Document,
    EpisodeContext,
    FxRate,
    GroundTruthChecklist,
    Invoice,
    LedgerLine,
    LegalTruth,
    Money,
    ScenarioBundle,
    ShipmentReport,
)
from models import FinanceDisputeState
from services import (
    audit_service,
    document_service,
    ledger_service,
    legal_service,
    matching_service,
    treasury_service,
)

# Resolve seed_data directory relative to project root
SEED_DATA_DIR = Path(__file__).resolve().parent.parent / "seed_data"

TASK_TO_DIFFICULTY = {
    "easy_batch_matching": "easy",
    "medium_fx_variance": "medium",
    "hard_liability_dispute": "hard",
}


class IntercompanyDisputeEnvironment(MCPEnvironment):
    """Intercompany financial dispute resolution environment.

    The agent is an Enterprise Consolidation Orchestrator that must
    resolve intercompany accounting disputes across a simulated
    multinational enterprise using MCP tool calls.

    Tools available:
        Read:  query_open_items, query_ledger_balance, fetch_document,
               ask_legal_analyst, calculate_fx
        Write: execute_match, post_adjustment, execute_elimination
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("intercompany_dispute_env")

        # ── Read-only tools ──────────────────────────────────────────────────

        @mcp.tool()
        def query_open_items(
            entity_id: str = "",
            counterparty_entity_id: str = "",
            status: str = "open",
            limit: int = 100,
        ) -> dict:
            """Query open intercompany transaction items.

            Filter by entity, counterparty, and lifecycle status. Returns a
            list of open ledger items that need to be matched and eliminated.

            Args:
                entity_id: Filter by booking entity (e.g. "US_PARENT"). Empty = all entities.
                counterparty_entity_id: Filter by counterparty. Empty = all.
                status: One of "open", "matched", "adjusted", "eliminated". Default "open".
                limit: Max items to return (1-500). Default 100.

            Returns:
                {"items": [...], "total_count": int, "returned_count": int}
            """
            limit = max(1, min(500, limit))
            audit_service.record_event(
                self._ctx, "orchestrator", "query_open_items", "ok",
                detail=f"entity={entity_id or 'all'} status={status}",
            )
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
                entity_id: The entity to query (e.g. "US_PARENT", "UK_SUB", "DE_SUB").
                account_code: Account code (e.g. "1300" for IC Receivable, "2300" for IC Payable).

            Returns:
                {"entity_id", "account_code", "debit_total", "credit_total", "net_balance", "currency"}
            """
            audit_service.record_event(
                self._ctx, "orchestrator", "query_ledger_balance", "ok",
                detail=f"{entity_id}:{account_code}",
            )
            return ledger_service.query_ledger_balance(self._ctx, entity_id, account_code)

        @mcp.tool()
        def fetch_document(document_id: str) -> dict:
            """Retrieve the full text of a document (invoice, contract, email, shipment report).

            Returns the document body, type, title, and related entities.
            Fetching a document also registers it as gathered evidence,
            which is required before posting write actions in medium/hard tasks.

            Args:
                document_id: The document ID to fetch (e.g. "DOC-E-001").

            Returns:
                {"document_id", "document_type", "title", "body", "related_entity_ids"}
            """
            result = document_service.fetch_document(self._ctx, document_id)
            status = "ok" if "error" not in result else "rejected"
            audit_service.record_event(
                self._ctx, "orchestrator", "fetch_document", status,
                detail=document_id,
            )
            return result

        @mcp.tool()
        def ask_legal_analyst(document_id: str, question: str) -> dict:
            """Consult the Legal & Compliance sub-agent about a contract document.

            Interprets contract terms (e.g. Incoterms like CIF/FOB) and returns
            a structured liability determination. Required before posting liability
            adjustments in hard-difficulty tasks.

            Args:
                document_id: A contract document ID (document_type must be "contract").
                question: Your legal question (e.g. "Who is liable for the damaged goods?").

            Returns:
                {"document_id", "question", "incoterm", "liable_entity_id", "liable_event", "rationale"}
            """
            result = legal_service.ask_legal_analyst(self._ctx, document_id, question)
            status = "ok" if "error" not in result and "answer" not in result else "rejected"
            audit_service.record_event(
                self._ctx, "orchestrator", "ask_legal_analyst", status,
                detail=document_id,
            )
            return result

        @mcp.tool()
        def calculate_fx(
            source_currency: str,
            target_currency: str,
            amount: str,
            conversion_date: str,
        ) -> dict:
            """Convert a monetary amount between currencies using historical FX rates.

            Uses the Treasury service's scenario-seeded historical rate table.
            You MUST use this tool to get the correct rate — do not guess rates.

            Args:
                source_currency: Source currency code ("USD", "GBP", or "EUR").
                target_currency: Target currency code ("USD", "GBP", or "EUR").
                amount: Amount to convert as a string (e.g. "10000.00").
                conversion_date: Date for the FX rate in ISO format (e.g. "2024-02-14").

            Returns:
                {"source_currency", "target_currency", "conversion_date", "rate",
                 "source_amount", "converted_amount"}
            """
            valid_currencies = {"USD", "GBP", "EUR"}
            if source_currency not in valid_currencies or target_currency not in valid_currencies:
                return {
                    "error": f"Invalid currency. Must be one of: {valid_currencies}",
                    "status": "rejected",
                }

            result = treasury_service.calculate_fx(
                self._ctx, source_currency, target_currency, amount, conversion_date,
            )
            status = "ok" if "error" not in result else "rejected"
            audit_service.record_event(
                self._ctx, "orchestrator", "calculate_fx", status,
                detail=f"{source_currency}->{target_currency} on {conversion_date}",
            )
            return result

        # ── Write tools ──────────────────────────────────────────────────────

        @mcp.tool()
        def execute_match(debit_txn_id: str, credit_txn_id: str) -> dict:
            """Match a debit transaction with a credit transaction.

            Links an intercompany receivable (debit) with its corresponding
            payable (credit). Both must be open or adjusted, have equal amounts
            in the same currency, and be counterparties of each other.

            Args:
                debit_txn_id: The debit transaction ID (e.g. "TXN-E-001-D").
                credit_txn_id: The credit transaction ID (e.g. "TXN-E-001-C").

            Returns:
                {"match_id", "debit_txn_id", "credit_txn_id", "status"}
            """
            if not debit_txn_id.strip() or not credit_txn_id.strip():
                self._ctx.invalid_action_count += 1
                return {"error": "Transaction IDs cannot be empty", "status": "invalid"}

            result = matching_service.execute_match(self._ctx, debit_txn_id, credit_txn_id)
            status = result.get("status", "rejected")
            audit_service.record_event(
                self._ctx, "orchestrator", "execute_match", status,
                detail=f"{debit_txn_id}<->{credit_txn_id}",
                reference_id=result.get("match_id"),
            )
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

            Creates a balanced debit/credit journal entry. Use to correct FX
            variances, recognize liabilities, or record inventory losses.

            Args:
                entity_id: Entity to post the adjustment for ("US_PARENT", "UK_SUB", "DE_SUB").
                debit_account_code: Account to debit (e.g. "6100" for FX Gain/Loss).
                credit_account_code: Account to credit (e.g. "1300" for IC Receivable).
                amount: Adjustment amount as string (e.g. "243.90"). Must be positive.
                currency: Currency code ("USD", "GBP", or "EUR").
                reason_code: One of: fx_variance, liability_recognition, inventory_loss, manual_true_up.
                evidence_refs: Comma-separated document IDs as evidence (e.g. "DOC-001,DOC-002").

            Returns:
                {"entry_id", "entity_id", "debit_account_code", "credit_account_code",
                 "amount", "currency", "reason_code", "status"}
            """
            valid_reasons = {"fx_variance", "liability_recognition", "inventory_loss", "manual_true_up"}
            if reason_code not in valid_reasons:
                self._ctx.invalid_action_count += 1
                return {
                    "error": f"Invalid reason_code: '{reason_code}'. Must be one of: {valid_reasons}",
                    "status": "invalid",
                }

            refs = [r.strip() for r in evidence_refs.split(",") if r.strip()] if evidence_refs else []
            result = matching_service.post_adjustment(
                self._ctx, entity_id, debit_account_code, credit_account_code,
                amount, currency, reason_code, refs,
            )
            status = result.get("status", "rejected")
            audit_service.record_event(
                self._ctx, "orchestrator", "post_adjustment", status,
                detail=f"{entity_id} {reason_code} {amount} {currency}",
                reference_id=result.get("entry_id"),
            )
            if status != "ok":
                self._ctx.invalid_action_count += 1
            return result

        @mcp.tool()
        def execute_elimination(entity_id: str, matched_pair_id: str) -> dict:
            """Eliminate a matched intercompany pair from the consolidation.

            After matching, you must eliminate the pair to complete the
            reconciliation cycle. The entity must be involved in the match.

            Args:
                entity_id: The entity performing the elimination (must be in the match).
                matched_pair_id: The match ID returned by execute_match (e.g. "MATCH-XXXXXXXX").

            Returns:
                {"elimination_id", "entity_id", "matched_pair_id", "status"}
            """
            if not matched_pair_id.strip():
                self._ctx.invalid_action_count += 1
                return {"error": "matched_pair_id cannot be empty", "status": "invalid"}

            result = matching_service.execute_elimination(self._ctx, entity_id, matched_pair_id)
            status = result.get("status", "rejected")
            audit_service.record_event(
                self._ctx, "orchestrator", "execute_elimination", status,
                detail=f"{entity_id}:{matched_pair_id}",
                reference_id=result.get("elimination_id"),
            )
            if status != "ok":
                self._ctx.invalid_action_count += 1
            return result

        # ── Register with MCPEnvironment ─────────────────────────────────────
        super().__init__(mcp)

        # Episode state — reset on every reset()
        self._ctx: EpisodeContext | None = None
        self._episode_id: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0

    # ── Scenario loading ─────────────────────────────────────────────────────

    def _load_scenario(self, task_id: str, scenario_id: str | None) -> ScenarioBundle:
        difficulty = TASK_TO_DIFFICULTY.get(task_id, task_id.split("_")[0])
        scenario_dir = SEED_DATA_DIR / difficulty

        if scenario_id:
            scenario_file = scenario_dir / f"{scenario_id}.json"
        else:
            json_files = sorted(scenario_dir.glob("*.json"))
            if not json_files:
                raise ValueError(f"No scenario files found in {scenario_dir}")
            scenario_file = json_files[0]

        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

        with open(scenario_file) as f:
            raw = json.load(f)
        return ScenarioBundle.model_validate(raw)

    def _init_episode_context(self, scenario: ScenarioBundle) -> EpisodeContext:
        ground_truth = GroundTruthChecklist.model_validate(scenario.ground_truth)

        # Parse ledger lines
        ledger_lines: dict = {}
        for raw_line in scenario.ledger_lines:
            line = LedgerLine.model_validate(raw_line)
            ledger_lines[line.txn_id] = line

        # Parse documents — dispatch by type
        documents: dict = {}
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

        # Parse legal truth (hard task only)
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

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode.

        Kwargs accepted (pass via reset kwargs):
            task_id (str): One of "easy_batch_matching", "medium_fx_variance",
                           "hard_liability_dispute". Default: "easy_batch_matching".
            scenario_id (str | None): Specific scenario filename (without .json).
                                      If None, loads the first file alphabetically.
        """
        task_id = kwargs.get("task_id", "easy_batch_matching")
        scenario_id = kwargs.get("scenario_id", None)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._done = False
        self._cumulative_reward = 0.0

        scenario = self._load_scenario(task_id, scenario_id)
        self._ctx = self._init_episode_context(scenario)

        audit_service.record_event(
            self._ctx, "environment", "reset", "ok",
            detail=f"task={task_id} scenario={scenario.scenario_id}",
        )

        # Build initial observation
        open_items_preview = ledger_service.query_open_items(self._ctx, limit=10)
        doc_ids = list(self._ctx.documents.keys())

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
                "available_document_ids": doc_ids,
                "open_items_preview": open_items_preview,
                "message": (
                    "Episode started. "
                    "Use list_tools to discover actions, "
                    "then query_open_items to see transactions."
                ),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute action, compute reward, check done condition."""
        # list_tools is stateless — allow it even without an active episode
        if isinstance(action, ListToolsAction):
            return super().step(action, timeout_s=timeout_s, **kwargs)

        if self._done:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode already finished. Call reset() to start a new one."},
            )

        if self._ctx is None:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "No active episode. Call reset() first."},
            )

        self._ctx.step_count += 1

        # Delegate to MCPEnvironment (handles ListTools/CallTool routing)
        try:
            obs = super().step(action, timeout_s=timeout_s, **kwargs)
        except Exception as e:
            self._ctx.invalid_action_count += 1
            audit_service.record_event(
                self._ctx, "environment", "step_error", "invalid", detail=str(e)[:200],
            )
            obs = Observation(
                done=False,
                reward=-0.05,
                metadata={"error": f"Action failed: {str(e)[:200]}"},
            )

        # Compute and inject step reward
        reward = self._compute_step_reward(obs)
        self._cumulative_reward += reward
        obs.reward = reward

        # Check termination
        done = self._check_done()
        self._done = done
        obs.done = done

        # Add terminal score on episode end
        if done:
            terminal_score = self._compute_terminal_score()
            obs.metadata = obs.metadata or {}
            obs.metadata["terminal_task_score"] = terminal_score
            obs.metadata["cumulative_reward"] = round(self._cumulative_reward, 4)
            obs.metadata["steps_used"] = self._ctx.step_count
            obs.metadata["step_limit"] = self._ctx.scenario.step_limit

        return obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Fallback for non-MCP actions."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": "Unknown action type. Use CallToolAction or ListToolsAction."},
        )

    @property
    def state(self) -> State:
        """Return public episode state. Never exposes grader truth."""
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

    # ── Reward engine ────────────────────────────────────────────────────────

    def _compute_step_reward(self, obs: Observation) -> float:
        """Dense per-step reward signal.

        Base costs and bonuses:
            -0.01  per step (efficiency pressure)
            +0.10  successful match
            +0.15  successful elimination
            +0.05  successful adjustment
            +0.02  evidence fetched (fetch_document, calculate_fx, ask_legal_analyst)
            -0.05  invalid/rejected action
            -0.10  loop detected
        """
        reward = -0.01  # base step cost

        # Try to extract result data from CallToolObservation
        tool_name = getattr(obs, "tool_name", None)
        result_obj = getattr(obs, "result", None)
        obs_error = getattr(obs, "error", None)

        if obs_error:
            reward -= 0.05
            return round(reward, 4)

        # Extract result dict from FastMCP result wrapper
        result_data = None
        if result_obj is not None:
            if hasattr(result_obj, "data"):
                result_data = result_obj.data
            elif hasattr(result_obj, "structured_content"):
                sc = getattr(result_obj, "structured_content", {}) or {}
                result_data = sc.get("result")
            elif isinstance(result_obj, dict):
                result_data = result_obj

        if tool_name and result_data and isinstance(result_data, dict):
            status = result_data.get("status", "")
            has_error = "error" in result_data

            if status in ("rejected", "invalid") or has_error:
                reward -= 0.05
            elif tool_name == "execute_match" and status == "ok":
                reward += 0.10
            elif tool_name == "execute_elimination" and status == "ok":
                reward += 0.15
            elif tool_name == "post_adjustment" and status == "ok":
                reward += 0.05
            elif tool_name in ("fetch_document", "calculate_fx", "ask_legal_analyst") and not has_error:
                reward += 0.02

        # Loop detection penalty
        if self._ctx and audit_service.detect_loops(self._ctx):
            reward -= 0.10

        return round(reward, 4)

    def _check_done(self) -> bool:
        """Check episode termination conditions."""
        ctx = self._ctx

        # Step limit exceeded
        if ctx.step_count >= ctx.scenario.step_limit:
            return True

        # Catastrophic failure: too many invalid actions
        if ctx.invalid_action_count >= 20:
            return True

        # All objectives completed: matches + eliminations + adjustments
        gt = ctx.ground_truth
        matches_done = len(ctx.matches) >= gt.total_expected_matches
        elims_done = len(ctx.eliminations) >= gt.total_expected_eliminations
        adjs_done = len(ctx.adjustments) >= gt.total_expected_adjustments

        if matches_done and elims_done and adjs_done:
            return True

        return False

    def _compute_terminal_score(self) -> float:
        """Compute final normalized score [0.0, 1.0].

        Dispatches to the appropriate grader based on task difficulty.
        Falls back to the generic scorer if grader not yet implemented.
        """
        try:
            from graders import get_grader
            grader = get_grader(self._ctx.scenario.difficulty)
            return grader.score(self._ctx)
        except (ImportError, AttributeError, ValueError):
            return self._generic_score()

    def _generic_score(self) -> float:
        """Generic fallback terminal scorer used before graders are wired in."""
        ctx = self._ctx
        gt = ctx.ground_truth
        score = 0.0

        if gt.total_expected_matches > 0:
            correct = self._count_correct_matches()
            score += 0.40 * (correct / gt.total_expected_matches)

        if gt.total_expected_eliminations > 0:
            elim_ratio = min(len(ctx.eliminations), gt.total_expected_eliminations)
            score += 0.30 * (elim_ratio / gt.total_expected_eliminations)

        if gt.total_expected_adjustments > 0:
            adj_ratio = min(len(ctx.adjustments), gt.total_expected_adjustments)
            score += 0.20 * (adj_ratio / gt.total_expected_adjustments)

        step_ratio = ctx.step_count / ctx.scenario.step_limit if ctx.scenario.step_limit > 0 else 1.0
        score += 0.10 * max(0.0, 1.0 - step_ratio)

        penalty = min(0.20, ctx.invalid_action_count * 0.02)
        score = max(0.0, score - penalty)

        return round(min(1.0, score), 4)

    def _count_correct_matches(self) -> int:
        gt_pairs = set()
        for pair in self._ctx.ground_truth.required_matches:
            gt_pairs.add((pair[0], pair[1]))
            gt_pairs.add((pair[1], pair[0]))

        correct = 0
        for match in self._ctx.matches.values():
            if (match.debit_txn_id, match.credit_txn_id) in gt_pairs:
                correct += 1
        return correct
