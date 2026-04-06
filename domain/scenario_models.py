"""Scenario models: seed data bundles, ground truth, and episode context."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .document_models import Document
from .enums import Difficulty
from .ledger_models import AuditEvent, EliminationRecord, JournalEntry, LedgerLine, MatchRecord
from .money import Money


class FxRate(BaseModel):
    """A single FX rate entry in the treasury table."""

    source_currency: str
    target_currency: str
    rate_date: date
    rate: Decimal  # 1 unit of source = rate units of target

    @classmethod
    def _validate_rate(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        return Decimal(v)


class LegalTruth(BaseModel):
    """Ground truth for legal liability determination (hard task)."""

    contract_document_id: str
    incoterm: str  # e.g. "CIF", "FOB"
    liable_entity_id: str
    liable_event: str  # e.g. "goods_damaged_in_transit"
    rationale: str


class GroundTruthChecklist(BaseModel):
    """Hidden grader truth — NEVER exposed to the agent.

    This defines what the correct solution looks like for a scenario.
    The grader compares the agent's actions against this checklist.
    """

    required_matches: list[list[str]] = Field(default_factory=list)  # [[debit_id, credit_id], ...]
    required_adjustments: list[dict[str, Any]] = Field(default_factory=list)
    required_eliminations: list[str] = Field(default_factory=list)  # Entity IDs
    required_liable_entity_id: str | None = None  # Hard task only
    required_fx_rate: Decimal | None = None  # Medium task only
    required_fx_adjustment_amount: Money | None = None  # Medium task only
    total_expected_matches: int = 0
    total_expected_adjustments: int = 0
    total_expected_eliminations: int = 0


class ScenarioBundle(BaseModel):
    """Complete seed data for a single scenario.

    Loaded from JSON files in seed_data/{difficulty}/.
    Contains everything needed to initialize an episode.
    """

    scenario_id: str
    task_id: str  # e.g. "easy_batch_matching"
    difficulty: Difficulty
    description: str
    ledger_lines: list[dict]  # Raw dicts → parsed to LedgerLine at load time
    documents: list[dict]  # Raw dicts → parsed to Document subclasses
    fx_rates: list[dict]  # Raw dicts → parsed to FxRate
    legal_truth: dict | None = None  # Raw dict → parsed to LegalTruth
    objectives: list[str]
    step_limit: int
    ground_truth: dict  # Raw dict → parsed to GroundTruthChecklist


@dataclass
class EpisodeContext:
    """Mutable internal episode state — NEVER exposed through API.

    This is the server-side working memory for a single episode.
    Created fresh on every reset().
    """

    scenario: ScenarioBundle
    ground_truth: GroundTruthChecklist

    # Mutable ledger state
    ledger_lines: dict[str, LedgerLine] = field(default_factory=dict)  # txn_id -> LedgerLine
    matches: dict[str, MatchRecord] = field(default_factory=dict)  # match_id -> MatchRecord
    adjustments: list[JournalEntry] = field(default_factory=list)
    eliminations: dict[str, EliminationRecord] = field(default_factory=dict)

    # Document store
    documents: dict[str, Document] = field(default_factory=dict)  # doc_id -> Document

    # FX rate table
    fx_rates: list[FxRate] = field(default_factory=list)

    # Legal truth (hard task only)
    legal_truth: LegalTruth | None = None

    # Tracking — what the agent has fetched
    evidence_cache: set[str] = field(default_factory=set)  # doc_ids the agent has fetched
    completed_objectives: list[str] = field(default_factory=list)
    audit_log: list[AuditEvent] = field(default_factory=list)

    # Counters
    step_count: int = 0
    invalid_action_count: int = 0
    legal_consulted: bool = False
    fx_queried: bool = False
