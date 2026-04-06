# Phase 2: Domain Models + Seed Data

## Goal
Model the accounting world as deterministic typed data. Define all domain entities, seed data format, and the public/private state split — before building any service logic.

## Expected Outcome
- Stable Pydantic domain contracts for money, ledger entries, documents, contracts, FX tables, and ground-truth checklists
- Clear separation: public state (exposed via observations/state) vs hidden grader truth (server-only)
- Seed data JSON schema defined and one smoke fixture per task difficulty
- Unit tests for schema validation and serialization round-trips

## Files to Create

```
domain/
├── __init__.py          # Re-exports key types
├── enums.py             # All string enums and literal types
├── money.py             # Money type with Decimal arithmetic
├── ledger_models.py     # LedgerLine, JournalEntry, MatchRecord, EliminationRecord
├── document_models.py   # Invoice, Email, Contract, ShipmentReport
└── scenario_models.py   # ScenarioBundle, GroundTruthChecklist, EpisodeContext
```

## File-by-File Implementation

### 1. `domain/enums.py`

Centralize all enum/literal constants so every other module imports from here.

```python
"""Enumerations and literal type aliases for the financial domain."""

from typing import Literal

# Entity identifiers used across the simulated multinational
EntityId = Literal["US_PARENT", "UK_SUB", "DE_SUB"]

# ISO currency codes we support
Currency = Literal["USD", "GBP", "EUR"]

# Ledger sides
Side = Literal["debit", "credit"]

# Transaction lifecycle
TxnStatus = Literal["open", "matched", "adjusted", "eliminated"]

# Document types available in the environment
DocumentType = Literal["invoice", "email", "contract", "shipment_report"]

# Reason codes for adjustments
ReasonCode = Literal["fx_variance", "liability_recognition", "inventory_loss", "manual_true_up"]

# Task difficulty levels
Difficulty = Literal["easy", "medium", "hard"]

# Action outcome status
ActionStatus = Literal["ok", "rejected", "invalid", "noop"]

# Incoterms relevant to the hard task
Incoterm = Literal["CIF", "FOB", "EXW", "DDP"]

# Actors in the audit log
Actor = Literal["orchestrator", "ledger_service", "treasury_service", "legal_service", "environment"]

# Account codes used in the simulated chart of accounts
# Format: 4-digit codes grouped by type
# 1xxx = Assets, 2xxx = Liabilities, 3xxx = Equity, 4xxx = Revenue, 5xxx = Expenses
# Intercompany-specific:
#   1300 = Intercompany Receivable
#   2300 = Intercompany Payable
#   1400 = Inventory
#   5100 = Cost of Goods Sold
#   6100 = FX Gain/Loss
#   9100 = Elimination (contra)
VALID_ACCOUNT_CODES = frozenset([
    "1100",  # Cash
    "1200",  # Accounts Receivable
    "1300",  # Intercompany Receivable
    "1400",  # Inventory
    "2100",  # Accounts Payable
    "2300",  # Intercompany Payable
    "3100",  # Retained Earnings
    "4100",  # Revenue
    "5100",  # Cost of Goods Sold
    "5200",  # Operating Expenses
    "6100",  # FX Gain/Loss
    "9100",  # Elimination
])
```

### 2. `domain/money.py`

```python
"""Money type with Decimal precision for accounting arithmetic."""

from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, field_validator
from .enums import Currency


class Money(BaseModel):
    """Monetary amount with currency.

    All financial amounts use Decimal to avoid floating-point errors.
    JSON serialization preserves decimal precision as strings.
    """
    amount: Decimal
    currency: Currency

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v):
        if isinstance(v, float):
            # Convert float to string first to avoid float precision issues
            return Decimal(str(v))
        return Decimal(v)

    def round_to_cents(self) -> "Money":
        """Round to 2 decimal places using banker's rounding."""
        return Money(
            amount=self.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            currency=self.currency,
        )

    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __abs__(self):
        return Money(amount=abs(self.amount), currency=self.currency)
```

### 3. `domain/ledger_models.py`

```python
"""Ledger domain models: transactions, journal entries, matches, eliminations."""

from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, Field
from typing import Optional

from .enums import EntityId, TxnStatus, Side, ReasonCode, ActionStatus, Actor
from .money import Money


class LedgerLine(BaseModel):
    """A single ledger transaction line (the atomic unit of the ERP)."""
    txn_id: str
    entity_id: str           # EntityId but str for flexibility
    counterparty_entity_id: str
    account_code: str
    side: Side
    money: Money
    booking_date: date
    settlement_date: date | None = None
    description: str
    status: TxnStatus = "open"
    document_ids: list[str] = Field(default_factory=list)


class OpenItemView(BaseModel):
    """Public view of a ledger line exposed to the agent.

    This is what the agent sees when querying open items.
    Does NOT include internal fields used for grading.
    """
    txn_id: str
    entity_id: str
    counterparty_entity_id: str
    account_code: str
    side: Side
    money: Money
    booking_date: date
    description: str
    status: TxnStatus
    document_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_ledger_line(cls, line: LedgerLine) -> "OpenItemView":
        return cls(
            txn_id=line.txn_id,
            entity_id=line.entity_id,
            counterparty_entity_id=line.counterparty_entity_id,
            account_code=line.account_code,
            side=line.side,
            money=line.money,
            booking_date=line.booking_date,
            description=line.description,
            status=line.status,
            document_ids=line.document_ids,
        )


class JournalEntry(BaseModel):
    """An adjustment journal entry posted by the agent."""
    entry_id: str
    entity_id: str
    debit_account_code: str
    credit_account_code: str
    money: Money
    reason_code: ReasonCode
    evidence_refs: list[str] = Field(default_factory=list)
    posted_at: datetime
    posted_by: str = "orchestrator"


class MatchRecord(BaseModel):
    """A matched pair of debit/credit transactions."""
    match_id: str
    debit_txn_id: str
    credit_txn_id: str
    matched_at: datetime


class EliminationRecord(BaseModel):
    """Record of a consolidation elimination."""
    elimination_id: str
    entity_id: str
    matched_pair_id: str
    eliminated_at: datetime


class AuditEvent(BaseModel):
    """A single event in the episode audit log."""
    timestamp: datetime
    actor: Actor
    action_type: str
    status: ActionStatus
    detail: str = ""
    reference_id: str | None = None
```

### 4. `domain/document_models.py`

```python
"""Document domain models: invoices, emails, contracts, shipment reports."""

from datetime import date
from pydantic import BaseModel, Field
from typing import Optional

from .enums import DocumentType
from .money import Money


class Document(BaseModel):
    """Base document model stored in the environment."""
    document_id: str
    document_type: DocumentType
    title: str
    body: str                     # Full text content
    related_entity_ids: list[str] = Field(default_factory=list)
    related_txn_ids: list[str] = Field(default_factory=list)
    issue_date: date | None = None


class DocumentSummary(BaseModel):
    """Public summary of a document exposed to the agent.

    The full body is returned by fetch_document tool.
    This summary appears in observation metadata.
    """
    document_id: str
    document_type: DocumentType
    title: str
    snippet: str   # First ~200 chars of body


class Invoice(Document):
    """Invoice document with structured financial data."""
    document_type: DocumentType = "invoice"
    vendor_name: str = ""
    buyer_name: str = ""
    amount: Money | None = None
    due_date: date | None = None


class Contract(Document):
    """Contract document with shipping/legal terms."""
    document_type: DocumentType = "contract"
    incoterm: str | None = None           # e.g. "CIF", "FOB"
    origin_entity_id: str | None = None
    destination_entity_id: str | None = None


class ShipmentReport(Document):
    """Shipment report with damage/loss details."""
    document_type: DocumentType = "shipment_report"
    shipment_id: str = ""
    damage_description: str = ""
    loss_amount: Money | None = None
```

### 5. `domain/scenario_models.py`

This is the most critical file. It defines the seed data format and the hidden grader truth.

```python
"""Scenario models: seed data bundles, ground truth, and episode context."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field

from .enums import Difficulty, EntityId
from .ledger_models import LedgerLine, MatchRecord, JournalEntry, EliminationRecord, AuditEvent
from .document_models import Document
from .money import Money


class FxRate(BaseModel):
    """A single FX rate entry in the treasury table."""
    source_currency: str
    target_currency: str
    rate_date: date
    rate: Decimal       # 1 unit of source = rate units of target


class LegalTruth(BaseModel):
    """Ground truth for legal liability determination (hard task)."""
    contract_document_id: str
    incoterm: str                     # e.g. "CIF", "FOB"
    liable_entity_id: str
    liable_event: str                 # e.g. "goods_damaged_in_transit"
    rationale: str


class GroundTruthChecklist(BaseModel):
    """Hidden grader truth — NEVER exposed to the agent.

    This defines what the correct solution looks like for a scenario.
    The grader compares the agent's actions against this checklist.
    """
    required_matches: list[tuple[str, str]]          # (debit_txn_id, credit_txn_id) pairs
    required_adjustments: list[dict[str, Any]]       # Expected adjustment params
    required_eliminations: list[str]                  # Entity IDs that need elimination
    required_liable_entity_id: str | None = None      # Hard task only
    required_fx_rate: Decimal | None = None            # Medium task only
    required_fx_adjustment_amount: Money | None = None # Medium task only
    total_expected_matches: int = 0
    total_expected_adjustments: int = 0
    total_expected_eliminations: int = 0


class ScenarioBundle(BaseModel):
    """Complete seed data for a single scenario.

    Loaded from JSON files in seed_data/{difficulty}/.
    Contains everything needed to initialize an episode.
    """
    scenario_id: str
    task_id: str                   # e.g. "easy_batch_matching"
    difficulty: Difficulty
    description: str               # Human-readable scenario description
    ledger_lines: list[dict]       # Raw dicts → parsed to LedgerLine at load time
    documents: list[dict]          # Raw dicts → parsed to Document subclasses
    fx_rates: list[dict]           # Raw dicts → parsed to FxRate
    legal_truth: dict | None = None  # Raw dict → parsed to LegalTruth
    objectives: list[str]          # Human-readable objective descriptions
    step_limit: int
    ground_truth: dict             # Raw dict → parsed to GroundTruthChecklist


@dataclass
class EpisodeContext:
    """Mutable internal episode state — NEVER exposed through API.

    This is the server-side working memory for a single episode.
    Created fresh on every reset().
    """
    scenario: ScenarioBundle
    ground_truth: GroundTruthChecklist

    # Mutable ledger state
    ledger_lines: dict[str, LedgerLine] = field(default_factory=dict)      # txn_id -> LedgerLine
    matches: dict[str, MatchRecord] = field(default_factory=dict)          # match_id -> MatchRecord
    adjustments: list[JournalEntry] = field(default_factory=list)
    eliminations: dict[str, EliminationRecord] = field(default_factory=dict)

    # Document store
    documents: dict[str, Document] = field(default_factory=dict)           # doc_id -> Document

    # FX rate table
    fx_rates: list[FxRate] = field(default_factory=list)

    # Legal truth (hard task)
    legal_truth: LegalTruth | None = None

    # Tracking
    evidence_cache: set[str] = field(default_factory=set)     # doc_ids the agent has fetched
    completed_objectives: list[str] = field(default_factory=list)
    audit_log: list[AuditEvent] = field(default_factory=list)

    # Counters
    step_count: int = 0
    invalid_action_count: int = 0
    legal_consulted: bool = False
    fx_queried: bool = False
```

### 6. `domain/__init__.py`

```python
"""Domain models for the intercompany dispute environment."""

from .enums import (
    EntityId, Currency, Side, TxnStatus, DocumentType,
    ReasonCode, Difficulty, ActionStatus, Incoterm,
    VALID_ACCOUNT_CODES,
)
from .money import Money
from .ledger_models import (
    LedgerLine, OpenItemView, JournalEntry,
    MatchRecord, EliminationRecord, AuditEvent,
)
from .document_models import Document, DocumentSummary, Invoice, Contract, ShipmentReport
from .scenario_models import (
    FxRate, LegalTruth, GroundTruthChecklist,
    ScenarioBundle, EpisodeContext,
)
```

## Seed Data JSON Schema

Each scenario is a single JSON file in `seed_data/{difficulty}/`. Example skeleton for easy:

```json
{
  "scenario_id": "easy_smoke",
  "task_id": "easy_batch_matching",
  "difficulty": "easy",
  "description": "Smoke test: 20 clean 1-to-1 intercompany transactions between US_PARENT and UK_SUB",
  "step_limit": 30,
  "objectives": [
    "Match all 10 debit-credit transaction pairs",
    "Eliminate all matched pairs"
  ],
  "ledger_lines": [
    {
      "txn_id": "TXN-E-001-D",
      "entity_id": "US_PARENT",
      "counterparty_entity_id": "UK_SUB",
      "account_code": "1300",
      "side": "debit",
      "money": {"amount": "10000.00", "currency": "USD"},
      "booking_date": "2024-01-15",
      "description": "Intercompany receivable - Invoice INV-001",
      "document_ids": ["DOC-E-001"]
    }
  ],
  "documents": [
    {
      "document_id": "DOC-E-001",
      "document_type": "invoice",
      "title": "Invoice INV-001 from US_PARENT to UK_SUB",
      "body": "Invoice INV-001\nFrom: US_PARENT\nTo: UK_SUB\nAmount: $10,000.00\nDate: 2024-01-15\nDescription: Management consulting services - Q1 2024",
      "related_entity_ids": ["US_PARENT", "UK_SUB"],
      "related_txn_ids": ["TXN-E-001-D", "TXN-E-001-C"]
    }
  ],
  "fx_rates": [],
  "legal_truth": null,
  "ground_truth": {
    "required_matches": [["TXN-E-001-D", "TXN-E-001-C"]],
    "required_adjustments": [],
    "required_eliminations": ["US_PARENT"],
    "total_expected_matches": 10,
    "total_expected_adjustments": 0,
    "total_expected_eliminations": 10
  }
}
```

**Actual seed data files will be created in Phases 5-7.** Phase 2 only defines the schema and creates placeholder files with the above structure (1-2 entries each) to validate parsing.

## Validation Gate

```bash
# Unit tests to write in tests/test_models.py:
# 1. Money: Decimal serialization round-trip (JSON -> Money -> JSON preserves precision)
# 2. Money: float coercion works correctly
# 3. LedgerLine: valid construction with all fields
# 4. OpenItemView.from_ledger_line: correct field mapping
# 5. ScenarioBundle: parse a sample JSON fixture
# 6. GroundTruthChecklist: validates required_matches as list of tuples
# 7. FinanceDisputeState: confirm no GroundTruthChecklist fields leak into State

uv run python -m pytest tests/test_models.py -v
```

## Critical Rules

- **Decimal, not float**: All `Money.amount` values are `Decimal`. The `field_validator` converts float inputs to Decimal via string to avoid precision loss.
- **Public vs private**: `OpenItemView` is the public projection of `LedgerLine`. `GroundTruthChecklist` and `EpisodeContext` are NEVER serialized to the agent.
- **Seed data is file-backed**: Scenarios are loaded from `seed_data/` JSON files, not generated at runtime. This ensures reproducibility.
- **Amount strings in JSON**: Seed data JSON stores amounts as strings (`"10000.00"`) not numbers, to preserve decimal precision.
