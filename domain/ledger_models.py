"""Ledger domain models: transactions, journal entries, matches, eliminations."""

from datetime import date, datetime

from pydantic import BaseModel, Field

from .enums import ActionStatus, Actor, ReasonCode, Side, TxnStatus
from .money import Money


class LedgerLine(BaseModel):
    """A single ledger transaction line (the atomic unit of the ERP)."""

    txn_id: str
    entity_id: str  # EntityId but str for flexibility in seed data
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
