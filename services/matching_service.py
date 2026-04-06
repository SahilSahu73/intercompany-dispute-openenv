"""Matching service: validate and execute transaction matches, adjustments, eliminations."""

import uuid
from datetime import date as date_type
from datetime import datetime, timezone
from decimal import Decimal

from domain.enums import VALID_ACCOUNT_CODES
from domain.ledger_models import EliminationRecord, JournalEntry, LedgerLine, MatchRecord
from domain.money import Money
from domain.scenario_models import EpisodeContext


def execute_match(ctx: EpisodeContext, debit_txn_id: str, credit_txn_id: str) -> dict:
    """Match a debit transaction with a credit transaction.

    Validation rules:
    1. Both txn_ids must exist in the ledger
    2. The first must be a debit, the second a credit
    3. Both must have status 'open' or 'adjusted'
    4. They must be intercompany counterparties of each other
    5. Amounts must be equal in the same currency

    Returns on success:
        {"match_id": str, "debit_txn_id": str, "credit_txn_id": str, "status": "ok"}
    Returns on failure:
        {"error": str, "status": "rejected"}
    """
    debit_line = ctx.ledger_lines.get(debit_txn_id)
    credit_line = ctx.ledger_lines.get(credit_txn_id)

    if not debit_line:
        return {"error": f"Transaction not found: {debit_txn_id}", "status": "rejected"}
    if not credit_line:
        return {"error": f"Transaction not found: {credit_txn_id}", "status": "rejected"}

    if debit_line.side != "debit":
        return {"error": f"{debit_txn_id} is not a debit transaction", "status": "rejected"}
    if credit_line.side != "credit":
        return {"error": f"{credit_txn_id} is not a credit transaction", "status": "rejected"}

    valid_statuses = {"open", "adjusted"}
    if debit_line.status not in valid_statuses:
        return {
            "error": f"{debit_txn_id} status is '{debit_line.status}', expected open/adjusted",
            "status": "rejected",
        }
    if credit_line.status not in valid_statuses:
        return {
            "error": f"{credit_txn_id} status is '{credit_line.status}', expected open/adjusted",
            "status": "rejected",
        }

    # Must be intercompany counterparties
    if debit_line.entity_id != credit_line.counterparty_entity_id:
        return {
            "error": "Transactions are not intercompany counterparties of each other",
            "status": "rejected",
        }

    # Amounts must match
    if abs(debit_line.money.amount) != abs(credit_line.money.amount):
        return {
            "error": (
                f"Amount mismatch: debit={debit_line.money.amount} {debit_line.money.currency} "
                f"vs credit={credit_line.money.amount} {credit_line.money.currency}. "
                "Post an adjustment first."
            ),
            "status": "rejected",
        }

    if debit_line.money.currency != credit_line.money.currency:
        return {
            "error": (
                f"Currency mismatch: {debit_line.money.currency} vs {credit_line.money.currency}. "
                "Post an FX adjustment first."
            ),
            "status": "rejected",
        }

    # Prevent duplicate matches
    for existing in ctx.matches.values():
        if existing.debit_txn_id == debit_txn_id and existing.credit_txn_id == credit_txn_id:
            return {
                "error": "This pair is already matched",
                "match_id": existing.match_id,
                "status": "rejected",
            }

    # All checks passed — create match record
    match_id = f"MATCH-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)

    match_record = MatchRecord(
        match_id=match_id,
        debit_txn_id=debit_txn_id,
        credit_txn_id=credit_txn_id,
        matched_at=now,
    )
    ctx.matches[match_id] = match_record

    debit_line.status = "matched"
    credit_line.status = "matched"

    return {
        "match_id": match_id,
        "debit_txn_id": debit_txn_id,
        "credit_txn_id": credit_txn_id,
        "status": "ok",
    }


def execute_elimination(ctx: EpisodeContext, entity_id: str, matched_pair_id: str) -> dict:
    """Eliminate a matched pair from the consolidation.

    Validation rules:
    1. matched_pair_id must exist in ctx.matches
    2. The match must involve the given entity_id
    3. Both transactions must have status 'matched'

    Returns on success:
        {"elimination_id": str, "entity_id": str, "matched_pair_id": str, "status": "ok"}
    Returns on failure:
        {"error": str, "status": "rejected"}
    """
    match_record = ctx.matches.get(matched_pair_id)
    if not match_record:
        return {"error": f"Match not found: {matched_pair_id}", "status": "rejected"}

    debit_line = ctx.ledger_lines.get(match_record.debit_txn_id)
    credit_line = ctx.ledger_lines.get(match_record.credit_txn_id)

    if not debit_line or not credit_line:
        return {
            "error": "Internal error: matched transactions not found in ledger",
            "status": "rejected",
        }

    involved_entities = {debit_line.entity_id, credit_line.entity_id}
    if entity_id not in involved_entities:
        return {
            "error": f"Entity {entity_id} is not involved in match {matched_pair_id}",
            "status": "rejected",
        }

    if debit_line.status != "matched" or credit_line.status != "matched":
        return {
            "error": "Transactions are not in 'matched' status — cannot eliminate",
            "status": "rejected",
        }

    # Check for double elimination
    for existing_elim in ctx.eliminations.values():
        if existing_elim.matched_pair_id == matched_pair_id:
            return {
                "error": f"Match {matched_pair_id} is already eliminated",
                "elimination_id": existing_elim.elimination_id,
                "status": "rejected",
            }

    elimination_id = f"ELIM-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)

    elim = EliminationRecord(
        elimination_id=elimination_id,
        entity_id=entity_id,
        matched_pair_id=matched_pair_id,
        eliminated_at=now,
    )
    ctx.eliminations[elimination_id] = elim

    debit_line.status = "eliminated"
    credit_line.status = "eliminated"

    return {
        "elimination_id": elimination_id,
        "entity_id": entity_id,
        "matched_pair_id": matched_pair_id,
        "status": "ok",
    }


def post_adjustment(
    ctx: EpisodeContext,
    entity_id: str,
    debit_account_code: str,
    credit_account_code: str,
    amount: float | str,
    currency: str,
    reason_code: str,
    evidence_refs: list[str] | None = None,
) -> dict:
    """Post an adjustment journal entry to correct a discrepancy.

    Validation rules:
    1. entity_id must be a known entity in the scenario
    2. Account codes must be in VALID_ACCOUNT_CODES
    3. Amount must be positive
    4. Creates balanced debit/credit ledger lines for the adjustment

    Returns on success:
        {"entry_id": str, "entity_id": str, "amount": str, "status": "ok"}
    Returns on failure:
        {"error": str, "status": "rejected"}
    """
    evidence_refs = evidence_refs or []

    try:
        amt = Decimal(str(amount))
    except Exception:
        return {"error": f"Invalid amount format: {amount}", "status": "rejected"}

    if amt <= 0:
        return {"error": "Adjustment amount must be positive", "status": "rejected"}

    if debit_account_code not in VALID_ACCOUNT_CODES:
        return {
            "error": f"Invalid account code: {debit_account_code}. Valid codes: {sorted(VALID_ACCOUNT_CODES)}",
            "status": "rejected",
        }
    if credit_account_code not in VALID_ACCOUNT_CODES:
        return {
            "error": f"Invalid account code: {credit_account_code}. Valid codes: {sorted(VALID_ACCOUNT_CODES)}",
            "status": "rejected",
        }

    # Verify entity exists in scenario
    known_entities = {line.entity_id for line in ctx.ledger_lines.values()}
    # Also check scenario ledger_lines raw data for counterparties
    for raw in ctx.scenario.ledger_lines:
        known_entities.add(raw.get("entity_id", ""))
        known_entities.add(raw.get("counterparty_entity_id", ""))
    known_entities.discard("")

    if entity_id not in known_entities:
        return {"error": f"Unknown entity: {entity_id}", "status": "rejected"}

    money = Money(amount=amt, currency=currency)
    entry_id = f"ADJ-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)

    journal = JournalEntry(
        entry_id=entry_id,
        entity_id=entity_id,
        debit_account_code=debit_account_code,
        credit_account_code=credit_account_code,
        money=money,
        reason_code=reason_code,
        evidence_refs=evidence_refs,
        posted_at=now,
    )
    ctx.adjustments.append(journal)

    # Create adjustment ledger lines so balance queries reflect the adjustment
    today = date_type.today()
    debit_txn_id = f"{entry_id}-D"
    credit_txn_id = f"{entry_id}-C"

    debit_line = LedgerLine(
        txn_id=debit_txn_id,
        entity_id=entity_id,
        counterparty_entity_id=entity_id,
        account_code=debit_account_code,
        side="debit",
        money=money,
        booking_date=today,
        description=f"Adjustment: {reason_code} ({entry_id})",
        status="adjusted",
        document_ids=evidence_refs,
    )
    credit_line = LedgerLine(
        txn_id=credit_txn_id,
        entity_id=entity_id,
        counterparty_entity_id=entity_id,
        account_code=credit_account_code,
        side="credit",
        money=money,
        booking_date=today,
        description=f"Adjustment: {reason_code} ({entry_id})",
        status="adjusted",
        document_ids=evidence_refs,
    )

    ctx.ledger_lines[debit_txn_id] = debit_line
    ctx.ledger_lines[credit_txn_id] = credit_line

    return {
        "entry_id": entry_id,
        "entity_id": entity_id,
        "debit_account_code": debit_account_code,
        "credit_account_code": credit_account_code,
        "amount": str(amt),
        "currency": currency,
        "reason_code": reason_code,
        "status": "ok",
    }
