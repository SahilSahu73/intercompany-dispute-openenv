# Phase 3: Deterministic Service Layer

## Goal
Build the internal services that back each MCP tool. Every service is deterministic, stateless (receives context as arguments), and designed so that later replacement by real MCP servers changes only adapters.

## Expected Outcome
- 6 service modules, each with a narrow interface
- All services are pure functions of their inputs (no randomness, no network calls)
- Same `scenario_id` + `seed` always produces the same outputs
- Service-level unit tests pass

## Architecture Principle

```
Agent → MCPEnvironment.step()
         ├─ CallToolAction("query_open_items") → LedgerService.query_open_items(ctx, ...)
         ├─ CallToolAction("execute_match")    → MatchingService.execute_match(ctx, ...)
         ├─ CallToolAction("fetch_document")   → DocumentService.fetch_document(ctx, ...)
         ├─ CallToolAction("calculate_fx")     → TreasuryService.calculate_fx(ctx, ...)
         ├─ CallToolAction("ask_legal_analyst") → LegalService.ask_legal_analyst(ctx, ...)
         └─ (all mutations via ctx)            → AuditService.record(ctx, ...)
```

Each service method receives the `EpisodeContext` and returns a plain dict (the tool result). Write services mutate `ctx` in place. Read services do not.

## Files to Create

```
services/
├── __init__.py
├── ledger_service.py      # query_open_items, query_ledger_balance
├── document_service.py    # fetch_document
├── treasury_service.py    # calculate_fx
├── legal_service.py       # ask_legal_analyst
├── matching_service.py    # execute_match, execute_elimination
└── audit_service.py       # record_event, detect_loops
```

## File-by-File Implementation

### 1. `services/ledger_service.py`

Two read-only tools: `query_open_items` and `query_ledger_balance`.

```python
"""Ledger service: query open items and account balances."""

from decimal import Decimal
from domain.scenario_models import EpisodeContext
from domain.ledger_models import OpenItemView


def query_open_items(
    ctx: EpisodeContext,
    entity_id: str | None = None,
    counterparty_entity_id: str | None = None,
    status: str = "open",
    limit: int = 100,
) -> dict:
    """Return open items matching the filter criteria.

    Returns:
        {
            "items": [OpenItemView.model_dump(), ...],
            "total_count": int,
            "returned_count": int,
        }
    """
    items = []
    for line in ctx.ledger_lines.values():
        if entity_id and line.entity_id != entity_id:
            continue
        if counterparty_entity_id and line.counterparty_entity_id != counterparty_entity_id:
            continue
        if line.status != status:
            continue
        items.append(OpenItemView.from_ledger_line(line))

    total = len(items)
    items = items[:limit]

    return {
        "items": [item.model_dump(mode="json") for item in items],
        "total_count": total,
        "returned_count": len(items),
    }


def query_ledger_balance(
    ctx: EpisodeContext,
    entity_id: str,
    account_code: str,
) -> dict:
    """Return the net balance for an entity's account.

    Sums debit amounts as positive, credit amounts as negative.

    Returns:
        {
            "entity_id": str,
            "account_code": str,
            "debit_total": str (Decimal),
            "credit_total": str (Decimal),
            "net_balance": str (Decimal),
            "currency": str,
        }
    """
    debit_total = Decimal("0")
    credit_total = Decimal("0")
    currency = None

    for line in ctx.ledger_lines.values():
        if line.entity_id != entity_id or line.account_code != account_code:
            continue
        if line.status == "eliminated":
            continue
        currency = currency or line.money.currency
        if line.side == "debit":
            debit_total += line.money.amount
        else:
            credit_total += line.money.amount

    return {
        "entity_id": entity_id,
        "account_code": account_code,
        "debit_total": str(debit_total),
        "credit_total": str(credit_total),
        "net_balance": str(debit_total - credit_total),
        "currency": currency or "USD",
    }
```

### 2. `services/document_service.py`

```python
"""Document service: retrieve document text and metadata."""

from domain.scenario_models import EpisodeContext


def fetch_document(ctx: EpisodeContext, document_id: str) -> dict:
    """Retrieve full document text by ID.

    Side effect: adds document_id to ctx.evidence_cache so the
    environment knows the agent has gathered this evidence.

    Returns:
        {
            "document_id": str,
            "document_type": str,
            "title": str,
            "body": str,
            "related_entity_ids": list[str],
        }
    OR
        {"error": "Document not found: {document_id}"}
    """
    doc = ctx.documents.get(document_id)
    if not doc:
        return {"error": f"Document not found: {document_id}"}

    # Track that agent has fetched this document (used for evidence checking)
    ctx.evidence_cache.add(document_id)

    return {
        "document_id": doc.document_id,
        "document_type": doc.document_type,
        "title": doc.title,
        "body": doc.body,
        "related_entity_ids": doc.related_entity_ids,
    }
```

### 3. `services/treasury_service.py`

```python
"""Treasury service: deterministic FX rate lookup and conversion."""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from domain.scenario_models import EpisodeContext


def calculate_fx(
    ctx: EpisodeContext,
    source_currency: str,
    target_currency: str,
    amount: float | str,
    conversion_date: str,
) -> dict:
    """Look up the historical FX rate and compute conversion.

    The rate table is seeded per scenario. Only exact date matches
    are returned. If no rate exists for the given date, the nearest
    prior date is used.

    Side effect: sets ctx.fx_queried = True

    Returns:
        {
            "source_currency": str,
            "target_currency": str,
            "conversion_date": str,
            "rate": str (Decimal),
            "source_amount": str (Decimal),
            "converted_amount": str (Decimal),
        }
    OR
        {"error": "No FX rate found for {source_currency}/{target_currency}"}
    """
    if source_currency == target_currency:
        amt = Decimal(str(amount))
        return {
            "source_currency": source_currency,
            "target_currency": target_currency,
            "conversion_date": conversion_date,
            "rate": "1.0",
            "source_amount": str(amt),
            "converted_amount": str(amt),
        }

    conv_date = date.fromisoformat(conversion_date)
    amt = Decimal(str(amount))

    # Find the best matching rate: exact date or nearest prior
    best_rate = None
    best_date = None

    for fx in ctx.fx_rates:
        if fx.source_currency != source_currency or fx.target_currency != target_currency:
            continue
        if fx.rate_date <= conv_date:
            if best_date is None or fx.rate_date > best_date:
                best_rate = fx.rate
                best_date = fx.rate_date

    if best_rate is None:
        # Try the inverse
        for fx in ctx.fx_rates:
            if fx.source_currency != target_currency or fx.target_currency != source_currency:
                continue
            if fx.rate_date <= conv_date:
                if best_date is None or fx.rate_date > best_date:
                    best_rate = Decimal("1") / fx.rate
                    best_date = fx.rate_date

    if best_rate is None:
        return {"error": f"No FX rate found for {source_currency}/{target_currency} on or before {conversion_date}"}

    ctx.fx_queried = True

    converted = (amt * best_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return {
        "source_currency": source_currency,
        "target_currency": target_currency,
        "conversion_date": conversion_date,
        "rate": str(best_rate),
        "source_amount": str(amt),
        "converted_amount": str(converted),
    }
```

### 4. `services/legal_service.py`

```python
"""Legal service: deterministic liability determination from seeded contract truth."""

from domain.scenario_models import EpisodeContext


def ask_legal_analyst(ctx: EpisodeContext, document_id: str, question: str) -> dict:
    """Interpret a contract document and return liability determination.

    The legal truth is pre-seeded per scenario. This service does NOT
    perform LLM inference — it returns the deterministic answer from
    the scenario's ground truth.

    Side effect: sets ctx.legal_consulted = True, adds document_id to evidence_cache

    Returns:
        {
            "document_id": str,
            "question": str,
            "incoterm": str,
            "liable_entity_id": str,
            "liable_event": str,
            "rationale": str,
        }
    OR
        {"error": "..."}
    """
    if not ctx.legal_truth:
        return {"error": "No legal context available for this scenario. Legal analyst not needed for this task difficulty."}

    # The document must exist and be a contract
    doc = ctx.documents.get(document_id)
    if not doc:
        return {"error": f"Document not found: {document_id}"}
    if doc.document_type != "contract":
        return {"error": f"Document {document_id} is a {doc.document_type}, not a contract. Legal analysis requires a contract document."}

    # Check it's the right contract
    if document_id != ctx.legal_truth.contract_document_id:
        return {
            "document_id": document_id,
            "question": question,
            "answer": "This contract does not contain relevant liability terms for the current dispute. Try fetching the transit/shipping contract.",
        }

    ctx.legal_consulted = True
    ctx.evidence_cache.add(document_id)

    return {
        "document_id": document_id,
        "question": question,
        "incoterm": ctx.legal_truth.incoterm,
        "liable_entity_id": ctx.legal_truth.liable_entity_id,
        "liable_event": ctx.legal_truth.liable_event,
        "rationale": ctx.legal_truth.rationale,
    }
```

### 5. `services/matching_service.py`

Two write tools: `execute_match` and `execute_elimination`.

```python
"""Matching service: validate and execute transaction matches and eliminations."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from domain.enums import VALID_ACCOUNT_CODES
from domain.ledger_models import MatchRecord, EliminationRecord
from domain.scenario_models import EpisodeContext


def execute_match(ctx: EpisodeContext, debit_txn_id: str, credit_txn_id: str) -> dict:
    """Match a debit transaction with a credit transaction.

    Validation rules:
    1. Both txn_ids must exist in the ledger
    2. One must be a debit, the other a credit
    3. Both must have status "open" (or "adjusted")
    4. They must be intercompany counterparties of each other
    5. Amounts must be equal (after any adjustments)

    Returns:
        {"match_id": str, "debit_txn_id": str, "credit_txn_id": str, "status": "ok"}
    OR
        {"error": str, "status": "rejected"}
    """
    debit_line = ctx.ledger_lines.get(debit_txn_id)
    credit_line = ctx.ledger_lines.get(credit_txn_id)

    if not debit_line:
        return {"error": f"Transaction not found: {debit_txn_id}", "status": "rejected"}
    if not credit_line:
        return {"error": f"Transaction not found: {credit_txn_id}", "status": "rejected"}

    # Ensure correct sides
    if debit_line.side != "debit":
        return {"error": f"{debit_txn_id} is not a debit transaction", "status": "rejected"}
    if credit_line.side != "credit":
        return {"error": f"{credit_txn_id} is not a credit transaction", "status": "rejected"}

    # Must be open or adjusted
    valid_statuses = {"open", "adjusted"}
    if debit_line.status not in valid_statuses:
        return {"error": f"{debit_txn_id} status is '{debit_line.status}', expected open/adjusted", "status": "rejected"}
    if credit_line.status not in valid_statuses:
        return {"error": f"{credit_txn_id} status is '{credit_line.status}', expected open/adjusted", "status": "rejected"}

    # Must be intercompany counterparties
    if debit_line.entity_id != credit_line.counterparty_entity_id:
        return {"error": "Transactions are not intercompany counterparties", "status": "rejected"}

    # Amounts must match (absolute value, same currency after FX adjustments)
    if abs(debit_line.money.amount) != abs(credit_line.money.amount):
        return {
            "error": f"Amount mismatch: debit={debit_line.money.amount} vs credit={credit_line.money.amount}. Post an adjustment first.",
            "status": "rejected",
        }

    if debit_line.money.currency != credit_line.money.currency:
        return {
            "error": f"Currency mismatch: {debit_line.money.currency} vs {credit_line.money.currency}. Post an FX adjustment first.",
            "status": "rejected",
        }

    # All checks passed — create match
    match_id = f"MATCH-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)

    match_record = MatchRecord(
        match_id=match_id,
        debit_txn_id=debit_txn_id,
        credit_txn_id=credit_txn_id,
        matched_at=now,
    )
    ctx.matches[match_id] = match_record

    # Update ledger line statuses
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
    3. Both transactions must have status "matched"

    Returns:
        {"elimination_id": str, "entity_id": str, "matched_pair_id": str, "status": "ok"}
    OR
        {"error": str, "status": "rejected"}
    """
    match_record = ctx.matches.get(matched_pair_id)
    if not match_record:
        return {"error": f"Match not found: {matched_pair_id}", "status": "rejected"}

    # Verify entity is involved
    debit_line = ctx.ledger_lines.get(match_record.debit_txn_id)
    credit_line = ctx.ledger_lines.get(match_record.credit_txn_id)

    if not debit_line or not credit_line:
        return {"error": "Internal error: matched transactions not found in ledger", "status": "rejected"}

    involved_entities = {debit_line.entity_id, credit_line.entity_id}
    if entity_id not in involved_entities:
        return {"error": f"Entity {entity_id} is not involved in match {matched_pair_id}", "status": "rejected"}

    # Both must still be matched (not already eliminated)
    if debit_line.status != "matched" or credit_line.status != "matched":
        return {"error": f"Transactions not in 'matched' status", "status": "rejected"}

    # Execute elimination
    elimination_id = f"ELIM-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)

    elim = EliminationRecord(
        elimination_id=elimination_id,
        entity_id=entity_id,
        matched_pair_id=matched_pair_id,
        eliminated_at=now,
    )
    ctx.eliminations[elimination_id] = elim

    # Update statuses
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
    """Post an adjustment journal entry.

    Validation rules:
    1. entity_id must be a known entity in the scenario
    2. account codes must be valid
    3. amount must be positive
    4. For medium/hard tasks: evidence_refs should contain document IDs
       the agent has previously fetched (checked by grader, not hard-blocked)

    Returns:
        {"entry_id": str, "entity_id": str, "amount": str, "status": "ok"}
    OR
        {"error": str, "status": "rejected"}
    """
    from domain.money import Money
    from domain.ledger_models import JournalEntry, LedgerLine

    evidence_refs = evidence_refs or []
    amt = Decimal(str(amount))

    if amt <= 0:
        return {"error": "Adjustment amount must be positive", "status": "rejected"}

    if debit_account_code not in VALID_ACCOUNT_CODES:
        return {"error": f"Invalid account code: {debit_account_code}", "status": "rejected"}
    if credit_account_code not in VALID_ACCOUNT_CODES:
        return {"error": f"Invalid account code: {credit_account_code}", "status": "rejected"}

    # Verify entity exists in scenario
    known_entities = {line.entity_id for line in ctx.ledger_lines.values()}
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

    # Create new ledger lines for the adjustment (debit and credit sides)
    debit_txn_id = f"{entry_id}-D"
    credit_txn_id = f"{entry_id}-C"

    from datetime import date as date_type
    today = date_type.today()

    debit_line = LedgerLine(
        txn_id=debit_txn_id,
        entity_id=entity_id,
        counterparty_entity_id=entity_id,  # Self-referencing for adjustments
        account_code=debit_account_code,
        side="debit",
        money=money,
        booking_date=today,
        description=f"Adjustment: {reason_code} - {entry_id}",
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
        description=f"Adjustment: {reason_code} - {entry_id}",
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
```

### 6. `services/audit_service.py`

```python
"""Audit service: record actions and detect exploit patterns."""

from datetime import datetime, timezone
from domain.ledger_models import AuditEvent
from domain.enums import ActionStatus, Actor
from domain.scenario_models import EpisodeContext


def record_event(
    ctx: EpisodeContext,
    actor: str,
    action_type: str,
    status: str,
    detail: str = "",
    reference_id: str | None = None,
) -> None:
    """Append an audit event to the episode log."""
    event = AuditEvent(
        timestamp=datetime.now(timezone.utc),
        actor=actor,
        action_type=action_type,
        status=status,
        detail=detail,
        reference_id=reference_id,
    )
    ctx.audit_log.append(event)


def detect_loops(ctx: EpisodeContext, window: int = 5) -> bool:
    """Check if the last `window` actions are repetitive.

    Returns True if all of the last `window` actions have the same
    action_type AND the same detail (suggesting the agent is stuck in a loop).
    """
    if len(ctx.audit_log) < window:
        return False

    recent = ctx.audit_log[-window:]
    first_type = recent[0].action_type
    first_detail = recent[0].detail

    return all(
        e.action_type == first_type and e.detail == first_detail
        for e in recent
    )


def count_action_type(ctx: EpisodeContext, action_type: str) -> int:
    """Count how many times a specific action type has been called."""
    return sum(1 for e in ctx.audit_log if e.action_type == action_type)
```

### 7. `services/__init__.py`

```python
"""Service layer for the intercompany dispute environment."""

from . import (
    ledger_service,
    document_service,
    treasury_service,
    legal_service,
    matching_service,
    audit_service,
)
```

## Validation Gate

Write tests in `tests/test_services.py`:

```
1. LedgerService:
   - query_open_items returns correct items with entity filter
   - query_open_items respects status filter
   - query_ledger_balance sums debits/credits correctly
   - query_ledger_balance excludes eliminated items

2. DocumentService:
   - fetch_document returns full body for valid doc_id
   - fetch_document returns error for unknown doc_id
   - fetch_document adds doc_id to evidence_cache

3. TreasuryService:
   - calculate_fx returns correct conversion with seeded rate
   - calculate_fx finds nearest prior date when exact date missing
   - calculate_fx returns error when no rate exists
   - calculate_fx handles same-currency (rate = 1.0)

4. LegalService:
   - ask_legal_analyst returns liability determination for correct contract
   - ask_legal_analyst returns error when no legal context exists (easy task)
   - ask_legal_analyst rejects non-contract documents

5. MatchingService:
   - execute_match creates match for valid pair
   - execute_match rejects mismatched amounts
   - execute_match rejects already-matched transactions
   - execute_elimination updates statuses correctly
   - post_adjustment creates balanced journal entry lines
   - post_adjustment rejects invalid account codes

6. AuditService:
   - detect_loops returns True for repetitive actions
   - detect_loops returns False for varied actions

7. Determinism:
   - Running the same scenario twice produces identical results
```

Run: `uv run python -m pytest tests/test_services.py -v`

## Critical Rules

- **No randomness**: UUID generation uses `uuid.uuid4()` which is fine for match/elimination IDs (they're not graded on value). But all financial amounts, rates, and decisions must come from seed data only.
- **Context mutation**: Only write services (`execute_match`, `execute_elimination`, `post_adjustment`) mutate `EpisodeContext`. Read services (`query_*`, `fetch_document`, `calculate_fx`, `ask_legal_analyst`) only read + update tracking flags.
- **Return dicts**: Services return plain dicts, not Pydantic models. This is because MCP tool results are serialized as dicts in `CallToolObservation.result`.
- **Error format**: On failure, return `{"error": "message", "status": "rejected"}`. On success, return the result dict with `"status": "ok"`.
