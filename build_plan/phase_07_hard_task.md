# Phase 7: Hard Task — Adversarial Liability Dispute + Grader

## Goal
Add the multi-hop reasoning task where the agent must fetch a shipping contract, consult the legal analyst to determine liability via Incoterms, force the liable entity to recognize the payable/loss, post adjustments, match, and eliminate.

## Expected Outcome
- Seed data: 1 benchmark scenario (3 disputes) + 1 smoke scenario (1 dispute)
- Legal service returns structured liability determination
- Grader checks: correct liable entity, correct accounts, evidence-first ordering, legal consultation
- Cannot be solved by skipping legal retrieval — the grader penalizes blind attempts

## Task Specification

- **Task ID**: `hard_liability_dispute`
- **Difficulty**: `hard`
- **Step limit**: 50 (1 dispute × ~15 steps + buffer for smoke; 3 disputes for benchmark)
- **Entities**: US_PARENT, UK_SUB, DE_SUB (three entities make this harder)
- **Currencies**: USD, GBP, EUR
- **Core mechanic**: Goods damaged in transit between subsidiaries. The receiving entity refuses to recognize the payable. The agent must:
  1. Query open items to identify the imbalance
  2. Fetch the shipment report to understand what happened
  3. Fetch the transit contract to find the Incoterm
  4. Consult the legal analyst with the contract to determine liability
  5. Post adjustments: force the liable entity to recognize the payable + inventory loss
  6. Match the adjusted amounts
  7. Eliminate

## Scenario Design

### How the Liability Dispute Works

**Scenario**: DE_SUB ships €50,000 of inventory to UK_SUB. Goods are damaged in transit.

1. DE_SUB books: Debit 1300 (IC Receivable) €50,000 — they shipped the goods
2. UK_SUB should book: Credit 2300 (IC Payable) — but REFUSES because goods arrived damaged
3. The ledger is imbalanced: DE_SUB has a receivable with no matching payable

**Resolution depends on the Incoterm in the shipping contract:**

- **CIF (Cost, Insurance, Freight)**: The SELLER (DE_SUB) bears risk until goods arrive at destination port. DE_SUB is liable.
  - DE_SUB must: recognize inventory loss (debit 5100 COGS, credit 1400 Inventory) and write off receivable
  - The IC receivable is eliminated as a loss, not matched

- **FOB (Free On Board)**: The BUYER (UK_SUB) bears risk once goods are loaded onto the ship. UK_SUB is liable.
  - UK_SUB must: recognize the payable (credit 2300 IC Payable) and book damaged inventory (debit 1400 Inventory, credit 5100 Inventory Loss)
  - Then the receivable/payable can be matched and eliminated normally

The agent MUST consult the legal analyst to determine which Incoterm applies. Guessing wrong means posting to the wrong entity.

### Prerequisite Chain

```
fetch_document(shipment_report)    → Understand what happened
    ↓
fetch_document(contract)           → Get the shipping terms
    ↓
ask_legal_analyst(contract, ...)   → Determine liability (CIF vs FOB)
    ↓
post_adjustment(liable_entity)     → Recognize payable/loss
    ↓
execute_match(...)                 → Match the pair
    ↓
execute_elimination(...)           → Clean up
```

If the agent skips to `post_adjustment` without consulting legal, the grader penalizes heavily even if they guess correctly.

## Seed Data Files

### `seed_data/hard/smoke.json` — 1 dispute (CIF scenario)

```json
{
  "scenario_id": "hard_smoke",
  "task_id": "hard_liability_dispute",
  "difficulty": "hard",
  "description": "1 adversarial dispute: DE_SUB shipped €50,000 inventory to UK_SUB, goods damaged in transit. Agent must determine liability via shipping contract Incoterms.",
  "step_limit": 30,
  "objectives": [
    "Retrieve the shipment report to understand the damage",
    "Retrieve the transit contract to identify shipping terms",
    "Consult the legal analyst to determine liability",
    "Post adjustments to the liable entity's ledger",
    "Match the intercompany pair",
    "Eliminate the matched pair"
  ],
  "ledger_lines": [
    {
      "txn_id": "TXN-H-001-D",
      "entity_id": "DE_SUB",
      "counterparty_entity_id": "UK_SUB",
      "account_code": "1300",
      "side": "debit",
      "money": {"amount": "50000.00", "currency": "EUR"},
      "booking_date": "2024-03-01",
      "description": "IC Receivable - Inventory shipment to UK_SUB, PO-2024-DE-100",
      "document_ids": ["DOC-H-001", "DOC-H-002"]
    },
    {
      "txn_id": "TXN-H-001-MISSING",
      "entity_id": "UK_SUB",
      "counterparty_entity_id": "DE_SUB",
      "account_code": "2300",
      "side": "credit",
      "money": {"amount": "0.00", "currency": "EUR"},
      "booking_date": "2024-03-01",
      "status": "open",
      "description": "IC Payable - DISPUTED - Goods damaged in transit, payable not recognized",
      "document_ids": ["DOC-H-001", "DOC-H-003"]
    }
  ],
  "documents": [
    {
      "document_id": "DOC-H-001",
      "document_type": "shipment_report",
      "title": "Shipment Report SR-2024-DE-UK-001",
      "body": "SHIPMENT REPORT\nShipment ID: SR-2024-DE-UK-001\nOrigin: DE_SUB (Frankfurt warehouse)\nDestination: UK_SUB (London warehouse)\nDate Shipped: 2024-03-01\nDate Arrived: 2024-03-08\nContents: Industrial components (500 units)\nDeclared Value: EUR 50,000.00\n\nDAMAGE REPORT:\nUpon inspection at London warehouse, 100% of shipment found water-damaged.\nInsurance Claim Filed: Yes (Ref: INS-2024-0301)\nCarrier: TransEuro Logistics GmbH\n\nThe receiving party (UK_SUB) has refused to recognize the payable pending liability determination.",
      "related_entity_ids": ["DE_SUB", "UK_SUB"],
      "related_txn_ids": ["TXN-H-001-D"]
    },
    {
      "document_id": "DOC-H-002",
      "document_type": "contract",
      "title": "Transit Contract TC-2024-DE-UK-001",
      "body": "TRANSIT CONTRACT\nContract ID: TC-2024-DE-UK-001\nSeller: DE_SUB (German Subsidiary)\nBuyer: UK_SUB (UK Subsidiary)\nDate: 2024-02-15\n\nSHIPPING TERMS:\nIncoterm: CIF (Cost, Insurance, and Freight)\nPort of Loading: Hamburg\nPort of Destination: London\n\nLIABILITY CLAUSE:\nPer CIF terms, the seller (DE_SUB) bears all costs, insurance, and freight charges until goods arrive at the destination port. Risk of loss or damage transfers to the buyer only upon delivery at the destination port.\n\nINSURANCE:\nSeller has arranged marine cargo insurance (Policy: MCI-2024-0215) covering transit from Hamburg to London.\n\nSigned by authorized representatives of both parties.",
      "related_entity_ids": ["DE_SUB", "UK_SUB"],
      "related_txn_ids": ["TXN-H-001-D"],
      "incoterm": "CIF",
      "origin_entity_id": "DE_SUB",
      "destination_entity_id": "UK_SUB"
    },
    {
      "document_id": "DOC-H-003",
      "document_type": "email",
      "title": "RE: Damaged Shipment SR-2024-DE-UK-001",
      "body": "From: uk-finance@company.com\nTo: de-finance@company.com\nDate: 2024-03-10\nSubject: RE: Damaged Shipment SR-2024-DE-UK-001\n\nTeam,\n\nWe have received the shipment SR-2024-DE-UK-001 but the entire contents are water-damaged and unusable. We will NOT be recognizing the payable of EUR 50,000 until liability is formally determined.\n\nPlease refer to the transit contract and insurance policy. Under CIF terms, we believe the seller retains liability until destination delivery.\n\nRegards,\nUK Finance Team",
      "related_entity_ids": ["UK_SUB", "DE_SUB"],
      "related_txn_ids": []
    }
  ],
  "fx_rates": [
    {"source_currency": "EUR", "target_currency": "GBP", "rate_date": "2024-03-01", "rate": "0.8560"},
    {"source_currency": "EUR", "target_currency": "USD", "rate_date": "2024-03-01", "rate": "1.0850"}
  ],
  "legal_truth": {
    "contract_document_id": "DOC-H-002",
    "incoterm": "CIF",
    "liable_entity_id": "DE_SUB",
    "liable_event": "goods_damaged_in_transit",
    "rationale": "Under CIF terms, the seller (DE_SUB) bears risk until goods arrive at the destination port (London). Since the goods were damaged in transit before reaching the destination, DE_SUB is liable for the loss."
  },
  "ground_truth": {
    "required_matches": [],
    "required_adjustments": [
      {
        "entity_id": "DE_SUB",
        "debit_account_code": "5100",
        "credit_account_code": "1300",
        "amount": "50000.00",
        "currency": "EUR",
        "reason_code": "inventory_loss"
      }
    ],
    "required_eliminations": [],
    "required_liable_entity_id": "DE_SUB",
    "total_expected_matches": 0,
    "total_expected_adjustments": 1,
    "total_expected_eliminations": 0
  }
}
```

**Note on CIF resolution**: Under CIF, DE_SUB must write off its receivable as an inventory loss (debit 5100 COGS/Inventory Loss, credit 1300 IC Receivable). The UK_SUB disputed payable of €0 stays as-is (they correctly refused). There's no match/elimination because the receivable is written off, not matched.

### `seed_data/hard/benchmark.json` — 3 disputes

Mix of CIF and FOB scenarios involving different entity pairs:
1. DE_SUB → UK_SUB, CIF, DE_SUB liable (same as smoke)
2. US_PARENT → DE_SUB, FOB, DE_SUB liable (buyer)
3. UK_SUB → US_PARENT, CIF, UK_SUB liable (seller)

Step limit: 50.

## Grader Implementation

### `graders/hard_grader.py`

```python
"""Grader for the hard liability dispute task."""

from decimal import Decimal
from domain.scenario_models import EpisodeContext
from .base import BaseGrader


class HardGrader(BaseGrader):
    """Scores the hard task: adversarial liability dispute resolution.

    Scoring breakdown:
        - Evidence gathering:      10% (fetched shipment report + contract)
        - Legal consultation:      20% (consulted legal analyst with correct contract)
        - Liable entity correct:   25% (posted adjustment to correct entity)
        - Adjustment accuracy:     20% (correct accounts, amount, reason)
        - Process ordering:        15% (evidence before action)
        - Efficiency:              10% (steps vs limit)
        - Penalties: -0.05 per invalid, -0.15 if legal not consulted,
                     -0.20 if wrong entity
    """

    def score(self, ctx: EpisodeContext) -> float:
        return self.detailed_report(ctx)["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Evidence gathering (10%) ---
        required_docs = set()
        if ctx.legal_truth:
            required_docs.add(ctx.legal_truth.contract_document_id)
        # Also check for shipment reports
        for doc in ctx.documents.values():
            if doc.document_type == "shipment_report":
                required_docs.add(doc.document_id)
        fetched_required = len(required_docs & ctx.evidence_cache)
        evidence_ratio = fetched_required / len(required_docs) if required_docs else 1.0
        evidence_score = 0.10 * evidence_ratio

        # --- Legal consultation (20%) ---
        legal_score = 0.20 if ctx.legal_consulted else 0.0

        # --- Liable entity correctness (25%) ---
        liable_score = 0.0
        if gt.required_liable_entity_id and ctx.adjustments:
            # Check if any adjustment targets the correct entity
            for adj in ctx.adjustments:
                if adj.entity_id == gt.required_liable_entity_id:
                    liable_score = 0.25
                    break

        # --- Adjustment accuracy (20%) ---
        adj_score = self._score_adjustments(ctx)

        # --- Process ordering (15%) ---
        # Legal must come before adjustments
        order_score = 0.0
        if ctx.legal_consulted:
            # Check if legal was consulted before first adjustment
            legal_events = [e for e in ctx.audit_log if e.action_type == "ask_legal_analyst" and e.status == "ok"]
            adj_events = [e for e in ctx.audit_log if e.action_type == "post_adjustment" and e.status == "ok"]
            if legal_events and adj_events:
                if legal_events[0].timestamp < adj_events[0].timestamp:
                    order_score = 0.15
            elif legal_events and not adj_events:
                order_score = 0.075  # Consulted but didn't act
        # Evidence before legal also checked
        doc_events = [e for e in ctx.audit_log if e.action_type == "fetch_document" and e.status == "ok"]
        if doc_events and legal_events and doc_events[0].timestamp < legal_events[0].timestamp:
            pass  # Already correct, order_score stays
        elif not doc_events:
            order_score *= 0.5  # Didn't fetch documents first

        # --- Efficiency (10%) ---
        step_ratio = ctx.step_count / ctx.scenario.step_limit if ctx.scenario.step_limit > 0 else 1.0
        efficiency_score = 0.10 * max(0.0, 1.0 - step_ratio)

        # --- Penalties ---
        penalty = min(0.20, ctx.invalid_action_count * 0.05)
        if not ctx.legal_consulted:
            penalty += 0.15
        # Wrong entity penalty
        if gt.required_liable_entity_id and ctx.adjustments:
            wrong_entity_adjs = [a for a in ctx.adjustments if a.entity_id != gt.required_liable_entity_id]
            if wrong_entity_adjs and liable_score == 0.0:
                penalty += 0.20

        raw = evidence_score + legal_score + liable_score + adj_score + order_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "evidence_fetched": fetched_required,
            "evidence_required": len(required_docs),
            "evidence_score": round(evidence_score, 4),
            "legal_consulted": ctx.legal_consulted,
            "legal_score": round(legal_score, 4),
            "liable_entity_correct": liable_score > 0,
            "liable_score": round(liable_score, 4),
            "adjustment_score": round(adj_score, 4),
            "process_order_score": round(order_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

    def _score_adjustments(self, ctx: EpisodeContext) -> float:
        """Score adjustments against ground truth (20% total)."""
        gt = ctx.ground_truth
        expected_adjs = gt.required_adjustments
        if not expected_adjs:
            return 0.20

        correct = 0
        for expected in expected_adjs:
            for actual in ctx.adjustments:
                if (
                    actual.entity_id == expected.get("entity_id")
                    and actual.debit_account_code == expected.get("debit_account_code")
                    and actual.credit_account_code == expected.get("credit_account_code")
                    and actual.reason_code == expected.get("reason_code")
                ):
                    expected_amt = Decimal(str(expected.get("amount", "0")))
                    if abs(actual.money.amount - expected_amt) <= Decimal("1.00"):
                        correct += 1
                        break

        return 0.20 * (correct / len(expected_adjs))
```

## Expected Agent Workflow (Hard Task — CIF)

```
1. reset(task_id="hard_liability_dispute")
2. list_tools
3. query_open_items()                   → See DE_SUB receivable + UK_SUB disputed payable
4. fetch_document("DOC-H-001")          → Read shipment report: goods damaged in transit
5. fetch_document("DOC-H-002")          → Read contract: CIF terms, DE_SUB is seller
6. ask_legal_analyst("DOC-H-002", "Who is liable for damaged goods under this contract?")
   → Response: CIF, DE_SUB liable, must recognize inventory loss
7. post_adjustment(entity_id="DE_SUB", debit_account_code="5100",
                   credit_account_code="1300", amount="50000.00",
                   currency="EUR", reason_code="inventory_loss",
                   evidence_refs="DOC-H-001,DOC-H-002")
   → DE_SUB writes off receivable as loss
8. Episode terminates (all required adjustments done)
   → terminal_task_score ≈ 0.85+
```

## Integration: Update Environment Grader Dispatch

In `server/environment.py`, update `_compute_terminal_score()`:

```python
from graders import EasyGrader, MediumGrader, HardGrader

def _compute_terminal_score(self) -> float:
    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }
    grader = graders.get(self._ctx.scenario.difficulty, EasyGrader())
    return grader.score(self._ctx)
```

## Validation Gate

```bash
# Tests:
# 1. CIF scenario: correct entity (seller) identified → high score
# 2. CIF scenario: wrong entity (buyer) posted → low score + penalty
# 3. FOB scenario: correct entity (buyer) identified → high score
# 4. Skipped legal consultation → heavy penalty
# 5. Posted adjustment before evidence → process ordering penalty
# 6. Perfect run → score ≈ 0.85+

uv run python -m pytest tests/test_graders.py::test_hard_grader -v
uv run python scripts/smoke_eval.py --task hard --scenario smoke
```
