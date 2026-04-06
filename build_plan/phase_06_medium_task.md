# Phase 6: Medium Task — Noisy Text + FX Variance + Grader

## Goal
Add the evidence-heavy task where the agent must read documents, infer linkage from noisy text (no explicit txn IDs), query the Treasury service for exact FX rates, post adjustments, and then match.

## Expected Outcome
- Seed data: 1 benchmark scenario (~10 disputes) + 1 smoke scenario (3 disputes)
- FX rates seeded deterministically
- The grader verifies correct FX rate usage, correct adjustment amounts, and evidence gathering
- An agent that skips evidence or guesses FX rates scores poorly

## Task Specification

- **Task ID**: `medium_fx_variance`
- **Difficulty**: `medium`
- **Step limit**: 60 (3 disputes × ~15 steps each + buffer)
- **Entities**: US_PARENT, UK_SUB
- **Currencies**: USD (US_PARENT books in USD), GBP (UK_SUB books in GBP)
- **Core mechanic**: 30-day payment delay creates FX variance between booking date and settlement date. The agent must:
  1. Fetch the invoice to identify which transactions belong together
  2. Query the FX rate for the settlement date
  3. Post an adjustment for the FX gain/loss
  4. Match the now-equal amounts
  5. Eliminate the matched pair

## Scenario Design

### How the FX Variance Works

1. US_PARENT books an intercompany receivable of $10,000 on Jan 15 at rate 1 USD = 0.80 GBP
2. UK_SUB books the corresponding payable of £8,000 on the same day
3. Payment settles 30 days later (Feb 14) when the rate is 1 USD = 0.82 GBP
4. The £8,000 payable is now worth $9,756.10 (at the new rate), not $10,000
5. There's an FX variance of $243.90 that needs to be posted as an adjustment

The agent must NOT guess the rate — it must call `calculate_fx` with the settlement date to get the exact seeded rate.

### Noisy Text

Invoice documents do NOT contain explicit `TXN-xxx` references. Instead, they contain vendor names, dates, and descriptions that the agent must use to infer linkage. Example:

```
INVOICE #2024-UK-0042
Date: 15 January 2024
From: Acme Holdings Ltd (UK Subsidiary)
To: Consolidated Corp (US Parent)
Re: IT Infrastructure Support Services - Q4 2023
Amount: GBP 8,000.00
Payment Terms: Net 30
Settlement Due: 14 February 2024
Reference: PO-2024-0015
```

The agent must connect this to the matching ledger entries by entity names, dates, and amounts.

## Seed Data Files

### `seed_data/medium/smoke.json` — 3 disputes

```json
{
  "scenario_id": "medium_smoke",
  "task_id": "medium_fx_variance",
  "difficulty": "medium",
  "description": "3 intercompany disputes between US_PARENT (USD) and UK_SUB (GBP) with 30-day FX variance. Invoices have noisy text without explicit transaction IDs.",
  "step_limit": 50,
  "objectives": [
    "Fetch invoice documents to identify transaction linkages",
    "Query FX rates for each settlement date",
    "Post FX variance adjustments for each dispute",
    "Match all adjusted transaction pairs",
    "Eliminate all matched pairs"
  ],
  "ledger_lines": [
    {
      "txn_id": "TXN-M-001-D",
      "entity_id": "US_PARENT",
      "counterparty_entity_id": "UK_SUB",
      "account_code": "1300",
      "side": "debit",
      "money": {"amount": "10000.00", "currency": "USD"},
      "booking_date": "2024-01-15",
      "settlement_date": "2024-02-14",
      "description": "IC Receivable - IT Infrastructure Support Q4 2023",
      "document_ids": ["DOC-M-001"]
    },
    {
      "txn_id": "TXN-M-001-C",
      "entity_id": "UK_SUB",
      "counterparty_entity_id": "US_PARENT",
      "account_code": "2300",
      "side": "credit",
      "money": {"amount": "8000.00", "currency": "GBP"},
      "booking_date": "2024-01-15",
      "settlement_date": "2024-02-14",
      "description": "IC Payable - IT Infrastructure Support Q4 2023",
      "document_ids": ["DOC-M-001"]
    }
  ],
  "documents": [
    {
      "document_id": "DOC-M-001",
      "document_type": "invoice",
      "title": "Invoice #2024-UK-0042",
      "body": "INVOICE #2024-UK-0042\nDate: 15 January 2024\nFrom: Acme Holdings Ltd (UK Subsidiary)\nTo: Consolidated Corp (US Parent)\nRe: IT Infrastructure Support Services - Q4 2023\nOriginal Amount: GBP 8,000.00\nEquivalent at booking: USD 10,000.00 (rate: 1 USD = 0.80 GBP)\nPayment Terms: Net 30\nSettlement Due: 14 February 2024\nReference: PO-2024-0015\n\nNote: Final settlement amount subject to FX rate on settlement date.",
      "related_entity_ids": ["US_PARENT", "UK_SUB"],
      "related_txn_ids": ["TXN-M-001-D", "TXN-M-001-C"]
    }
  ],
  "fx_rates": [
    {"source_currency": "USD", "target_currency": "GBP", "rate_date": "2024-01-15", "rate": "0.8000"},
    {"source_currency": "USD", "target_currency": "GBP", "rate_date": "2024-02-14", "rate": "0.8200"},
    {"source_currency": "USD", "target_currency": "GBP", "rate_date": "2024-02-28", "rate": "0.8150"},
    {"source_currency": "USD", "target_currency": "GBP", "rate_date": "2024-03-15", "rate": "0.8300"}
  ],
  "legal_truth": null,
  "ground_truth": {
    "required_matches": [["TXN-M-001-D", "TXN-M-001-C"]],
    "required_adjustments": [
      {
        "entity_id": "US_PARENT",
        "debit_account_code": "6100",
        "credit_account_code": "1300",
        "amount": "243.90",
        "currency": "USD",
        "reason_code": "fx_variance"
      }
    ],
    "required_eliminations": ["US_PARENT"],
    "required_fx_rate": "0.8200",
    "required_fx_adjustment_amount": {"amount": "243.90", "currency": "USD"},
    "total_expected_matches": 3,
    "total_expected_adjustments": 3,
    "total_expected_eliminations": 3
  }
}
```

**Note**: Above shows 1 dispute. The actual file has 3 disputes with different amounts, dates, and FX rates. Each follows the same pattern but with unique values so the agent can't hardcode answers.

### `seed_data/medium/benchmark.json` — 10 disputes

Same structure with 10 disputes, step_limit: 60. Different amounts, settlement dates, and FX rates.

### How to compute the FX adjustment amount

For the example dispute above:
- US_PARENT booked $10,000 receivable at booking rate 0.80
- At settlement (rate 0.82), GBP 8,000 = USD 8,000 / 0.82 = $9,756.10
- FX loss for US_PARENT = $10,000 - $9,756.10 = $243.90
- Agent must post: debit 6100 (FX Loss), credit 1300 (IC Receivable), amount $243.90

After adjustment, the IC Receivable is $9,756.10, which at rate 0.82 = GBP 8,000, matching the UK_SUB payable.

**Note on matching after adjustment**: The matching service compares amounts in the same currency. After the FX adjustment reduces the USD receivable, the agent needs to convert for comparison. The simplest approach: the agent posts an adjustment on the US_PARENT side to bring the USD amount down, then the match compares the equivalent amounts. The grader checks that the correct FX rate and adjustment amount were used.

**Implementation detail**: For matching to work, both sides need the same currency and amount. After the FX adjustment, the US_PARENT receivable becomes $9,756.10 USD. The UK_SUB payable is £8,000 GBP. These can't be directly matched. Two approaches:

**Approach A (simpler, recommended)**: The medium scenario seed data has BOTH sides in USD. The UK_SUB books the payable in USD too (since it's an IC transaction). The FX variance arises because the UK_SUB's local-currency-equivalent differs. The adjustment is on the US_PARENT side only. After adjustment, both sides are equal in USD and can be matched.

**Use Approach A for the hackathon.** This keeps the matching logic simple while still testing FX knowledge.

## Grader Implementation

### `graders/medium_grader.py`

```python
"""Grader for the medium FX variance task."""

from decimal import Decimal
from domain.scenario_models import EpisodeContext
from .base import BaseGrader


class MediumGrader(BaseGrader):
    """Scores the medium task: FX variance resolution.

    Scoring breakdown:
        - Evidence gathering:    15% (fetched relevant documents)
        - FX rate correctness:   20% (used correct rate from treasury)
        - Adjustment accuracy:   25% (correct amount, account, entity)
        - Match coverage:        20% (correct matches)
        - Elimination coverage:  10% (eliminations completed)
        - Efficiency:            10% (steps used vs limit)
        - Penalties: -0.03 per invalid action, -0.10 if FX not queried
    """

    def score(self, ctx: EpisodeContext) -> float:
        return self.detailed_report(ctx)["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Evidence gathering (15%) ---
        total_docs = len(ctx.scenario.documents)
        fetched_docs = len(ctx.evidence_cache)
        evidence_ratio = min(fetched_docs, total_docs) / total_docs if total_docs > 0 else 1.0
        evidence_score = 0.15 * evidence_ratio

        # --- FX rate correctness (20%) ---
        fx_score = 0.0
        if ctx.fx_queried and gt.required_fx_rate:
            fx_score = 0.20  # Full marks if they queried FX
            # Could further check if they used the correct date, but
            # the treasury service is deterministic so correct query = correct rate
        elif not gt.required_fx_rate:
            fx_score = 0.20  # N/A for this scenario

        # --- Adjustment accuracy (25%) ---
        adj_score = self._score_adjustments(ctx)

        # --- Match coverage (20%) ---
        correct_matches = self._count_correct_matches(ctx)
        expected = gt.total_expected_matches
        match_ratio = correct_matches / expected if expected > 0 else 1.0
        match_score = 0.20 * match_ratio

        # --- Elimination coverage (10%) ---
        expected_elims = gt.total_expected_eliminations
        actual_elims = len(ctx.eliminations)
        elim_ratio = min(actual_elims, expected_elims) / expected_elims if expected_elims > 0 else 1.0
        elim_score = 0.10 * elim_ratio

        # --- Efficiency (10%) ---
        step_ratio = ctx.step_count / ctx.scenario.step_limit if ctx.scenario.step_limit > 0 else 1.0
        efficiency = max(0.0, 1.0 - step_ratio)
        efficiency_score = 0.10 * efficiency

        # --- Penalties ---
        penalty = min(0.20, ctx.invalid_action_count * 0.03)
        if not ctx.fx_queried and gt.required_fx_rate:
            penalty += 0.10  # Didn't use treasury service

        raw = evidence_score + fx_score + adj_score + match_score + elim_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "evidence_fetched": fetched_docs,
            "evidence_total": total_docs,
            "evidence_score": round(evidence_score, 4),
            "fx_queried": ctx.fx_queried,
            "fx_score": round(fx_score, 4),
            "adjustment_score": round(adj_score, 4),
            "correct_matches": correct_matches,
            "expected_matches": expected,
            "match_score": round(match_score, 4),
            "actual_eliminations": actual_elims,
            "expected_eliminations": expected_elims,
            "elim_score": round(elim_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

    def _score_adjustments(self, ctx: EpisodeContext) -> float:
        """Score adjustments against ground truth (25% total)."""
        gt = ctx.ground_truth
        expected_adjs = gt.required_adjustments
        if not expected_adjs:
            return 0.25  # No adjustments needed → full marks

        correct = 0
        for expected in expected_adjs:
            for actual in ctx.adjustments:
                if (
                    actual.entity_id == expected.get("entity_id")
                    and actual.reason_code == expected.get("reason_code")
                    and actual.debit_account_code == expected.get("debit_account_code")
                    and actual.credit_account_code == expected.get("credit_account_code")
                ):
                    # Check amount within tolerance (±$1 for rounding)
                    expected_amt = Decimal(str(expected.get("amount", "0")))
                    if abs(actual.money.amount - expected_amt) <= Decimal("1.00"):
                        correct += 1
                        break

        return 0.25 * (correct / len(expected_adjs))

    def _count_correct_matches(self, ctx: EpisodeContext) -> int:
        gt_pairs = set()
        for pair in ctx.ground_truth.required_matches:
            gt_pairs.add((pair[0], pair[1]))
            gt_pairs.add((pair[1], pair[0]))

        correct = 0
        for match in ctx.matches.values():
            if (match.debit_txn_id, match.credit_txn_id) in gt_pairs:
                correct += 1
        return correct
```

## Expected Agent Workflow (Medium Task)

```
1. reset(task_id="medium_fx_variance")
2. list_tools
3. query_open_items()                        → See all open items, note currency differences
4. fetch_document("DOC-M-001")               → Read invoice, find settlement date
5. calculate_fx("USD", "GBP", "10000", "2024-02-14")  → Get settlement FX rate
6. post_adjustment(entity_id="US_PARENT", debit_account_code="6100",
                   credit_account_code="1300", amount="243.90",
                   currency="USD", reason_code="fx_variance",
                   evidence_refs="DOC-M-001")
7. execute_match("TXN-M-001-D", "TXN-M-001-C")      → After amounts align
8. execute_elimination("US_PARENT", matched_pair_id)
9. Repeat 4-8 for remaining disputes
```

## Validation Gate

```bash
# Grader tests
uv run python -m pytest tests/test_graders.py::test_medium_grader -v

# Tests to write:
# 1. Perfect run with correct FX → score ≈ 0.90+
# 2. Correct FX but no evidence fetching → score ≈ 0.75 (evidence penalty)
# 3. Wrong FX rate (guessed) → score < 0.50
# 4. Matches without adjustment → rejected, low score
# 5. All evidence, FX, adjustments, matches, but no eliminations → score ≈ 0.80
```
