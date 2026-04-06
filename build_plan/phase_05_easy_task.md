# Phase 5: Easy Task — Baseline Batch Matching + Grader

## Goal
Ship the first fully playable, gradeable benchmark task. Clean 1-to-1 intercompany matching with explicit IDs, no ambiguity, no FX, no legal complexity.

## Expected Outcome
- Seed data: 1 benchmark scenario (~40 transactions = 20 pairs) + 1 smoke scenario (~10 transactions = 5 pairs)
- Easy grader returns deterministic score in [0.0, 1.0]
- An agent that correctly matches all pairs and eliminates them scores ~0.95+
- An agent that does nothing scores 0.0

## Task Specification

- **Task ID**: `easy_batch_matching`
- **Difficulty**: `easy`
- **Step limit**: 80 (enough for: 20 queries + 20 matches + 20 eliminations + buffer)
- **Entities**: US_PARENT, UK_SUB
- **Currency**: All USD (no FX needed)
- **Documents**: Simple invoices with explicit references linking debit↔credit
- **Objectives**:
  1. Match all debit-credit transaction pairs
  2. Eliminate all matched pairs

## Seed Data Files

### `seed_data/easy/smoke.json` — 5 pairs for testing

```json
{
  "scenario_id": "easy_smoke",
  "task_id": "easy_batch_matching",
  "difficulty": "easy",
  "description": "Smoke test: 10 clean intercompany transactions (5 pairs) between US_PARENT and UK_SUB. All amounts in USD, explicit 1-to-1 references.",
  "step_limit": 30,
  "objectives": [
    "Match all 5 debit-credit transaction pairs",
    "Eliminate all 5 matched pairs"
  ],
  "ledger_lines": [
    {
      "txn_id": "TXN-E-001-D", "entity_id": "US_PARENT", "counterparty_entity_id": "UK_SUB",
      "account_code": "1300", "side": "debit",
      "money": {"amount": "10000.00", "currency": "USD"},
      "booking_date": "2024-01-15",
      "description": "IC Receivable - Invoice INV-001 consulting services Q1",
      "document_ids": ["DOC-E-001"]
    },
    {
      "txn_id": "TXN-E-001-C", "entity_id": "UK_SUB", "counterparty_entity_id": "US_PARENT",
      "account_code": "2300", "side": "credit",
      "money": {"amount": "10000.00", "currency": "USD"},
      "booking_date": "2024-01-15",
      "description": "IC Payable - Invoice INV-001 consulting services Q1",
      "document_ids": ["DOC-E-001"]
    }
  ],
  "documents": [
    {
      "document_id": "DOC-E-001", "document_type": "invoice",
      "title": "Invoice INV-001: US_PARENT to UK_SUB",
      "body": "Invoice INV-001\nFrom: US_PARENT\nTo: UK_SUB\nAmount: USD 10,000.00\nDate: 2024-01-15\nDescription: Management consulting services - Q1 2024\nDebit Ref: TXN-E-001-D\nCredit Ref: TXN-E-001-C",
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
    "total_expected_matches": 5,
    "total_expected_adjustments": 0,
    "total_expected_eliminations": 5
  }
}
```

**Note**: The above shows the structure with 1 pair. The actual file must contain all 5 pairs (10 ledger lines, 5 documents). Generate them following the same pattern with IDs `TXN-E-001` through `TXN-E-005`, amounts varying between $5,000 and $50,000, descriptions referencing different services (consulting, licensing, IT support, training, marketing).

### `seed_data/easy/benchmark.json` — 20 pairs for benchmark

Same structure but with 40 ledger lines (20 pairs), 20 documents, IDs `TXN-E-001` through `TXN-E-020`. Step limit: 80. Amounts range from $1,000 to $100,000.

**Generation rules for both files:**
- Each pair has one debit line (US_PARENT, account 1300) and one credit line (UK_SUB, account 2300)
- Amounts are identical within each pair
- Transaction IDs follow pattern: `TXN-E-{NNN}-D` (debit) and `TXN-E-{NNN}-C` (credit)
- Document IDs follow pattern: `DOC-E-{NNN}`
- Each document body explicitly contains both transaction IDs
- All dates in January 2024, all amounts in USD

## Grader Implementation

### `graders/base.py`

```python
"""Base grader interface for all task graders."""

from abc import ABC, abstractmethod
from domain.scenario_models import EpisodeContext


class BaseGrader(ABC):
    """Base class for task graders.

    Graders compute the terminal task score by comparing the
    episode's final state against the ground truth checklist.
    """

    @abstractmethod
    def score(self, ctx: EpisodeContext) -> float:
        """Compute terminal score in [0.0, 1.0]."""
        ...

    @abstractmethod
    def detailed_report(self, ctx: EpisodeContext) -> dict:
        """Return a detailed scoring breakdown for debugging."""
        ...
```

### `graders/easy_grader.py`

```python
"""Grader for the easy batch matching task."""

from domain.scenario_models import EpisodeContext
from .base import BaseGrader


class EasyGrader(BaseGrader):
    """Scores the easy task: batch matching + elimination.

    Scoring breakdown:
        - Match coverage:       50% (correct matches / expected matches)
        - Elimination coverage: 40% (eliminations / expected eliminations)
        - Efficiency bonus:     10% (steps used vs step limit)
        - Invalid action penalty: -0.02 per invalid action, max -0.20
    """

    def score(self, ctx: EpisodeContext) -> float:
        report = self.detailed_report(ctx)
        return report["final_score"]

    def detailed_report(self, ctx: EpisodeContext) -> dict:
        gt = ctx.ground_truth

        # --- Match coverage (50%) ---
        correct_matches = self._count_correct_matches(ctx)
        expected = gt.total_expected_matches
        match_ratio = correct_matches / expected if expected > 0 else 1.0
        match_score = 0.50 * match_ratio

        # --- Elimination coverage (40%) ---
        expected_elims = gt.total_expected_eliminations
        actual_elims = len(ctx.eliminations)
        elim_ratio = min(actual_elims, expected_elims) / expected_elims if expected_elims > 0 else 1.0
        elim_score = 0.40 * elim_ratio

        # --- Efficiency (10%) ---
        step_ratio = ctx.step_count / ctx.scenario.step_limit if ctx.scenario.step_limit > 0 else 1.0
        efficiency = max(0.0, 1.0 - step_ratio)
        efficiency_score = 0.10 * efficiency

        # --- Penalty ---
        penalty = min(0.20, ctx.invalid_action_count * 0.02)

        raw = match_score + elim_score + efficiency_score
        final = round(max(0.0, min(1.0, raw - penalty)), 4)

        return {
            "correct_matches": correct_matches,
            "expected_matches": expected,
            "match_ratio": round(match_ratio, 4),
            "match_score": round(match_score, 4),
            "actual_eliminations": actual_elims,
            "expected_eliminations": expected_elims,
            "elim_ratio": round(elim_ratio, 4),
            "elim_score": round(elim_score, 4),
            "steps_used": ctx.step_count,
            "step_limit": ctx.scenario.step_limit,
            "efficiency_score": round(efficiency_score, 4),
            "invalid_actions": ctx.invalid_action_count,
            "penalty": round(penalty, 4),
            "raw_score": round(raw, 4),
            "final_score": final,
        }

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

### `graders/__init__.py`

```python
"""Task graders for the intercompany dispute environment."""

from .base import BaseGrader
from .easy_grader import EasyGrader
```

## Integration: Using Graders in the Environment

Update `server/environment.py`'s `_compute_terminal_score()` to dispatch to the correct grader:

```python
from graders import EasyGrader

def _compute_terminal_score(self) -> float:
    """Dispatch to the correct grader based on task difficulty."""
    difficulty = self._ctx.scenario.difficulty
    if difficulty == "easy":
        grader = EasyGrader()
    # medium and hard graders added in Phases 6/7
    else:
        grader = EasyGrader()  # fallback

    return grader.score(self._ctx)
```

## Expected Agent Workflow (Easy Task)

```
1. reset(task_id="easy_batch_matching")
   → Observation with open items preview and available document IDs

2. list_tools
   → 8 tools available

3. query_open_items(entity_id="US_PARENT")
   → List of debit transactions

4. query_open_items(entity_id="UK_SUB")
   → List of credit transactions

5. (Optional) fetch_document("DOC-E-001")
   → Invoice text with explicit debit/credit refs

6. execute_match(debit_txn_id="TXN-E-001-D", credit_txn_id="TXN-E-001-C")
   → match_id: "MATCH-XXXX"

7. execute_elimination(entity_id="US_PARENT", matched_pair_id="MATCH-XXXX")
   → elimination confirmed

8. Repeat steps 6-7 for all pairs

9. Episode auto-terminates when all matches + eliminations are done
   → terminal_task_score ≈ 0.95
```

## Validation Gate

```bash
# 1. Unit test the grader
uv run python -m pytest tests/test_graders.py::test_easy_grader -v

# 2. Run smoke scenario end-to-end (scripted, no LLM)
uv run python scripts/smoke_eval.py --task easy --scenario smoke

# 3. Verify determinism: run twice, compare scores
uv run python scripts/smoke_eval.py --task easy --scenario smoke  # should produce identical output
```

### Tests to write in `tests/test_graders.py`:
1. Perfect run (all matches + eliminations correct) → score ≈ 0.95+
2. No actions taken → score = 0.0
3. Half matches done → score ≈ 0.45-0.50
4. Correct matches but no eliminations → score ≈ 0.50
5. Invalid actions reduce score
