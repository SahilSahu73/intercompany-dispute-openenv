# Phase 8: Hardening — Anti-Exploit, Edge Cases, and Robustness

## Goal
Make the benchmark trustworthy for hackathon judging and future RL use. Harden the environment against exploit patterns, ensure graders are deterministic, and add comprehensive edge-case handling.

## Expected Outcome
- All three graders are deterministic (same inputs → same score, always)
- Exploit patterns (loops, spam, hallucinated IDs, random guessing) are penalized
- Episode termination is robust (success, failure, step-limit, and error paths)
- The environment never crashes on malformed input — it returns error observations

## Hardening Checklist

### 1. Input Validation (in tool closures, `server/environment.py`)

Add defensive validation at the tool closure level (before calling services):

```python
# In each @mcp.tool() closure, validate inputs BEFORE calling service:

# query_open_items: clamp limit to [1, 500]
limit = max(1, min(500, limit))

# execute_match: reject empty strings
if not debit_txn_id.strip() or not credit_txn_id.strip():
    return {"error": "Transaction IDs cannot be empty", "status": "invalid"}

# post_adjustment: reject unknown reason codes
VALID_REASONS = {"fx_variance", "liability_recognition", "inventory_loss", "manual_true_up"}
if reason_code not in VALID_REASONS:
    return {"error": f"Invalid reason_code: {reason_code}. Must be one of: {VALID_REASONS}", "status": "invalid"}

# post_adjustment: reject non-positive amounts
try:
    amt = Decimal(str(amount))
    if amt <= 0:
        return {"error": "Amount must be positive", "status": "invalid"}
except:
    return {"error": "Invalid amount format", "status": "invalid"}

# calculate_fx: validate currency codes
VALID_CURRENCIES = {"USD", "GBP", "EUR"}
if source_currency not in VALID_CURRENCIES or target_currency not in VALID_CURRENCIES:
    return {"error": f"Invalid currency. Must be one of: {VALID_CURRENCIES}", "status": "invalid"}

# calculate_fx: validate date format
try:
    date.fromisoformat(conversion_date)
except ValueError:
    return {"error": "Invalid date format. Use ISO format: YYYY-MM-DD", "status": "invalid"}
```

### 2. Loop Detection Enhancement (`services/audit_service.py`)

```python
def detect_loops(ctx: EpisodeContext, window: int = 5) -> bool:
    """Enhanced loop detection with multiple patterns."""
    if len(ctx.audit_log) < window:
        return False

    recent = ctx.audit_log[-window:]

    # Pattern 1: Exact same action repeated
    first = recent[0]
    if all(e.action_type == first.action_type and e.detail == first.detail for e in recent):
        return True

    # Pattern 2: Alternating between two actions (A-B-A-B-A)
    if window >= 4:
        types = [e.action_type for e in recent]
        if len(set(types)) == 2:
            if all(types[i] == types[i % 2] for i in range(len(types))):
                return True

    # Pattern 3: Only read actions for many consecutive steps (no write progress)
    read_only_window = 10
    if len(ctx.audit_log) >= read_only_window:
        last_n = ctx.audit_log[-read_only_window:]
        write_actions = {"execute_match", "post_adjustment", "execute_elimination"}
        if not any(e.action_type in write_actions for e in last_n):
            return True

    return False
```

### 3. Hallucinated ID Protection (already in services, verify completeness)

Every service that accepts an ID must return a clear error:

```python
# In ledger_service: already handles missing txn_ids
# In document_service: already handles missing doc_ids
# In matching_service.execute_match: already validates both txn_ids
# In matching_service.execute_elimination: already validates match_id

# Add: reject IDs with suspicious patterns (e.g., very long strings, special chars)
def validate_id(id_str: str, prefix: str = "") -> str | None:
    """Return error message if ID looks hallucinated."""
    if len(id_str) > 100:
        return f"ID too long (max 100 chars): {id_str[:20]}..."
    if not id_str.strip():
        return "ID cannot be empty"
    # IDs in our system follow patterns: TXN-X-NNN-D, DOC-X-NNN, MATCH-XXXX, ELIM-XXXX
    return None
```

### 4. Cross-Entity Write Protection

Prevent the agent from posting adjustments to entities not in the scenario:

```python
# Already in matching_service.post_adjustment — verify_entity check
# Strengthen: verify entity against scenario's known entities, not just ledger lines
known_entities = set()
for line_data in ctx.scenario.ledger_lines:
    known_entities.add(line_data.get("entity_id", ""))
    known_entities.add(line_data.get("counterparty_entity_id", ""))
```

### 5. Double-Action Prevention

Prevent the same match from being executed twice:

```python
# In matching_service.execute_match, add:
for existing_match in ctx.matches.values():
    if (existing_match.debit_txn_id == debit_txn_id and
        existing_match.credit_txn_id == credit_txn_id):
        return {"error": "This match already exists", "match_id": existing_match.match_id, "status": "rejected"}
```

### 6. Episode Termination Robustness

Update `_check_done()` in `server/environment.py`:

```python
def _check_done(self) -> bool:
    ctx = self._ctx

    # 1. Step limit exceeded
    if ctx.step_count >= ctx.scenario.step_limit:
        return True

    # 2. All objectives completed
    gt = ctx.ground_truth
    matches_done = len(ctx.matches) >= gt.total_expected_matches
    elims_done = len(ctx.eliminations) >= gt.total_expected_eliminations
    adjs_done = len(ctx.adjustments) >= gt.total_expected_adjustments

    if matches_done and elims_done and adjs_done:
        return True

    # 3. Catastrophic failure: too many invalid actions
    if ctx.invalid_action_count >= 20:
        return True

    return False
```

### 7. Grader Determinism Verification

Add a test that runs the same scenario twice and asserts identical scores:

```python
# tests/test_determinism.py

def test_easy_grader_determinism():
    """Same scenario, same actions → same score, always."""
    for _ in range(3):
        ctx = load_and_run_easy_scenario("easy_smoke")
        score = EasyGrader().score(ctx)
        assert score == expected_score  # Exact float equality

def test_medium_grader_determinism():
    # Same pattern for medium

def test_hard_grader_determinism():
    # Same pattern for hard
```

### 8. Malformed Action Handling

The MCPEnvironment already catches tool call errors. But add safety in our `step()` override:

```python
def step(self, action, timeout_s=None, **kwargs):
    if self._done:
        return Observation(done=True, reward=0.0,
            metadata={"error": "Episode already finished. Call reset()."})

    if not self._ctx:
        return Observation(done=True, reward=0.0,
            metadata={"error": "No active episode. Call reset() first."})

    self._ctx.step_count += 1

    try:
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
    except Exception as e:
        # Never crash — return error observation
        self._ctx.invalid_action_count += 1
        audit_service.record_event(self._ctx, "environment", "step_error", "invalid",
                                   detail=str(e)[:200])
        obs = Observation(done=False, reward=-0.05,
            metadata={"error": f"Action failed: {str(e)[:200]}"})

    # ... reward + done injection as before
```

### 9. Decimal Serialization Safety

Ensure all Decimal values serialize to JSON without errors:

```python
# In Money model, add json serializer:
class Money(BaseModel):
    model_config = ConfigDict(json_encoders={Decimal: str})
    amount: Decimal
    currency: Currency
```

Or use Pydantic v2's `model_dump(mode="json")` which handles Decimal → float automatically. Since we use `mode="json"` in service return dicts already, this should work. Add a test to confirm.

## Files Modified

| File | Changes |
|------|---------|
| `server/environment.py` | Input validation in tool closures, robust `step()`, `_check_done()` enhancement |
| `services/audit_service.py` | Enhanced loop detection patterns |
| `services/matching_service.py` | Double-action prevention, entity validation |
| `domain/money.py` | Decimal JSON serialization config |
| `tests/test_determinism.py` | New: determinism verification tests |
| `tests/test_edge_cases.py` | New: malformed input, hallucinated IDs, exploit patterns |

## Validation Gate

```bash
# Full test suite
uv run python -m pytest tests/ -v

# Specific hardening tests:
uv run python -m pytest tests/test_determinism.py -v
uv run python -m pytest tests/test_edge_cases.py -v

# Manual exploit testing:
# 1. Send 50 identical query_open_items calls → should trigger loop detection
# 2. Send execute_match with hallucinated txn IDs → should get rejected
# 3. Send post_adjustment with negative amount → should get rejected
# 4. Send 20 invalid actions → episode should terminate
```
