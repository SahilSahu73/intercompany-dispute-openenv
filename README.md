---
title: Intercompany Dispute Environment
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Intercompany Dispute Environment

An OpenEnv environment where an AI **Enterprise Consolidation Orchestrator** resolves complex intercompany accounting disputes across a simulated multinational enterprise using Model Context Protocol (MCP) tools.

## Overview

Intercompany reconciliation costs multinational companies thousands of hours annually. This environment simulates the core challenge: matching transactions across entities, resolving currency discrepancies, and determining legal liability from shipping contracts.

The agent must reason across invoices, FX rates, and Incoterms — hallucinating IDs or skipping evidence results in lower scores.

## Tasks

| Task ID | Difficulty | Description | Step Limit |
|---------|-----------|-------------|------------|
| `easy_batch_matching` | Easy | Match 20 clean 1-to-1 USD transactions between US_PARENT and UK_SUB | 80 |
| `medium_fx_variance` | Medium | Resolve FX compliance adjustments from noisy invoices (USD/GBP) | 60 |
| `hard_liability_dispute` | Hard | Determine legal liability via Incoterms (CIF/FOB) for damaged goods | 50 |

## Action Space (MCP Tools)

| Tool | Type | Description |
|------|------|-------------|
| `query_open_items` | Read | List open intercompany transaction lines (filterable by entity, status) |
| `query_ledger_balance` | Read | Get debit/credit totals for an entity's account |
| `fetch_document` | Read | Retrieve invoice, contract, or shipment report text |
| `ask_legal_analyst` | Read | Consult legal sub-agent on contract Incoterms for liability |
| `calculate_fx` | Read | Convert amounts using seeded historical FX rates |
| `execute_match` | Write | Match a debit-credit transaction pair |
| `post_adjustment` | Write | Post a corrective journal entry (FX variance, inventory loss, etc.) |
| `execute_elimination` | Write | Eliminate a matched pair from consolidation |

## Observation Space

Each action returns a `CallToolObservation` with:
- `result`: Tool return value (dict with items, status, amounts, etc.)
- `reward`: Dense step reward (float)
- `done`: Episode termination flag

On reset, the observation metadata includes task description, objectives, available document IDs, and an open items preview.

## Reward Design

**Step rewards** (dense, per action):
- `+0.10` successful match
- `+0.15` successful elimination
- `+0.05` successful adjustment
- `+0.02` evidence-gathering (fetch_document, calculate_fx, ask_legal_analyst)
- `-0.01` base step cost (efficiency pressure)
- `-0.05` invalid / rejected action
- `-0.10` loop detected

**Terminal score** (0.0–1.0): Weighted checklist comparing final state to hidden ground truth.

| Task | Scoring Breakdown |
|------|--------------------|
| Easy | 50% match coverage + 40% elimination + 10% efficiency − penalties |
| Medium | 15% evidence + 20% FX query + 25% adjustment accuracy + 20% match + 10% elimination + 10% efficiency − penalties |
| Hard | 10% evidence + 20% legal consultation + 25% liable entity + 20% adjustment accuracy + 15% process order + 10% efficiency − penalties |

## Setup

```bash
# Install dependencies
uv sync

# Run locally
uv run server

# Validate (in another terminal)
uv run openenv validate --verbose

# Run scripted smoke tests (no LLM needed)
uv run python scripts/smoke_eval.py --all
```

## Baseline Inference

Uses Groq API (free tier) with Llama via the OpenAI SDK.

```bash
export GROQ_API_KEY="gsk_..."

# Run all 3 tasks
uv run python inference.py
```

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model ID |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | API endpoint |

**Expected baseline scores** (with scripted oracle agent):
| Task | Oracle Score | Expected LLM Range |
|------|--------------|--------------------|
| Easy | ~0.96 | 0.30 – 0.80 |
| Medium | ~0.96 | 0.15 – 0.50 |
| Hard | ~0.98 | 0.10 – 0.40 |

## Docker

```bash
# Build
docker build -t intercompany-dispute-env -f server/Dockerfile .

# Run
docker run -p 8000:8000 intercompany-dispute-env

# Health check
curl http://localhost:8000/health
```

## Architecture

```
server/
  environment.py   # MCPEnvironment: 8 @mcp.tool() closures, reward engine, grader dispatch
  app.py           # FastAPI app via create_app()
services/
  ledger_service   # query_open_items, query_ledger_balance
  document_service # fetch_document (evidence cache)
  treasury_service # calculate_fx (nearest-prior-date lookup)
  legal_service    # ask_legal_analyst (deterministic from LegalTruth)
  matching_service # execute_match, post_adjustment, execute_elimination
  audit_service    # record_event, detect_loops
graders/
  easy_grader.py   # Batch matching scorer
  medium_grader.py # FX variance scorer
  hard_grader.py   # Liability dispute scorer
seed_data/
  easy/smoke.json      # 5 pairs for unit testing
  easy/benchmark.json  # 20 pairs for benchmarking
  medium/smoke.json    # 3 FX disputes
  medium/benchmark.json
  hard/smoke.json      # 1 CIF dispute
  hard/benchmark.json  # 3 disputes (CIF/FOB mix)
```

Internal "sub-agents" (Legal Analyst, Treasury Specialist) are deterministic service functions — not separate LLM workers. Their APIs mirror future MCP tool contracts for easy upgrade.
