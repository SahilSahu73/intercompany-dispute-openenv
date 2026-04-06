# Phase 9: Inference Script, Documentation, Docker, and HF Deployment

## Goal
Finish hackathon-required packaging: baseline inference script (using Groq + Llama via OpenAI SDK), comprehensive README, working Dockerfile, and deployment to Hugging Face Spaces.

## Expected Outcome
- `inference.py` runs all 3 tasks, produces standardized stdout, uses Groq API
- README covers all hackathon requirements
- `docker build && docker run` works
- `openenv validate --verbose` passes
- HF Space deploys and responds

## File-by-File Implementation

### 1. `inference.py` — Baseline Inference Script

Located at project root (required by hackathon rules: must be named `inference.py`).

```python
"""
Baseline Inference Script for Intercompany Dispute Environment
==============================================================
Uses Groq API with Llama model via the OpenAI SDK.

Environment Variables:
    API_BASE_URL    Groq API endpoint (default: https://api.groq.com/openai/v1)
    MODEL_NAME      Model to use (default: llama-3.1-8b-instant)
    API_KEY         Groq API key (or set GROQ_API_KEY)

STDOUT FORMAT (required by hackathon):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
from typing import List, Optional

from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
ENV_NAME = "intercompany_dispute_env"

TASKS = [
    {"task_id": "easy_batch_matching", "scenario_id": "smoke", "max_steps": 30},
    {"task_id": "medium_fx_variance", "scenario_id": "smoke", "max_steps": 50},
    {"task_id": "hard_liability_dispute", "scenario_id": "smoke", "max_steps": 30},
]

SYSTEM_PROMPT = """You are an Enterprise Consolidation Orchestrator agent.
You resolve intercompany accounting disputes by calling tools.

Your workflow:
1. Use list_tools to see available actions
2. Use query_open_items to see unresolved transactions
3. Use fetch_document to gather evidence from invoices/contracts
4. For FX issues: use calculate_fx to get the correct exchange rate
5. For legal issues: use ask_legal_analyst with contract documents
6. Use post_adjustment to correct discrepancies
7. Use execute_match to link debit/credit pairs
8. Use execute_elimination to finalize matched pairs

Always gather evidence before taking write actions.
Respond with exactly one tool call per turn in JSON format:
{"tool_name": "<name>", "arguments": {<args>}}
"""

# --- Logging ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rstr}", flush=True)


def parse_tool_call(response_text: str) -> tuple[str, dict]:
    """Extract tool_name and arguments from LLM response."""
    # Try to parse JSON from the response
    text = response_text.strip()

    # Try to find JSON object in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            tool_name = data.get("tool_name", "")
            arguments = data.get("arguments", {})
            return tool_name, arguments
        except json.JSONDecodeError:
            pass

    return "query_open_items", {}  # Fallback


def build_user_message(step: int, obs_metadata: dict, tools: list, history: list) -> str:
    """Build the user prompt from observation."""
    parts = [f"Step {step}/{obs_metadata.get('step_limit', '?')}"]

    if step == 1:
        parts.append(f"\nTask: {obs_metadata.get('description', 'Resolve disputes')}")
        parts.append(f"Objectives: {json.dumps(obs_metadata.get('objectives', []))}")
        parts.append(f"\nAvailable tools: {json.dumps([t['name'] for t in tools])}")

    if obs_metadata.get("open_items_preview"):
        items = obs_metadata["open_items_preview"]
        parts.append(f"\nOpen items: {json.dumps(items, indent=2)[:2000]}")

    if history:
        parts.append(f"\nRecent actions:\n" + "\n".join(history[-3:]))

    parts.append("\nRespond with a JSON tool call: {\"tool_name\": \"...\", \"arguments\": {...}}")
    return "\n".join(parts)


def run_task(client: OpenAI, env_client, task_config: dict):
    """Run a single task and log results."""
    task_id = task_config["task_id"]
    scenario_id = task_config["scenario_id"]
    max_steps = task_config["max_steps"]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False
    history = []

    try:
        # Reset
        result = env_client.reset(task_id=task_id, scenario_id=scenario_id)
        obs_metadata = result.observation.metadata or {}

        # Get tools
        from openenv.core.env_server.mcp_types import ListToolsAction
        tools_result = env_client.step(ListToolsAction())
        tools = [{"name": t.name, "description": t.description} for t in tools_result.observation.tools]

        for step in range(1, max_steps + 1):
            if result.done:
                break

            user_msg = build_user_message(step, obs_metadata, tools, history)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                response_text = '{"tool_name": "query_open_items", "arguments": {}}'

            tool_name, arguments = parse_tool_call(response_text)

            # Execute tool call
            from openenv.core.env_server.mcp_types import CallToolAction
            action = CallToolAction(tool_name=tool_name, arguments=arguments)
            result = env_client.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None
            if hasattr(result.observation, "error") and result.observation.error:
                error = str(result.observation.error)

            rewards.append(reward)
            steps_taken = step

            action_str = f"{tool_name}({json.dumps(arguments)[:100]})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward={reward:.2f}")

            obs_metadata = result.observation.metadata or {}

            if done:
                terminal_score = obs_metadata.get("terminal_task_score", 0.0)
                success = terminal_score > 0.3
                break

    finally:
        env_client.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main():
    """Run baseline inference on all 3 tasks."""
    if not API_KEY:
        print("ERROR: Set API_KEY or GROQ_API_KEY environment variable", flush=True)
        return

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_config in TASKS:
        # Create a fresh client for each task
        # Import here to allow running without the env server for testing
        from client import IntercompanyDisputeClient

        env_url = os.getenv("ENV_URL", "ws://localhost:8000")
        env_client = IntercompanyDisputeClient(base_url=env_url).sync()

        with env_client:
            run_task(llm_client, env_client, task_config)


if __name__ == "__main__":
    main()
```

### 2. `scripts/smoke_eval.py` — Quick Non-LLM Evaluation

Runs a scripted agent (no API calls) to verify the environment works:

```python
"""
Smoke evaluation: runs scripted actions against the environment.
No LLM needed. Tests that the environment, graders, and scoring work.

Usage:
    uv run python scripts/smoke_eval.py --task easy --scenario smoke
    uv run python scripts/smoke_eval.py --task medium --scenario smoke
    uv run python scripts/smoke_eval.py --task hard --scenario smoke
    uv run python scripts/smoke_eval.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.environment import IntercompanyDisputeEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


def run_easy(env):
    """Scripted easy task: query, match all pairs, eliminate."""
    obs = env.reset(task_id="easy_batch_matching", scenario_id="smoke")
    print(f"Reset: {obs.metadata.get('description')}")

    # List tools
    obs = env.step(ListToolsAction())
    print(f"Tools: {[t.name for t in obs.tools]}")

    # Query open items
    obs = env.step(CallToolAction(tool_name="query_open_items", arguments={"status": "open"}))
    items = obs.result.data if hasattr(obs.result, 'data') else {}
    print(f"Open items: {items.get('total_count', '?')}")

    # Match and eliminate each pair (read from seed data)
    seed_file = Path("seed_data/easy/smoke.json")
    with open(seed_file) as f:
        scenario = json.load(f)

    for pair in scenario["ground_truth"]["required_matches"]:
        debit_id, credit_id = pair

        obs = env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": debit_id, "credit_txn_id": credit_id}
        ))
        match_data = obs.result.data if hasattr(obs.result, 'data') else {}
        match_id = match_data.get("match_id", "")
        print(f"  Matched: {debit_id} <-> {credit_id} = {match_id} (reward={obs.reward:.2f})")

        if match_id:
            obs = env.step(CallToolAction(
                tool_name="execute_elimination",
                arguments={"entity_id": "US_PARENT", "matched_pair_id": match_id}
            ))
            print(f"  Eliminated: {match_id} (reward={obs.reward:.2f})")

        if obs.done:
            break

    print(f"\nFinal: done={obs.done}, terminal_score={obs.metadata.get('terminal_task_score', 'N/A')}")
    return obs


def run_medium(env):
    """Scripted medium task: fetch docs, query FX, adjust, match, eliminate."""
    obs = env.reset(task_id="medium_fx_variance", scenario_id="smoke")
    print(f"Reset: {obs.metadata.get('description')}")

    # Read scenario for ground truth actions
    seed_file = Path("seed_data/medium/smoke.json")
    with open(seed_file) as f:
        scenario = json.load(f)

    # Fetch all documents
    for doc in scenario["documents"]:
        obs = env.step(CallToolAction(
            tool_name="fetch_document",
            arguments={"document_id": doc["document_id"]}
        ))
        print(f"  Fetched: {doc['document_id']} (reward={obs.reward:.2f})")

    # Query FX rates for each settlement date
    for line in scenario["ledger_lines"]:
        if line.get("settlement_date") and line["side"] == "debit":
            obs = env.step(CallToolAction(
                tool_name="calculate_fx",
                arguments={
                    "source_currency": "USD",
                    "target_currency": "GBP",
                    "amount": str(line["money"]["amount"]),
                    "conversion_date": line["settlement_date"],
                }
            ))
            print(f"  FX query: {obs.reward:.2f}")

    # Post adjustments from ground truth
    for adj in scenario["ground_truth"]["required_adjustments"]:
        obs = env.step(CallToolAction(
            tool_name="post_adjustment",
            arguments={
                "entity_id": adj["entity_id"],
                "debit_account_code": adj["debit_account_code"],
                "credit_account_code": adj["credit_account_code"],
                "amount": adj["amount"],
                "currency": adj["currency"],
                "reason_code": adj["reason_code"],
                "evidence_refs": ",".join([d["document_id"] for d in scenario["documents"]]),
            }
        ))
        print(f"  Adjustment: {adj['entity_id']} {adj['amount']} (reward={obs.reward:.2f})")

    # Match and eliminate
    for pair in scenario["ground_truth"]["required_matches"]:
        obs = env.step(CallToolAction(
            tool_name="execute_match",
            arguments={"debit_txn_id": pair[0], "credit_txn_id": pair[1]}
        ))
        match_data = obs.result.data if hasattr(obs.result, 'data') else {}
        match_id = match_data.get("match_id", "")
        if match_id:
            obs = env.step(CallToolAction(
                tool_name="execute_elimination",
                arguments={"entity_id": "US_PARENT", "matched_pair_id": match_id}
            ))

        if obs.done:
            break

    print(f"\nFinal: done={obs.done}, terminal_score={obs.metadata.get('terminal_task_score', 'N/A')}")
    return obs


def run_hard(env):
    """Scripted hard task: fetch docs, consult legal, adjust."""
    obs = env.reset(task_id="hard_liability_dispute", scenario_id="smoke")
    print(f"Reset: {obs.metadata.get('description')}")

    seed_file = Path("seed_data/hard/smoke.json")
    with open(seed_file) as f:
        scenario = json.load(f)

    # Fetch shipment report
    for doc in scenario["documents"]:
        obs = env.step(CallToolAction(
            tool_name="fetch_document",
            arguments={"document_id": doc["document_id"]}
        ))
        print(f"  Fetched: {doc['document_id']} (reward={obs.reward:.2f})")

    # Consult legal
    if scenario["legal_truth"]:
        contract_id = scenario["legal_truth"]["contract_document_id"]
        obs = env.step(CallToolAction(
            tool_name="ask_legal_analyst",
            arguments={
                "document_id": contract_id,
                "question": "Who is liable for the damaged goods under this contract?"
            }
        ))
        print(f"  Legal consultation: {obs.reward:.2f}")

    # Post adjustments
    for adj in scenario["ground_truth"]["required_adjustments"]:
        evidence = ",".join([d["document_id"] for d in scenario["documents"]])
        obs = env.step(CallToolAction(
            tool_name="post_adjustment",
            arguments={
                "entity_id": adj["entity_id"],
                "debit_account_code": adj["debit_account_code"],
                "credit_account_code": adj["credit_account_code"],
                "amount": adj["amount"],
                "currency": adj["currency"],
                "reason_code": adj["reason_code"],
                "evidence_refs": evidence,
            }
        ))
        print(f"  Adjustment: {adj['entity_id']} (reward={obs.reward:.2f})")

        if obs.done:
            break

    print(f"\nFinal: done={obs.done}, terminal_score={obs.metadata.get('terminal_task_score', 'N/A')}")
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard"])
    parser.add_argument("--scenario", default="smoke")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    runners = {"easy": run_easy, "medium": run_medium, "hard": run_hard}

    tasks_to_run = list(runners.keys()) if args.all else [args.task]

    for task in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Running {task} task (scenario: {args.scenario})")
        print(f"{'='*60}\n")
        env = IntercompanyDisputeEnvironment()
        runners[task](env)


if __name__ == "__main__":
    main()
```

### 3. `README.md` — Full Documentation

```markdown
---
title: Intercompany Dispute Environment
emoji: "\U0001F4B0"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Intercompany Dispute Environment for OpenEnv

A multi-agent financial dispute resolution environment where an AI
Enterprise Consolidation Orchestrator autonomously navigates and resolves
complex intercompany accounting disputes across a simulated multinational
enterprise.

## Motivation

Intercompany reconciliation is a real-world task that costs multinational
companies thousands of hours annually. This environment simulates the
core challenge: matching transactions across entities, resolving currency
discrepancies, and determining legal liability in supply chain disputes.

## Tasks

| Task | Difficulty | Description | Step Limit |
|------|-----------|-------------|------------|
| `easy_batch_matching` | Easy | Match 20 clean 1-to-1 USD transactions | 80 |
| `medium_fx_variance` | Medium | Resolve FX variances from noisy invoices | 60 |
| `hard_liability_dispute` | Hard | Determine legal liability via Incoterms | 50 |

## Action Space (MCP Tools)

| Tool | Type | Description |
|------|------|-------------|
| `query_open_items` | Read | Query open intercompany transactions |
| `query_ledger_balance` | Read | Get account balance for an entity |
| `fetch_document` | Read | Retrieve invoice/contract/report text |
| `ask_legal_analyst` | Read | Consult legal sub-agent on contracts |
| `calculate_fx` | Read | Convert amounts using historical FX rates |
| `execute_match` | Write | Match a debit-credit transaction pair |
| `post_adjustment` | Write | Post a corrective journal entry |
| `execute_elimination` | Write | Eliminate a matched pair |

## Observation Space

Each tool call returns a `CallToolObservation` containing:
- `tool_name`: The tool that was called
- `result`: The tool's return value (dict)
- `done`: Whether the episode has ended
- `reward`: Dense step reward

On reset, the observation metadata includes task description, objectives,
available document IDs, and a preview of open items.

## Reward Design

**Step reward** (dense, per action):
- +0.10 per successful match
- +0.15 per successful elimination
- +0.05 per successful adjustment
- +0.02 per evidence-gathering action
- -0.01 base step cost
- -0.05 per invalid action
- -0.10 for detected loops

**Terminal score** (0.0–1.0): Weighted checklist comparing final state
to ground truth. Varies by task difficulty.

## Setup

```bash
# Install dependencies
uv sync

# Run locally
uv run --project . server

# Validate
uv run openenv validate --verbose

# Run smoke tests (no LLM needed)
uv run python scripts/smoke_eval.py --all
```

## Baseline Inference

Uses Groq API with Llama via the OpenAI SDK.

```bash
export GROQ_API_KEY="your-key-here"

# Start the environment server
uv run --project . server &

# Run inference
uv run python inference.py
```

**API Configuration:**
- `API_BASE_URL`: Groq endpoint (default: `https://api.groq.com/openai/v1`)
- `MODEL_NAME`: Model ID (default: `llama-3.1-8b-instant`)
- `API_KEY` / `GROQ_API_KEY`: API key

**Expected Baseline Scores:**
| Task | Expected Score Range |
|------|---------------------|
| Easy | 0.30 – 0.70 |
| Medium | 0.15 – 0.45 |
| Hard | 0.10 – 0.35 |

## Docker

```bash
docker build -t intercompany-dispute-env .
docker run -p 8000:8000 intercompany-dispute-env
```

## Architecture

The environment uses the MCPEnvironment pattern from OpenEnv.
All tools are registered as `@mcp.tool()` decorators, discoverable
via `ListToolsAction`, callable via `CallToolAction`.

Internal "sub-agents" (Legal Analyst, Treasury Specialist) are
modeled as deterministic service functions — not separate LLM
workers. Their APIs mirror future MCP tool contracts for easy
upgrade to real agent-backed services.
```

### 4. `server/Dockerfile` — Final Version

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY openenv.yaml ./

# Install dependencies first (cached layer)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY . .

# Install project itself
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "server"]
```

### 5. `openenv.yaml` — Final

```yaml
spec_version: 1
name: intercompany_dispute_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## Deployment Steps

```bash
# 1. Validate locally
uv run openenv validate --verbose

# 2. Run smoke tests
uv run python scripts/smoke_eval.py --all

# 3. Test Docker
docker build -t intercompany-dispute-env -f server/Dockerfile .
docker run -p 8000:8000 intercompany-dispute-env
# In another terminal: curl http://localhost:8000/health

# 4. Push to HF Spaces
uv run openenv push --repo-id <username>/intercompany-dispute-env

# 5. Run inference against deployed space
ENV_URL=wss://<username>-intercompany-dispute-env.hf.space \
GROQ_API_KEY=xxx \
uv run python inference.py
```

## Validation Checklist (Hackathon Requirements)

- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] HF Space deploys and responds to /health
- [ ] 3 tasks with graders (easy/medium/hard)
- [ ] All graders produce scores in [0.0, 1.0]
- [ ] Graders are deterministic
- [ ] Baseline inference script runs with Groq API
- [ ] Baseline produces reproducible scores
- [ ] README has: description, action/observation spaces, tasks, setup, baseline scores
- [ ] Reward function provides dense signal (not just binary)
- [ ] Hard task genuinely challenges frontier models
