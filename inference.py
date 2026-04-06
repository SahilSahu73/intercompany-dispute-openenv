"""
Baseline Inference Script for Intercompany Dispute Environment
==============================================================
Uses Groq API with Llama model via the OpenAI SDK.

Environment Variables:
    API_BASE_URL    Groq API endpoint (default: https://api.groq.com/openai/v1)
    MODEL_NAME      Model to use (default: llama-3.3-70b-versatile)
    API_KEY         Groq API key (or set GROQ_API_KEY)
    ENV_URL         Environment server URL (default: ws://localhost:8000)

STDOUT FORMAT (required by hackathon):
    [START] task=<task_name> env=<env_name> model=<model_name>
    [STEP]  step=<n> action=<action> reward=<r> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> terminal_score=<score>
"""

import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_NAME = "intercompany_dispute_env"

TASKS = [
    {"task_id": "easy_batch_matching", "scenario_id": "smoke", "max_steps": 30},
    {"task_id": "medium_fx_variance", "scenario_id": "smoke", "max_steps": 50},
    {"task_id": "hard_liability_dispute", "scenario_id": "smoke", "max_steps": 30},
]

SYSTEM_PROMPT = """You are an Enterprise Consolidation Orchestrator agent.
Your job is to resolve intercompany accounting disputes using the available tools.

WORKFLOW:
1. list_tools — discover available tools
2. query_open_items — see unresolved transactions
3. fetch_document — gather evidence from invoices/contracts/reports
4. calculate_fx — get FX rates for currency variance disputes (use the settlement date from the invoice)
5. ask_legal_analyst — determine liability from contract Incoterms (required for hard disputes)
6. post_adjustment — record FX variance or liability adjustments (use correct reason_code)
7. execute_match — link matching debit/credit pairs
8. execute_elimination — finalize each matched pair

RULES:
- Always gather evidence BEFORE taking write actions
- For FX disputes: query the FX rate on the SETTLEMENT date (not booking date)
- For legal disputes: consult the legal analyst with the CONTRACT document_id
- Valid reason_codes: fx_variance, liability_recognition, inventory_loss, manual_true_up
- Valid account codes: 1100, 1200, 1300, 1400, 2100, 2300, 3100, 4100, 5100, 5200, 6100, 9100

Respond with EXACTLY one tool call in JSON format:
{"tool_name": "<name>", "arguments": {<args>}}

Nothing else. No explanation. Just the JSON."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error or "null"
    print(f"[STEP]  step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, terminal_score: float) -> None:
    print(f"[END]   success={str(success).lower()} steps={steps} terminal_score={terminal_score:.4f}", flush=True)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def parse_tool_call(response_text: str) -> tuple[str, dict]:
    """Extract tool_name and arguments from LLM response."""
    text = response_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            tool_name = data.get("tool_name", "")
            arguments = data.get("arguments", {})
            if tool_name:
                return tool_name, arguments
        except json.JSONDecodeError:
            pass
    # Fallback: just query open items
    return "query_open_items", {}


def extract_obs_summary(obs_metadata: dict, max_chars: int = 1500) -> str:
    """Build a concise summary of the observation for the LLM."""
    parts = []
    if "description" in obs_metadata:
        parts.append(f"Task: {obs_metadata['description'][:200]}")
    if "objectives" in obs_metadata:
        parts.append(f"Objectives: {json.dumps(obs_metadata['objectives'])}")
    if "open_items_preview" in obs_metadata:
        preview = obs_metadata["open_items_preview"]
        parts.append(f"Open items count: {preview.get('total_count', 0)}")
        items = preview.get("items", [])[:3]
        for item in items:
            parts.append(f"  - {item.get('txn_id')} {item.get('entity_id')} {item.get('side')} "
                         f"{item.get('money', {}).get('amount')} {item.get('money', {}).get('currency')}")
    if "available_document_ids" in obs_metadata:
        parts.append(f"Available docs: {obs_metadata['available_document_ids']}")
    if "error" in obs_metadata:
        parts.append(f"Error: {obs_metadata['error']}")
    return "\n".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# In-process execution (avoids WebSocket complexity)
# ---------------------------------------------------------------------------


def run_task_inprocess(llm_client, task_config: dict) -> dict:
    """Run a task directly against the env class (no HTTP/WebSocket needed)."""
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
    from server.environment import IntercompanyDisputeEnvironment

    task_id = task_config["task_id"]
    scenario_id = task_config["scenario_id"]
    max_steps = task_config["max_steps"]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    env = IntercompanyDisputeEnvironment()
    obs = env.reset(task_id=task_id, scenario_id=scenario_id)

    # Get tool listing
    tools_obs = env.step(ListToolsAction())
    tools_info = "\n".join(
        f"  - {t.name}: {t.description[:80]}" for t in (tools_obs.tools or [])
    )

    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    terminal_score = 0.0
    success = False

    obs_metadata = obs.metadata or {}

    for step_n in range(1, max_steps + 1):
        if env._done:
            break

        # Build prompt
        obs_summary = extract_obs_summary(obs_metadata)
        step_info = f"Step {step_n}/{max_steps}\n"
        history_str = "\n".join(history[-4:]) if history else "None yet."
        user_msg = (
            f"{step_info}"
            f"Available tools:\n{tools_info}\n\n"
            f"Current observation:\n{obs_summary}\n\n"
            f"Recent history:\n{history_str}\n\n"
            "Respond with exactly one JSON tool call."
        )

        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = '{"tool_name": "query_open_items", "arguments": {}}'
            print(f"  LLM error at step {step_n}: {e}", file=sys.stderr)

        tool_name, arguments = parse_tool_call(response_text)

        # Execute
        step_obs = env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        reward = step_obs.reward or 0.0
        done = step_obs.done

        error_msg = None
        step_meta = step_obs.metadata or {}
        if "error" in step_meta:
            error_msg = str(step_meta["error"])[:100]

        rewards.append(reward)
        steps_taken = step_n
        action_str = f"{tool_name}({json.dumps(arguments)[:80]})"
        log_step(step=step_n, action=action_str, reward=reward, done=done, error=error_msg)

        history.append(f"Step {step_n}: {action_str} reward={reward:.2f}")

        obs_metadata = step_meta

        if done:
            terminal_score = step_meta.get("terminal_task_score", 0.0)
            success = float(terminal_score) >= 0.30
            break

    log_end(success=success, steps=steps_taken, terminal_score=float(terminal_score))
    return {"task_id": task_id, "success": success, "steps": steps_taken,
            "terminal_score": terminal_score, "rewards": rewards}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        print("ERROR: Set API_KEY or GROQ_API_KEY environment variable", flush=True)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: uv add openai", flush=True)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    for task_config in TASKS:
        result = run_task_inprocess(llm_client, task_config)
        all_results.append(result)
        print()  # separator between tasks

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['task_id']}: score={r['terminal_score']:.4f} "
              f"steps={r['steps']} success={r['success']}")


if __name__ == "__main__":
    main()
