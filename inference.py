"""
Inference Script for Intercompany Dispute Environment
======================================================

Environment Variables:
    API_BASE_URL        LLM API endpoint
    MODEL_NAME          Model identifier
    HF_TOKEN            API key (also accepts API_KEY or GROQ_API_KEY)

    IMAGE_NAME          Docker image name → spins up container via Docker
    LOCAL_IMAGE_NAME    Alias for IMAGE_NAME
    ENV_URL             WebSocket URL of a running server (e.g. ws://localhost:8000
                        or wss://user-space.hf.space). Overrides IMAGE_NAME.
    USE_INPROCESS       Set to "true" to bypass client and run env in-process
                        (local dev only — does not test the Docker container).

STDOUT FORMAT (required by hackathon):
    [START] task=<task_name> env=<env_name> model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent import (
    SYSTEM_PROMPT,
    EpisodeTracker,
    build_user_prompt,
    extract_initial_context,
    extract_tool_result,
    format_tool_schema,
    log_end,
    log_start,
    log_step,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_NAME = "intercompany_dispute_env"

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL")
USE_INPROCESS = os.getenv("USE_INPROCESS", "").lower() in ("1", "true", "yes")

TASKS = [
    {"task_id": "easy_batch_matching",  "scenario_id": "smoke", "max_steps": 20},
    {"task_id": "medium_fx_variance",   "scenario_id": "smoke", "max_steps": 25},
    {"task_id": "hard_liability_dispute","scenario_id": "smoke", "max_steps": 15},
]

SUCCESS_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# In-process adapter (local dev fallback — not the Docker container)
# ---------------------------------------------------------------------------

class _InProcessAdapter:
    """Wraps in-process IntercompanyDisputeEnvironment with async EnvClient interface."""

    def __init__(self):
        from server.environment import IntercompanyDisputeEnvironment
        self._env = IntercompanyDisputeEnvironment()

    async def reset(self, **kwargs):
        from openenv.core.client_types import StepResult
        obs = self._env.reset(**kwargs)
        return StepResult(observation=obs, reward=obs.reward or 0.0, done=obs.done)

    async def step(self, action):
        from openenv.core.client_types import StepResult
        obs = self._env.step(action)
        return StepResult(observation=obs, reward=obs.reward or 0.0, done=obs.done)

    async def state(self):
        return self._env.state

    async def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

async def create_env():
    """Return an async env handle.

    Priority: ENV_URL > IMAGE_NAME > USE_INPROCESS > error.
    """
    if ENV_URL:
        from client import IntercompanyDisputeClient
        print(f"[DEBUG] Connecting to server: {ENV_URL}", file=sys.stderr, flush=True)
        client = IntercompanyDisputeClient(base_url=ENV_URL)
        await client.connect()
        return client

    if IMAGE_NAME:
        from client import IntercompanyDisputeClient
        print(f"[DEBUG] Starting Docker container: {IMAGE_NAME}", file=sys.stderr, flush=True)
        return await IntercompanyDisputeClient.from_docker_image(IMAGE_NAME)

    if USE_INPROCESS:
        print("[DEBUG] Using in-process env (Docker not tested)", file=sys.stderr, flush=True)
        return _InProcessAdapter()

    # Default: in-process for local dev convenience
    print(
        "[DEBUG] No IMAGE_NAME or ENV_URL set — running env in-process.\n"
        "        For Docker: set IMAGE_NAME=<image>  For running server: set ENV_URL=ws://localhost:8000",
        file=sys.stderr, flush=True,
    )
    return _InProcessAdapter()


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> tuple[str, dict]:
    """Extract {tool_name, arguments} JSON from LLM response text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            name = data.get("tool_name", "")
            args = data.get("arguments", {})
            if name:
                return name, args
        except json.JSONDecodeError:
            pass
    return "query_open_items", {}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

async def run_task(llm_client: OpenAI, task_config: dict, env) -> dict:
    """Run a single task episode against the environment."""
    task_id = task_config["task_id"]
    scenario_id = task_config["scenario_id"]
    max_steps = task_config["max_steps"]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    # Reset — result.observation is ResetObservation (direct fields) or base Observation (metadata)
    reset_result = await env.reset(task_id=task_id, scenario_id=scenario_id)
    initial_ctx = extract_initial_context(reset_result.observation)
    tracker = EpisodeTracker(initial_ctx)

    # Discover tools
    tools_result = await env.step(ListToolsAction())
    tools_info = "\n".join(
        format_tool_schema(t) for t in (getattr(tools_result.observation, "tools", None) or [])
    )

    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_result = "(none — call query_open_items first)"
    last_reward: float | None = None

    for step_n in range(1, max_steps + 1):
        user_msg = build_user_prompt(
            step=step_n,
            max_steps=max_steps,
            initial_context=initial_ctx,
            tools_info=tools_info,
            last_result=last_result,
            history=history,
            directives=tracker.build_directives(),
            last_reward=last_reward,
        )

        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = '{"tool_name": "query_open_items", "arguments": {}}'
            print(f"[DEBUG] LLM error step {step_n}: {e}", file=sys.stderr)

        tool_name, arguments = parse_tool_call(response_text)

        step_result = await env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        obs = step_result.observation
        reward = step_result.reward or 0.0
        done = step_result.done
        last_reward = reward

        # Extract error message for logging
        error_msg = None
        obs_error = getattr(obs, "error", None)
        if obs_error:
            # ToolError object (from WS client or in-process)
            error_msg = str(getattr(obs_error, "message", obs_error))[:100]
        if not error_msg:
            obs_meta = getattr(obs, "metadata", {}) or {}
            if "error" in obs_meta:
                error_msg = str(obs_meta["error"])[:100]

        last_result = extract_tool_result(obs)
        if error_msg:
            last_result = f"ERROR: {error_msg}\n{last_result}"

        tracker.update(tool_name, arguments, reward, last_result)
        rewards.append(reward)
        steps_taken = step_n
        action_str = f"{tool_name}({json.dumps(arguments)[:80]})"

        log_step(step=step_n, action=action_str, reward=reward, done=done, error=error_msg)
        history.append(f"Step {step_n}: {action_str} -> reward={reward:+.2f}")

        if done:
            # Try metadata first (in-process), then env.state() (WS client)
            obs_meta = getattr(obs, "metadata", {}) or {}
            terminal_score = obs_meta.get("terminal_task_score")
            if terminal_score is None:
                try:
                    state = await env.state()
                    terminal_score = getattr(state, "terminal_task_score", None)
                except Exception:
                    pass
            score = float(terminal_score or 0.0)
            success = score >= SUCCESS_THRESHOLD
            break

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN, API_KEY, or GROQ_API_KEY", flush=True)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await create_env()

    results = []
    try:
        for task_cfg in TASKS:
            result = await run_task(llm, task_cfg, env)
            results.append(result)
            print()
    finally:
        await env.close()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id']}: score={r['score']:.2f} "
              f"steps={r['steps']} success={r['success']}")


if __name__ == "__main__":
    asyncio.run(main())
