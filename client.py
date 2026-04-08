"""
Client for the Intercompany Dispute Environment.

Usage:
    from client import IntercompanyDisputeClient
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    # Connect to a running server (WS or WSS)
    env = IntercompanyDisputeClient(base_url="ws://localhost:8000").sync()
    with env:
        result = env.reset(task_id="easy_batch_matching")
        tools_result = env.step(ListToolsAction())
        result = env.step(CallToolAction(
            tool_name="query_open_items",
            arguments={"entity_id": "US_PARENT"}
        ))

    # Or spin up a local Docker container:
    import asyncio
    env = asyncio.run(IntercompanyDisputeClient.from_docker_image("intercompany-dispute-env"))
"""

from typing import Any, Dict, Union

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Observation

from models import FinanceDisputeState, ResetObservation


class IntercompanyDisputeClient(
    EnvClient[
        Union[CallToolAction, ListToolsAction],
        Union[CallToolObservation, ListToolsObservation, ResetObservation],
        FinanceDisputeState,
    ]
):
    """Typed WebSocket client for the Intercompany Dispute Environment."""

    def _step_payload(self, action: Union[CallToolAction, ListToolsAction]) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Parse the WS response payload.

        The server wraps observation fields under an "observation" key:
            {"observation": {...obs fields...}, "reward": float, "done": bool}
        """
        done = payload.get("done", False)
        reward = payload.get("reward", 0.0)

        # Server sends: {"observation": {...}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", {})

        if "tools" in obs_data:
            obs = ListToolsObservation.model_validate(obs_data)
        elif "tool_name" in obs_data:
            obs = CallToolObservation.model_validate(obs_data)
        elif "task_id" in obs_data or "description" in obs_data:
            # Reset observation with task context fields
            obs = ResetObservation.model_validate(obs_data)
        else:
            # Fallback: bare observation (e.g., error states)
            obs = Observation(done=done, reward=reward)

        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> FinanceDisputeState:
        return FinanceDisputeState.model_validate(payload)
