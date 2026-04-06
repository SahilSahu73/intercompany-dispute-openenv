"""
Client for the Intercompany Dispute Environment.

Usage:
    from client import IntercompanyDisputeClient
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    env = IntercompanyDisputeClient(base_url="ws://localhost:8000").sync()
    with env:
        result = env.reset(task_id="easy_batch_matching", seed=42)
        tools_obs = env.step(ListToolsAction())
        result = env.step(CallToolAction(
            tool_name="query_open_items",
            arguments={"entity_id": "US_PARENT"}
        ))
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

from models import FinanceDisputeState


class IntercompanyDisputeClient(
    EnvClient[
        Union[CallToolAction, ListToolsAction],
        Union[CallToolObservation, ListToolsObservation],
        FinanceDisputeState,
    ]
):
    """Typed WebSocket client for the Intercompany Dispute Environment."""

    def _step_payload(self, action: Union[CallToolAction, ListToolsAction]) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        done = payload.get("done", False)
        reward = payload.get("reward", 0.0)
        # Route based on presence of 'tools' key (ListToolsObservation)
        if "tools" in payload:
            obs = ListToolsObservation.model_validate(payload)
        else:
            obs = CallToolObservation.model_validate(payload)
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> FinanceDisputeState:
        return FinanceDisputeState.model_validate(payload)
