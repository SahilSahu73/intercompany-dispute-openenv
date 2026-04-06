"""
Public models for the Intercompany Dispute Environment.

Action types: CallToolAction, ListToolsAction (from OpenEnv MCP types)
Observation types: CallToolObservation, ListToolsObservation (from OpenEnv MCP types)
State type: FinanceDisputeState (custom)
"""

from typing import Literal

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class FinanceDisputeState(State):
    """Public episode state exposed via GET /state.

    Must NEVER contain hidden grader truth (ground truth checklists,
    expected matches, expected FX rates, etc.).
    """

    task_id: str = ""
    scenario_id: str = ""
    difficulty: Literal["easy", "medium", "hard", ""] = ""
    step_limit: int = 0
    completed_objectives: list[str] = Field(default_factory=list)
    remaining_objectives: list[str] = Field(default_factory=list)
    violations_count: int = 0
