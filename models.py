"""
Public models for the Intercompany Dispute Environment.

Action types: CallToolAction, ListToolsAction (from OpenEnv MCP types)
Observation types: CallToolObservation, ListToolsObservation, ResetObservation (custom)
State type: FinanceDisputeState (custom)
"""

from typing import Literal, Optional

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ResetObservation(Observation):
    """Initial observation returned by reset().

    Fields are declared directly (not in metadata) so they survive
    WS wire serialization via serialize_observation().
    """

    task_id: str = Field(default="")
    scenario_id: str = Field(default="")
    description: str = Field(default="")
    objectives: list = Field(default_factory=list)
    available_document_ids: list = Field(default_factory=list)
    open_items_preview: dict = Field(default_factory=dict)


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
    terminal_task_score: Optional[float] = Field(default=None)
