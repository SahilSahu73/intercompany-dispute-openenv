# Phase 1: Project Scaffold + OpenEnv Skeleton

## Goal
Turn the repo root into a valid, runnable OpenEnv MCP environment that starts, serves health checks, and passes `openenv validate` — before any domain logic exists.

## Expected Outcome
- `uv run --project . server` starts a FastAPI app on port 8000
- `GET /health` returns 200
- `openenv validate --verbose` passes
- The package structure matches OpenEnv conventions

## Updated Project Structure After This Phase

```
.
├── __init__.py                  # Package root — exports Action/Obs/Client
├── models.py                    # Re-exports MCP types + custom State
├── client.py                    # EnvClient subclass for typed WS access
├── openenv.yaml                 # Environment metadata
├── README.md                    # Stub with HF Space YAML frontmatter
├── pyproject.toml               # Updated with deps + entry point
├── uv.lock                      # Regenerated
├── server/
│   ├── __init__.py
│   ├── app.py                   # create_app() + main()
│   └── Dockerfile               # Single-container for HF Spaces
├── domain/                      # Empty __init__.py placeholder
│   └── __init__.py
├── services/                    # Empty __init__.py placeholder
│   └── __init__.py
├── graders/                     # Empty __init__.py placeholder
│   └── __init__.py
├── tasks/                       # Empty __init__.py placeholder
│   └── __init__.py
├── seed_data/                   # Empty dirs
│   ├── easy/
│   ├── medium/
│   └── hard/
├── tests/
│   └── __init__.py
├── build_plan/                  # Existing planning docs
└── requirements_n_scheme/       # Existing hackathon docs
```

## File-by-File Implementation

### 1. `openenv.yaml`

```yaml
spec_version: 1
name: intercompany_dispute_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### 2. `pyproject.toml`

```toml
[project]
name = "intercompany-dispute-env"
version = "0.1.0"
description = "Multi-agent financial dispute resolution environment for OpenEnv"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "openenv-core>=0.2.2",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "fastmcp>=2.0.0",
    "pydantic>=2.0.0",
]

[project.scripts]
server = "server.app:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["."]
include = ["server*", "domain*", "services*", "graders*", "tasks*"]
```

**Important**: The `[project.scripts]` entry `server = "server.app:main"` allows `uv run --project . server` to work. The package find config ensures all sub-packages are included.

### 3. `models.py`

This file re-exports the MCP action/observation types from OpenEnv and defines our custom State.

```python
"""
Public models for the Intercompany Dispute Environment.

Action types: CallToolAction, ListToolsAction (from OpenEnv MCP types)
Observation types: CallToolObservation, ListToolsObservation (from OpenEnv MCP types)
State type: FinanceDisputeState (custom)
"""

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Literal


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
```

### 4. `client.py`

```python
"""
Client for the Intercompany Dispute Environment.

Usage:
    from intercompany_dispute_env import IntercompanyDisputeClient

    async with IntercompanyDisputeClient(base_url="ws://localhost:8000") as env:
        result = await env.reset(task_id="easy", seed=42)
        tools = await env.step(ListToolsAction())
        result = await env.step(CallToolAction(
            tool_name="query_open_items",
            arguments={"entity_id": "US_PARENT"}
        ))
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from .models import FinanceDisputeState

from typing import Any, Dict, Union


class IntercompanyDisputeClient(EnvClient[
    Union[CallToolAction, ListToolsAction],
    Union[CallToolObservation, ListToolsObservation],
    FinanceDisputeState,
]):
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
```

**Note**: The exact import path for `StepResult` may need adjustment. Check `from openenv.core.env_client import StepResult` or `from openenv.core.client_types import StepResult` during implementation. Verify with:
```bash
.venv/bin/python -c "from openenv.core.env_client import EnvClient; help(EnvClient._parse_result)"
```

### 5. `__init__.py` (repo root)

```python
"""Intercompany Dispute Environment for OpenEnv."""

from .models import (
    CallToolAction,
    CallToolObservation,
    FinanceDisputeState,
    ListToolsAction,
    ListToolsObservation,
)
from .client import IntercompanyDisputeClient
```

### 6. `server/__init__.py`

```python
"""Server package for the Intercompany Dispute Environment."""
```

### 7. `server/app.py`

```python
"""
FastAPI application for the Intercompany Dispute Environment.

Entry points:
    - `app`: The FastAPI ASGI application (used by openenv.yaml)
    - `main()`: CLI entry point (used by pyproject.toml [project.scripts])
"""

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

from .environment import IntercompanyDisputeEnvironment

app = create_app(
    env=IntercompanyDisputeEnvironment,
    action_cls=Action,           # MCP types auto-routed by deserialize_action()
    observation_cls=Observation,  # Base type — subtypes returned at runtime
    env_name="intercompany_dispute_env",
)


def main():
    """CLI entry point: `uv run --project . server`"""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
```

**Key detail**: `create_app` receives the *class* `IntercompanyDisputeEnvironment`, not an instance. Each WebSocket session gets a fresh instance. The `action_cls=Action` base type is correct — the framework's `deserialize_action()` auto-routes `call_tool`/`list_tools` type discriminators to the correct MCP classes.

### 8. `server/environment.py`

This is a **stub** that will be fully implemented in Phase 4. For Phase 1, it must be just enough to pass validation.

```python
"""
Stub environment for Phase 1 validation.
Will be replaced with full MCPEnvironment in Phase 4.
"""

from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolObservation,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State


class IntercompanyDisputeEnvironment(MCPEnvironment):
    """Intercompany financial dispute resolution environment.

    The agent is an Enterprise Consolidation Orchestrator that must
    resolve intercompany accounting disputes across a simulated
    multinational enterprise using tool calls.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("intercompany_dispute_env")

        # Placeholder tool — proves MCP wiring works
        @mcp.tool()
        def ping() -> str:
            """Health check tool. Returns 'pong'."""
            return "pong"

        super().__init__(mcp)

        self._episode_id: str | None = None
        self._step_count: int = 0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        import uuid
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        return Observation(
            done=False,
            reward=0.0,
            metadata={"message": "Environment reset. Use list_tools to see available actions."},
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (none expected for this environment)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": "Use CallToolAction or ListToolsAction."},
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )
```

### 9. `server/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY . .

# Install dependencies (frozen from uv.lock)
RUN uv sync --frozen --no-dev

# Expose OpenEnv port
EXPOSE 8000

# Start the environment server
CMD ["uv", "run", "server"]
```

### 10. `README.md` (stub with HF Space frontmatter)

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

> Full documentation will be added in Phase 9.
```

### 11. Placeholder `__init__.py` files

Create empty `__init__.py` in: `domain/`, `services/`, `graders/`, `tasks/`, `tests/`

Create empty directories: `seed_data/easy/`, `seed_data/medium/`, `seed_data/hard/`

## Validation Gate

Run these commands in order:

```bash
# 1. Regenerate lock file with new deps
uv sync

# 2. Start server (should bind to 0.0.0.0:8000)
uv run --project . server &
sleep 2

# 3. Health check
curl -s http://localhost:8000/health | python -m json.tool

# 4. Schema check
curl -s http://localhost:8000/schema | python -m json.tool

# 5. Kill server
kill %1

# 6. OpenEnv validation
uv run openenv validate --verbose
```

All must pass before moving to Phase 2.

## Common Pitfalls

- **Import path**: `server/environment.py` is imported as `server.environment` — make sure `server/__init__.py` exists.
- **`create_app` class vs instance**: Pass `IntercompanyDisputeEnvironment` (the class), NOT `IntercompanyDisputeEnvironment()` (an instance).
- **`fastmcp` dependency**: Must be installed. If `uv sync` fails, check that `fastmcp>=2.0.0` is in `pyproject.toml` dependencies.
- **Port conflict**: If 8000 is in use, change it in both `openenv.yaml` and `server/app.py`.
