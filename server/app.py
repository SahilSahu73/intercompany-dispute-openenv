"""
FastAPI application for the Intercompany Dispute Environment.

Entry points:
    - `app`: The FastAPI ASGI application (referenced by openenv.yaml)
    - `main()`: CLI entry point (referenced by pyproject.toml [project.scripts])
"""

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

from server.environment import IntercompanyDisputeEnvironment

# create_app receives the CLASS (not an instance) so each WebSocket
# session gets its own isolated environment instance.
# action_cls=Action (base) is correct — deserialize_action() auto-routes
# "call_tool"/"list_tools" type discriminators to the MCP subclasses.
app = create_app(
    env=IntercompanyDisputeEnvironment,
    action_cls=Action,
    observation_cls=Observation,
    env_name="intercompany_dispute_env",
)


def main():
    """CLI entry point: `uv run server`"""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
