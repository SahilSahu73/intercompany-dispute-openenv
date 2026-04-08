"""System prompt and per-step prompt builder for the LLM agent."""

import json

SYSTEM_PROMPT = """\
You are an Enterprise Consolidation Orchestrator resolving intercompany disputes.

WORKFLOW — follow in order, do NOT repeat completed steps:
1. query_open_items → see transactions (entity_id like US_PARENT, UK_SUB, DE_SUB)
2. fetch_document → read each document_id ONCE
3. calculate_fx → FX rate on settlement date (medium tasks only)
4. ask_legal_analyst → liability from contract Incoterms (hard tasks only)
5. post_adjustment → record variance/liability entry
6. execute_match → pair debit + credit txn_ids → returns match_id
7. execute_elimination → use match_id as matched_pair_id

FX VARIANCE (medium): variance ≈ amount × |settlement_rate - booking_rate|. \
Post with reason_code=fx_variance, debit=6100, credit=9100.

LIABILITY (hard): ask_legal_analyst with the CONTRACT doc. Post with the liable \
entity_id, reason_code=inventory_loss or liability_recognition.

EASY TASKS: Just match + eliminate each debit/credit pair. No adjustments needed.

RULES:
- NEVER re-fetch a document or re-call the same calculate_fx
- NEVER call list_tools
- After execute_match, IMMEDIATELY call execute_elimination with the match_id
- Use EXACT parameter names from the tool signatures

Reply with ONE JSON object: {"tool_name":"<name>","arguments":{...}}"""


def format_tool_schema(tool) -> str:
    """Format a single MCP tool as a compact signature string."""
    schema = getattr(tool, "input_schema", None) or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    params = []
    for name, prop in props.items():
        ptype = prop.get("type", "any")
        opt = "" if name in required else "?"
        params.append(f"{name}{opt}:{ptype}")

    sig = ", ".join(params)
    desc = (tool.description or "").split("\n")[0][:80]
    return f"  {tool.name}({sig}) — {desc}"


def build_user_prompt(
    step: int,
    max_steps: int,
    initial_context: str,
    tools_info: str,
    last_result: str,
    history: list[str],
    directives: str,
    last_reward: float | None = None,
) -> str:
    """Build the single user message sent each step."""
    parts = [f"Step {step}/{max_steps}"]

    if last_reward is not None and step > 1:
        if last_reward >= 0.09:
            signal = "GOOD — action succeeded"
        elif last_reward > 0:
            signal = "OK — evidence gathered"
        elif last_reward < -0.05:
            signal = "BAD — rejected/loop penalty. Change your action!"
        else:
            signal = "neutral step cost"
        parts.append(f"LAST REWARD: {last_reward:+.2f} ({signal})")

    if directives:
        parts.append(directives)

    parts.append(f"TASK:\n{initial_context}")
    parts.append(f"TOOLS:\n{tools_info}")
    parts.append(f"LAST RESULT:\n{last_result[:600]}")

    if history:
        parts.append("HISTORY:\n" + "\n".join(history[-5:]))

    parts.append("Next action (one JSON object):")
    return "\n\n".join(parts)


def extract_initial_context(obs_or_meta) -> str:
    """Extract compact task context from reset observation or metadata dict.

    Accepts either:
    - A ResetObservation (client/WS mode) with direct attributes
    - A dict (in-process metadata dict)
    - Any observation with a .metadata dict
    """
    if isinstance(obs_or_meta, dict):
        meta = obs_or_meta
    else:
        # Try direct attributes first (ResetObservation via WS client)
        meta: dict = {}
        for field in ("description", "objectives", "open_items_preview", "available_document_ids"):
            val = getattr(obs_or_meta, field, None)
            if val:
                meta[field] = val
        # Fall back to .metadata dict (in-process or base Observation)
        if not meta:
            meta = getattr(obs_or_meta, "metadata", {}) or {}

    parts = []
    if "description" in meta:
        parts.append(meta["description"][:300])
    if "objectives" in meta:
        parts.append(f"Goals: {json.dumps(meta['objectives'])}")
    if "open_items_preview" in meta:
        preview = meta["open_items_preview"]
        parts.append(f"Open items ({preview.get('total_count', 0)}):")
        for item in preview.get("items", []):
            parts.append(
                f"  {item.get('txn_id')}  entity={item.get('entity_id')}  "
                f"side={item.get('side')}  "
                f"{item.get('money', {}).get('amount')} {item.get('money', {}).get('currency')}"
            )
    if "available_document_ids" in meta:
        parts.append(f"Docs: {meta['available_document_ids']}")
    return "\n".join(parts)


def extract_tool_result(obs) -> str:
    """Extract actual tool result content from a step observation.

    Works with both in-process CallToolResult objects and dict results
    from WS client serialization.
    """
    result = getattr(obs, "result", None)
    if result is None:
        return "(no result)"

    # Handle dict result (came through WS wire serialization)
    if isinstance(result, dict):
        sc = result.get("structured_content")
        if sc and isinstance(sc, dict):
            return json.dumps(sc, indent=2)[:600]
        data = result.get("data")
        if data is not None:
            return json.dumps(data, indent=2)[:600]
        for item in result.get("content", []):
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                try:
                    return json.dumps(json.loads(text), indent=2)[:600]
                except (json.JSONDecodeError, TypeError):
                    return str(text)[:600]
        return json.dumps(result, indent=2)[:600]

    # Handle in-process CallToolResult object
    content = None
    if hasattr(result, "structured_content") and result.structured_content:
        content = result.structured_content
    elif hasattr(result, "data") and result.data is not None:
        content = result.data
    else:
        for item in (getattr(result, "content", None) or []):
            text = getattr(item, "text", None)
            if text:
                try:
                    content = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return str(text)[:600]
                break
    if content is not None:
        return json.dumps(content, indent=2)[:600]
    return "(empty)"
