"""Episode state tracker — tracks fetched docs, pending eliminations, FX queries."""

import json


class EpisodeTracker:
    """Lightweight state that persists within a single episode to guide the LLM."""

    def __init__(self, initial_context: str):
        self.initial_context = initial_context
        self.fetched_docs: set[str] = set()
        self.fx_queried: bool = False
        self.pending_eliminations: dict[str, str] = {}  # match_id -> entity_id
        self.completed_match_pairs: list[tuple[str, str]] = []  # (debit, credit)
        self.completed_eliminations: set[str] = set()  # match_ids
        self.consecutive_same: int = 0
        self._last_action_key: str = ""

    def update(self, tool_name: str, arguments: dict, reward: float,
               result_json: str) -> None:
        """Update tracker after a step."""
        # Track fetched documents
        if tool_name == "fetch_document":
            doc_id = arguments.get("document_id", "")
            if doc_id:
                self.fetched_docs.add(doc_id)

        # Track FX queries
        if tool_name == "calculate_fx":
            self.fx_queried = True

        # Track pending eliminations from successful matches
        if tool_name == "execute_match" and reward > 0:
            debit = arguments.get("debit_txn_id", "")
            credit = arguments.get("credit_txn_id", "")
            if debit and credit:
                self.completed_match_pairs.append((debit, credit))
            try:
                data = json.loads(result_json)
                pair_id = data.get("match_id") or data.get("matched_pair_id")
                entity_id = self._resolve_entity(debit)
                if pair_id:
                    self.pending_eliminations[pair_id] = entity_id or "US_PARENT"
            except (json.JSONDecodeError, AttributeError):
                pass

        # Remove from pending after successful elimination
        if tool_name == "execute_elimination" and reward > 0:
            pair_id = arguments.get("matched_pair_id")
            if pair_id:
                self.pending_eliminations.pop(pair_id, None)
                self.completed_eliminations.add(pair_id)

        # Track consecutive identical actions for loop detection
        action_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        if action_key == self._last_action_key:
            self.consecutive_same += 1
        else:
            self.consecutive_same = 0
            self._last_action_key = action_key

    def _resolve_entity(self, debit_txn_id: str) -> str:
        """Look up entity_id from the initial context text."""
        for line in self.initial_context.splitlines():
            if debit_txn_id in line and "entity=" in line:
                return line.split("entity=")[1].split()[0]
        return ""

    def build_directives(self) -> str:
        """Build urgent directive blocks for the prompt based on current state."""
        parts: list[str] = []

        if self.pending_eliminations:
            elim_lines = "\n".join(
                f'  {{"tool_name":"execute_elimination","arguments":'
                f'{{"entity_id":"{eid}","matched_pair_id":"{pid}"}}}}'
                for pid, eid in self.pending_eliminations.items()
            )
            parts.append(f"ACTION REQUIRED — execute_elimination next:\n{elim_lines}")

        if self.completed_match_pairs:
            pairs_str = ", ".join(f"{d}↔{c}" for d, c in self.completed_match_pairs)
            parts.append(f"ALREADY MATCHED (do NOT re-match): {pairs_str}")

        if self.fetched_docs:
            parts.append(
                f"ALREADY FETCHED (do NOT re-fetch): {', '.join(sorted(self.fetched_docs))}"
            )

        if self.consecutive_same >= 2:
            parts.append(
                "WARNING: You are repeating the same action. "
                "Try a DIFFERENT tool: post_adjustment, execute_match, or execute_elimination."
            )

        return "\n\n".join(parts)
