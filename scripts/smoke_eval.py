"""
Smoke evaluation: runs scripted actions against the environment.
No LLM needed. Tests that the environment, graders, and scoring work.

Usage:
    uv run python scripts/smoke_eval.py --task easy
    uv run python scripts/smoke_eval.py --task medium
    uv run python scripts/smoke_eval.py --task hard
    uv run python scripts/smoke_eval.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from server.environment import IntercompanyDisputeEnvironment


def extract_result(obs) -> dict:
    """Extract result dict from CallToolObservation."""
    r = getattr(obs, "result", None)
    if r is None:
        return {}
    if hasattr(r, "structured_content") and r.structured_content:
        return r.structured_content
    if hasattr(r, "data") and r.data:
        return r.data
    content = getattr(r, "content", [])
    for item in content:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    return {}


def step(env, tool_name, **args):
    """Convenience wrapper."""
    return env.step(CallToolAction(tool_name=tool_name, arguments=args))


def run_easy(env, scenario: str = "smoke") -> dict:
    """Scripted easy task: match all pairs and eliminate."""
    obs = env.reset(task_id="easy_batch_matching", scenario_id=scenario)
    print(f"  Task: {obs.metadata.get('description', 'easy_batch_matching')[:80]}")
    print(f"  Step limit: {obs.metadata.get('step_limit')}")

    seed_file = Path(__file__).resolve().parent.parent / "seed_data" / "easy" / f"{scenario}.json"
    with open(seed_file) as f:
        scenario_data = json.load(f)

    # List tools
    obs = env.step(ListToolsAction())
    print(f"  Available tools: {[t.name for t in obs.tools]}")

    # Query open items
    obs = step(env, "query_open_items", status="open")
    r = extract_result(obs)
    print(f"  Open items: {r.get('total_count', '?')}")

    # Match and eliminate each pair
    for pair in scenario_data["ground_truth"]["required_matches"]:
        debit_id, credit_id = pair
        obs = step(env, "execute_match", debit_txn_id=debit_id, credit_txn_id=credit_id)
        match_data = extract_result(obs)
        match_id = match_data.get("match_id", "")
        print(f"    Match {debit_id} ↔ {credit_id}: {match_data.get('status')} reward={obs.reward:.2f}")

        if match_id:
            obs = step(env, "execute_elimination", entity_id="US_PARENT", matched_pair_id=match_id)
            print(f"    Eliminate {match_id}: {extract_result(obs).get('status')} reward={obs.reward:.2f}")

        if env._done:
            break

    final_meta = obs.metadata or {}
    score = final_meta.get("terminal_task_score", "N/A")
    print(f"  RESULT: done={obs.done}, terminal_score={score}")
    return {"done": obs.done, "terminal_score": score}


def run_medium(env, scenario: str = "smoke") -> dict:
    """Scripted medium task: fetch docs, query FX, adjust, match, eliminate."""
    obs = env.reset(task_id="medium_fx_variance", scenario_id=scenario)
    print(f"  Task: {obs.metadata.get('description', 'medium_fx_variance')[:80]}")

    seed_file = Path(__file__).resolve().parent.parent / "seed_data" / "medium" / f"{scenario}.json"
    with open(seed_file) as f:
        scenario_data = json.load(f)

    # Fetch all documents (evidence gathering)
    for doc in scenario_data["documents"]:
        obs = step(env, "fetch_document", document_id=doc["document_id"])
        print(f"    Fetched {doc['document_id']}: reward={obs.reward:.2f}")

    # Query FX rates from the fx_rates table (use settlement dates)
    # Settlement dates are: booking_date + 30 days (from invoice body)
    fx_dates_seen = set()
    for fx in scenario_data["fx_rates"]:
        date_key = (fx["source_currency"], fx["target_currency"], fx["rate_date"])
        if date_key in fx_dates_seen:
            continue
        if fx["rate_date"] not in [r["rate_date"] for r in scenario_data["fx_rates"]
                                    if r["rate_date"] < fx["rate_date"]]:
            # Use as a settlement date query
            obs = step(env, "calculate_fx",
                       source_currency=fx["source_currency"],
                       target_currency=fx["target_currency"],
                       amount="10000",
                       conversion_date=fx["rate_date"])
            r = extract_result(obs)
            if "rate" in r:
                print(f"    FX {fx['rate_date']}: rate={r.get('rate')} reward={obs.reward:.2f}")
                fx_dates_seen.add(date_key)

    # Post adjustments from ground truth
    for adj in scenario_data["ground_truth"]["required_adjustments"]:
        doc_ids = ",".join(d["document_id"] for d in scenario_data["documents"])
        obs = step(env, "post_adjustment",
                   entity_id=adj["entity_id"],
                   debit_account_code=adj["debit_account_code"],
                   credit_account_code=adj["credit_account_code"],
                   amount=adj["amount"],
                   currency=adj["currency"],
                   reason_code=adj["reason_code"],
                   evidence_refs=doc_ids)
        print(f"    Adjustment {adj['entity_id']} {adj['amount']}: {extract_result(obs).get('status')} reward={obs.reward:.2f}")

    # Match and eliminate
    for pair in scenario_data["ground_truth"]["required_matches"]:
        obs = step(env, "execute_match", debit_txn_id=pair[0], credit_txn_id=pair[1])
        match_data = extract_result(obs)
        match_id = match_data.get("match_id", "")
        print(f"    Match {pair[0]}: {match_data.get('status')} reward={obs.reward:.2f}")
        if match_id:
            obs = step(env, "execute_elimination", entity_id="US_PARENT", matched_pair_id=match_id)
            print(f"    Eliminate {match_id}: {extract_result(obs).get('status')} reward={obs.reward:.2f}")
        if env._done:
            break

    final_meta = obs.metadata or {}
    score = final_meta.get("terminal_task_score", "N/A")
    print(f"  RESULT: done={obs.done}, terminal_score={score}")
    return {"done": obs.done, "terminal_score": score}


def run_hard(env, scenario: str = "smoke") -> dict:
    """Scripted hard task: fetch docs, consult legal, post adjustment."""
    obs = env.reset(task_id="hard_liability_dispute", scenario_id=scenario)
    print(f"  Task: {obs.metadata.get('description', 'hard_liability_dispute')[:80]}")

    seed_file = Path(__file__).resolve().parent.parent / "seed_data" / "hard" / f"{scenario}.json"
    with open(seed_file) as f:
        scenario_data = json.load(f)

    # Fetch all documents
    for doc in scenario_data["documents"]:
        obs = step(env, "fetch_document", document_id=doc["document_id"])
        print(f"    Fetched {doc['document_id']} ({doc['document_type']}): reward={obs.reward:.2f}")

    # Consult legal analyst
    if scenario_data.get("legal_truth"):
        contract_id = scenario_data["legal_truth"]["contract_document_id"]
        obs = step(env, "ask_legal_analyst",
                   document_id=contract_id,
                   question="Who is liable for the damaged goods under this contract?")
        r = extract_result(obs)
        print(f"    Legal consultation ({contract_id}): {r.get('liable_entity_id', 'N/A')} liable, reward={obs.reward:.2f}")

    # Post adjustments from ground truth
    doc_ids = ",".join(d["document_id"] for d in scenario_data["documents"])
    for adj in scenario_data["ground_truth"]["required_adjustments"]:
        obs = step(env, "post_adjustment",
                   entity_id=adj["entity_id"],
                   debit_account_code=adj["debit_account_code"],
                   credit_account_code=adj["credit_account_code"],
                   amount=adj["amount"],
                   currency=adj["currency"],
                   reason_code=adj["reason_code"],
                   evidence_refs=doc_ids)
        print(f"    Adjustment {adj['entity_id']} {adj['amount']} {adj['currency']}: {extract_result(obs).get('status')} reward={obs.reward:.2f}")
        if env._done:
            break

    final_meta = obs.metadata or {}
    score = final_meta.get("terminal_task_score", "N/A")
    print(f"  RESULT: done={obs.done}, terminal_score={score}")
    return {"done": obs.done, "terminal_score": score}


RUNNERS = {
    "easy": run_easy,
    "medium": run_medium,
    "hard": run_hard,
}


def main():
    parser = argparse.ArgumentParser(description="Smoke evaluation for intercompany dispute env")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], help="Task to run")
    parser.add_argument("--scenario", default="smoke", help="Scenario name (default: smoke)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.error("Must specify --task or --all")

    tasks = list(RUNNERS.keys()) if args.all else [args.task]
    results = {}

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task} (scenario={args.scenario})")
        print("=" * 60)
        env = IntercompanyDisputeEnvironment()
        result = RUNNERS[task](env, scenario=args.scenario)
        results[task] = result

    if args.all:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        for task, res in results.items():
            print(f"  {task}: terminal_score={res.get('terminal_score', 'N/A')}")


if __name__ == "__main__":
    main()
