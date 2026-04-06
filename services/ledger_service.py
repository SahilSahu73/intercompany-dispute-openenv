"""Ledger service: query open items and account balances."""

from decimal import Decimal

from domain.ledger_models import OpenItemView
from domain.scenario_models import EpisodeContext


def query_open_items(
    ctx: EpisodeContext,
    entity_id: str | None = None,
    counterparty_entity_id: str | None = None,
    status: str = "open",
    limit: int = 100,
) -> dict:
    """Return open items matching the filter criteria.

    Returns:
        {
            "items": [OpenItemView as dict, ...],
            "total_count": int,
            "returned_count": int,
        }
    """
    items = []
    for line in ctx.ledger_lines.values():
        if entity_id and line.entity_id != entity_id:
            continue
        if counterparty_entity_id and line.counterparty_entity_id != counterparty_entity_id:
            continue
        if line.status != status:
            continue
        items.append(OpenItemView.from_ledger_line(line))

    total = len(items)
    items = items[:limit]

    return {
        "items": [item.model_dump(mode="json") for item in items],
        "total_count": total,
        "returned_count": len(items),
    }


def query_ledger_balance(
    ctx: EpisodeContext,
    entity_id: str,
    account_code: str,
) -> dict:
    """Return the net balance for an entity's account.

    Sums debit amounts as positive, credit amounts as negative.
    Excludes eliminated transactions.

    Returns:
        {
            "entity_id": str,
            "account_code": str,
            "debit_total": str (Decimal),
            "credit_total": str (Decimal),
            "net_balance": str (Decimal),
            "currency": str,
        }
    """
    debit_total = Decimal("0")
    credit_total = Decimal("0")
    currency = None

    for line in ctx.ledger_lines.values():
        if line.entity_id != entity_id or line.account_code != account_code:
            continue
        if line.status == "eliminated":
            continue
        currency = currency or line.money.currency
        if line.side == "debit":
            debit_total += line.money.amount
        else:
            credit_total += line.money.amount

    return {
        "entity_id": entity_id,
        "account_code": account_code,
        "debit_total": str(debit_total),
        "credit_total": str(credit_total),
        "net_balance": str(debit_total - credit_total),
        "currency": currency or "USD",
    }
