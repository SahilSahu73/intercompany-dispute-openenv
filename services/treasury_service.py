"""Treasury service: deterministic FX rate lookup and conversion."""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from domain.scenario_models import EpisodeContext


def calculate_fx(
    ctx: EpisodeContext,
    source_currency: str,
    target_currency: str,
    amount: float | str,
    conversion_date: str,
) -> dict:
    """Look up the historical FX rate and compute the conversion.

    The rate table is seeded per scenario. Only rates on or before the
    conversion date are used (nearest prior date if exact date missing).

    Side effect: sets ctx.fx_queried = True

    Returns on success:
        {
            "source_currency": str,
            "target_currency": str,
            "conversion_date": str,
            "rate": str (Decimal),
            "source_amount": str (Decimal),
            "converted_amount": str (Decimal),
        }
    Returns on failure:
        {"error": str}
    """
    if source_currency == target_currency:
        amt = Decimal(str(amount))
        return {
            "source_currency": source_currency,
            "target_currency": target_currency,
            "conversion_date": conversion_date,
            "rate": "1.0",
            "source_amount": str(amt),
            "converted_amount": str(amt),
        }

    try:
        conv_date = date.fromisoformat(conversion_date)
    except ValueError:
        return {"error": f"Invalid date format: {conversion_date}. Use ISO format YYYY-MM-DD."}

    try:
        amt = Decimal(str(amount))
    except Exception:
        return {"error": f"Invalid amount: {amount}"}

    # Find the best matching rate: exact date or nearest prior
    best_rate: Decimal | None = None
    best_date: date | None = None

    for fx in ctx.fx_rates:
        if fx.source_currency != source_currency or fx.target_currency != target_currency:
            continue
        if fx.rate_date <= conv_date:
            if best_date is None or fx.rate_date > best_date:
                best_rate = fx.rate
                best_date = fx.rate_date

    if best_rate is None:
        # Try the inverse rate
        for fx in ctx.fx_rates:
            if fx.source_currency != target_currency or fx.target_currency != source_currency:
                continue
            if fx.rate_date <= conv_date:
                if best_date is None or fx.rate_date > best_date:
                    best_rate = Decimal("1") / fx.rate
                    best_date = fx.rate_date

    if best_rate is None:
        return {
            "error": (
                f"No FX rate found for {source_currency}/{target_currency} "
                f"on or before {conversion_date}"
            )
        }

    ctx.fx_queried = True

    converted = (amt * best_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return {
        "source_currency": source_currency,
        "target_currency": target_currency,
        "conversion_date": conversion_date,
        "rate": str(best_rate),
        "source_amount": str(amt),
        "converted_amount": str(converted),
    }
