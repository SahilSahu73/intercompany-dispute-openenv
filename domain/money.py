"""Money type with Decimal precision for accounting arithmetic."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated

from pydantic import BaseModel, ConfigDict, field_validator, PlainSerializer

from .enums import Currency

# Decimal serializes to string in JSON to preserve precision
DecimalStr = Annotated[Decimal, PlainSerializer(lambda x: str(x), return_type=str)]


class Money(BaseModel):
    """Monetary amount with currency.

    All financial amounts use Decimal to avoid floating-point errors.
    JSON serialization preserves decimal precision as strings.
    """

    model_config = ConfigDict()

    amount: Decimal
    currency: Currency

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v):
        if isinstance(v, float):
            # Convert float to string first to avoid float precision issues
            return Decimal(str(v))
        return Decimal(v)

    def round_to_cents(self) -> "Money":
        """Round to 2 decimal places using half-up rounding."""
        return Money(
            amount=self.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            currency=self.currency,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __abs__(self) -> "Money":
        return Money(amount=abs(self.amount), currency=self.currency)
