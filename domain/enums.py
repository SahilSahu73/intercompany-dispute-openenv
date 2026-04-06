"""Enumerations and literal type aliases for the financial domain."""

from typing import Literal

# Entity identifiers used across the simulated multinational
EntityId = Literal["US_PARENT", "UK_SUB", "DE_SUB"]

# ISO currency codes we support
Currency = Literal["USD", "GBP", "EUR"]

# Ledger sides
Side = Literal["debit", "credit"]

# Transaction lifecycle
TxnStatus = Literal["open", "matched", "adjusted", "eliminated"]

# Document types available in the environment
DocumentType = Literal["invoice", "email", "contract", "shipment_report"]

# Reason codes for adjustments
ReasonCode = Literal["fx_variance", "liability_recognition", "inventory_loss", "manual_true_up"]

# Task difficulty levels
Difficulty = Literal["easy", "medium", "hard"]

# Action outcome status
ActionStatus = Literal["ok", "rejected", "invalid", "noop"]

# Incoterms relevant to the hard task
Incoterm = Literal["CIF", "FOB", "EXW", "DDP"]

# Actors in the audit log
Actor = Literal["orchestrator", "ledger_service", "treasury_service", "legal_service", "environment"]

# Account codes used in the simulated chart of accounts
# Format: 4-digit codes grouped by type
# 1xxx = Assets, 2xxx = Liabilities, 3xxx = Equity, 4xxx = Revenue, 5xxx = Expenses
# Intercompany-specific:
#   1300 = Intercompany Receivable
#   2300 = Intercompany Payable
#   1400 = Inventory
#   5100 = Cost of Goods Sold
#   6100 = FX Gain/Loss
#   9100 = Elimination (contra)
VALID_ACCOUNT_CODES = frozenset([
    "1100",  # Cash
    "1200",  # Accounts Receivable
    "1300",  # Intercompany Receivable
    "1400",  # Inventory
    "2100",  # Accounts Payable
    "2300",  # Intercompany Payable
    "3100",  # Retained Earnings
    "4100",  # Revenue
    "5100",  # Cost of Goods Sold
    "5200",  # Operating Expenses
    "6100",  # FX Gain/Loss
    "9100",  # Elimination
])
