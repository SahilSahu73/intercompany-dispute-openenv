"""Tests for domain models — Phase 2 validation gate."""

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from domain.money import Money
from domain.ledger_models import LedgerLine, OpenItemView, JournalEntry, MatchRecord, EliminationRecord, AuditEvent
from domain.document_models import Document, Invoice, Contract, ShipmentReport
from domain.scenario_models import ScenarioBundle, GroundTruthChecklist, FxRate, LegalTruth, EpisodeContext
from models import FinanceDisputeState


# ── Money tests ──────────────────────────────────────────────────────────────

class TestMoney:
    def test_decimal_from_string(self):
        m = Money(amount="10000.00", currency="USD")
        assert isinstance(m.amount, Decimal)
        assert m.amount == Decimal("10000.00")

    def test_decimal_from_float_avoids_precision_loss(self):
        m = Money(amount=0.1, currency="USD")
        # float 0.1 has precision issues; our validator coerces via str
        assert m.amount == Decimal("0.1")

    def test_decimal_from_int(self):
        m = Money(amount=5000, currency="USD")
        assert m.amount == Decimal("5000")

    def test_round_to_cents(self):
        m = Money(amount="10000.555", currency="USD")
        rounded = m.round_to_cents()
        assert rounded.amount == Decimal("10000.56")

    def test_equality(self):
        a = Money(amount="100.00", currency="USD")
        b = Money(amount="100.00", currency="USD")
        c = Money(amount="100.00", currency="GBP")
        assert a == b
        assert a != c

    def test_abs(self):
        m = Money(amount="-500.00", currency="EUR")
        assert abs(m).amount == Decimal("500.00")

    def test_json_roundtrip(self):
        m = Money(amount="12345.67", currency="GBP")
        data = m.model_dump(mode="json")
        m2 = Money.model_validate(data)
        assert m == m2

    def test_currency_validation(self):
        with pytest.raises(Exception):
            Money(amount="100", currency="CHF")  # Not in Currency literal


# ── LedgerLine tests ─────────────────────────────────────────────────────────

class TestLedgerLine:
    def test_basic_construction(self):
        line = LedgerLine(
            txn_id="TXN-001",
            entity_id="US_PARENT",
            counterparty_entity_id="UK_SUB",
            account_code="1300",
            side="debit",
            money=Money(amount="10000.00", currency="USD"),
            booking_date=date(2024, 1, 15),
            description="IC Receivable",
        )
        assert line.txn_id == "TXN-001"
        assert line.status == "open"  # default
        assert line.document_ids == []

    def test_from_dict(self):
        data = {
            "txn_id": "TXN-002",
            "entity_id": "UK_SUB",
            "counterparty_entity_id": "US_PARENT",
            "account_code": "2300",
            "side": "credit",
            "money": {"amount": "5000.00", "currency": "USD"},
            "booking_date": "2024-01-15",
            "description": "IC Payable",
        }
        line = LedgerLine.model_validate(data)
        assert line.money.amount == Decimal("5000.00")
        assert line.booking_date == date(2024, 1, 15)


class TestOpenItemView:
    def test_from_ledger_line(self):
        line = LedgerLine(
            txn_id="TXN-003",
            entity_id="US_PARENT",
            counterparty_entity_id="DE_SUB",
            account_code="1300",
            side="debit",
            money=Money(amount="20000.00", currency="EUR"),
            booking_date=date(2024, 2, 1),
            description="Test",
            document_ids=["DOC-001"],
        )
        view = OpenItemView.from_ledger_line(line)
        assert view.txn_id == line.txn_id
        assert view.money == line.money
        assert view.document_ids == ["DOC-001"]
        # settlement_date not in OpenItemView (hidden detail)
        assert not hasattr(view, "settlement_date")


# ── Document tests ────────────────────────────────────────────────────────────

class TestDocuments:
    def test_invoice_from_dict(self):
        data = {
            "document_id": "DOC-001",
            "document_type": "invoice",
            "title": "Test Invoice",
            "body": "Invoice body text",
        }
        doc = Invoice.model_validate(data)
        assert doc.document_id == "DOC-001"
        assert doc.document_type == "invoice"

    def test_contract_fields(self):
        contract = Contract(
            document_id="CON-001",
            document_type="contract",
            title="Transit Contract",
            body="CIF terms apply",
            incoterm="CIF",
            origin_entity_id="DE_SUB",
            destination_entity_id="UK_SUB",
        )
        assert contract.incoterm == "CIF"

    def test_shipment_report_fields(self):
        sr = ShipmentReport(
            document_id="SR-001",
            document_type="shipment_report",
            title="Damage Report",
            body="All goods damaged",
            shipment_id="SHIP-001",
            damage_description="Water damage",
            loss_amount=Money(amount="50000.00", currency="EUR"),
        )
        assert sr.loss_amount.amount == Decimal("50000.00")


# ── ScenarioBundle + GroundTruthChecklist tests ───────────────────────────────

class TestScenarioBundle:
    def test_parse_smoke_fixture(self):
        fixture_path = Path("seed_data/easy/smoke.json")
        with open(fixture_path) as f:
            raw = json.load(f)
        scenario = ScenarioBundle.model_validate(raw)
        assert scenario.scenario_id == "easy_smoke"
        assert scenario.difficulty == "easy"
        assert len(scenario.ledger_lines) == 10
        assert len(scenario.documents) == 5

    def test_ground_truth_from_smoke(self):
        fixture_path = Path("seed_data/easy/smoke.json")
        with open(fixture_path) as f:
            raw = json.load(f)
        gt = GroundTruthChecklist.model_validate(raw["ground_truth"])
        assert gt.total_expected_matches == 5
        assert gt.total_expected_eliminations == 5
        assert len(gt.required_matches) == 5
        assert gt.required_matches[0] == ["TXN-E-001-D", "TXN-E-001-C"]


class TestFxRate:
    def test_fx_rate_construction(self):
        fx = FxRate(
            source_currency="USD",
            target_currency="GBP",
            rate_date=date(2024, 2, 14),
            rate=Decimal("0.8200"),
        )
        assert fx.rate == Decimal("0.8200")

    def test_fx_rate_from_dict(self):
        data = {
            "source_currency": "USD",
            "target_currency": "GBP",
            "rate_date": "2024-02-14",
            "rate": "0.8200",
        }
        fx = FxRate.model_validate(data)
        assert fx.rate_date == date(2024, 2, 14)


# ── FinanceDisputeState tests — no grader truth leaks ───────────────────────

class TestFinanceDisputeState:
    def test_no_grader_fields(self):
        state = FinanceDisputeState(
            episode_id="ep-001",
            step_count=5,
            task_id="easy_batch_matching",
            scenario_id="easy_smoke",
            difficulty="easy",
            step_limit=30,
        )
        state_dict = state.model_dump()
        # These fields must NOT be present in state
        assert "required_matches" not in state_dict
        assert "ground_truth" not in state_dict
        assert "required_fx_rate" not in state_dict
        assert "required_liable_entity_id" not in state_dict

    def test_state_fields_present(self):
        state = FinanceDisputeState(
            task_id="hard_liability_dispute",
            scenario_id="hard_smoke",
            difficulty="hard",
            step_limit=30,
            violations_count=2,
        )
        assert state.violations_count == 2
        assert state.task_id == "hard_liability_dispute"
