"""Tests for the service layer — Phase 3 validation gate."""

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from domain.document_models import Contract, Document, Invoice
from domain.ledger_models import AuditEvent, LedgerLine
from domain.money import Money
from domain.scenario_models import (
    EpisodeContext,
    FxRate,
    GroundTruthChecklist,
    LegalTruth,
    ScenarioBundle,
)
from services import audit_service, document_service, ledger_service, legal_service, matching_service, treasury_service


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_smoke_ctx() -> EpisodeContext:
    """Load the easy smoke scenario and return a fresh EpisodeContext."""
    with open(Path("seed_data/easy/smoke.json")) as f:
        raw = json.load(f)
    scenario = ScenarioBundle.model_validate(raw)
    gt = GroundTruthChecklist.model_validate(raw["ground_truth"])

    ledger_lines = {}
    for line_data in scenario.ledger_lines:
        line = LedgerLine.model_validate(line_data)
        ledger_lines[line.txn_id] = line

    documents = {}
    for doc_data in scenario.documents:
        doc = Invoice.model_validate(doc_data)
        documents[doc.document_id] = doc

    return EpisodeContext(
        scenario=scenario,
        ground_truth=gt,
        ledger_lines=ledger_lines,
        documents=documents,
        fx_rates=[],
    )


def make_fx_ctx() -> EpisodeContext:
    """Return a minimal EpisodeContext with FX rates for treasury tests."""
    scenario = ScenarioBundle(
        scenario_id="test_fx",
        task_id="medium_fx_variance",
        difficulty="medium",
        description="FX test",
        ledger_lines=[
            {
                "txn_id": "TXN-FX-D", "entity_id": "US_PARENT",
                "counterparty_entity_id": "UK_SUB", "account_code": "1300",
                "side": "debit", "money": {"amount": "10000.00", "currency": "USD"},
                "booking_date": "2024-01-15", "description": "Test",
            },
            {
                "txn_id": "TXN-FX-C", "entity_id": "UK_SUB",
                "counterparty_entity_id": "US_PARENT", "account_code": "2300",
                "side": "credit", "money": {"amount": "10000.00", "currency": "USD"},
                "booking_date": "2024-01-15", "description": "Test",
            },
        ],
        documents=[],
        fx_rates=[],
        objectives=["test"],
        step_limit=20,
        ground_truth={
            "required_matches": [], "required_adjustments": [],
            "required_eliminations": [], "total_expected_matches": 0,
            "total_expected_adjustments": 0, "total_expected_eliminations": 0,
        },
    )
    gt = GroundTruthChecklist.model_validate(scenario.ground_truth)

    ledger_lines = {}
    for line_data in scenario.ledger_lines:
        line = LedgerLine.model_validate(line_data)
        ledger_lines[line.txn_id] = line

    ctx = EpisodeContext(
        scenario=scenario,
        ground_truth=gt,
        ledger_lines=ledger_lines,
        fx_rates=[
            FxRate(source_currency="USD", target_currency="GBP",
                   rate_date=date(2024, 1, 15), rate=Decimal("0.8000")),
            FxRate(source_currency="USD", target_currency="GBP",
                   rate_date=date(2024, 2, 14), rate=Decimal("0.8200")),
        ],
    )
    return ctx


# ── LedgerService tests ───────────────────────────────────────────────────────

class TestLedgerService:
    def test_query_open_items_no_filter(self):
        ctx = load_smoke_ctx()
        result = ledger_service.query_open_items(ctx)
        assert result["total_count"] == 10  # 5 pairs = 10 lines, all open
        assert result["returned_count"] == 10

    def test_query_open_items_entity_filter(self):
        ctx = load_smoke_ctx()
        result = ledger_service.query_open_items(ctx, entity_id="US_PARENT")
        assert result["total_count"] == 5  # 5 debit lines

    def test_query_open_items_status_filter(self):
        ctx = load_smoke_ctx()
        # Match one pair first
        matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        result = ledger_service.query_open_items(ctx, status="matched")
        assert result["total_count"] == 2

    def test_query_open_items_limit(self):
        ctx = load_smoke_ctx()
        result = ledger_service.query_open_items(ctx, limit=3)
        assert result["returned_count"] == 3
        assert result["total_count"] == 10  # total unchanged

    def test_query_ledger_balance(self):
        ctx = load_smoke_ctx()
        result = ledger_service.query_ledger_balance(ctx, "US_PARENT", "1300")
        # 5 debits: 10000 + 25000 + 5000 + 50000 + 15000 = 105000
        assert Decimal(result["debit_total"]) == Decimal("105000")
        assert Decimal(result["credit_total"]) == Decimal("0")
        assert Decimal(result["net_balance"]) == Decimal("105000")

    def test_query_ledger_balance_excludes_eliminated(self):
        ctx = load_smoke_ctx()
        match_res = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        matching_service.execute_elimination(ctx, "US_PARENT", match_res["match_id"])
        result = ledger_service.query_ledger_balance(ctx, "US_PARENT", "1300")
        assert Decimal(result["debit_total"]) == Decimal("95000")  # 105000 - 10000 eliminated


# ── DocumentService tests ─────────────────────────────────────────────────────

class TestDocumentService:
    def test_fetch_document_success(self):
        ctx = load_smoke_ctx()
        result = document_service.fetch_document(ctx, "DOC-E-001")
        assert "error" not in result
        assert result["document_id"] == "DOC-E-001"
        assert "INV-001" in result["body"]

    def test_fetch_document_adds_to_evidence_cache(self):
        ctx = load_smoke_ctx()
        assert "DOC-E-001" not in ctx.evidence_cache
        document_service.fetch_document(ctx, "DOC-E-001")
        assert "DOC-E-001" in ctx.evidence_cache

    def test_fetch_document_not_found(self):
        ctx = load_smoke_ctx()
        result = document_service.fetch_document(ctx, "DOC-NONEXISTENT")
        assert "error" in result
        assert "not found" in result["error"]


# ── TreasuryService tests ─────────────────────────────────────────────────────

class TestTreasuryService:
    def test_calculate_fx_exact_date(self):
        ctx = make_fx_ctx()
        result = treasury_service.calculate_fx(ctx, "USD", "GBP", "10000.00", "2024-01-15")
        assert "error" not in result
        assert result["rate"] == "0.8000"
        assert result["converted_amount"] == "8000.00"

    def test_calculate_fx_nearest_prior_date(self):
        ctx = make_fx_ctx()
        # No rate for Feb 1 — should use Jan 15 rate
        result = treasury_service.calculate_fx(ctx, "USD", "GBP", "10000.00", "2024-02-01")
        assert result["rate"] == "0.8000"  # Jan 15 rate used

    def test_calculate_fx_later_date_uses_nearest(self):
        ctx = make_fx_ctx()
        result = treasury_service.calculate_fx(ctx, "USD", "GBP", "10000.00", "2024-02-14")
        assert result["rate"] == "0.8200"  # Feb 14 rate
        assert result["converted_amount"] == "8200.00"

    def test_calculate_fx_same_currency(self):
        ctx = make_fx_ctx()
        result = treasury_service.calculate_fx(ctx, "USD", "USD", "5000.00", "2024-01-15")
        assert result["rate"] == "1.0"
        assert result["converted_amount"] == "5000.00"

    def test_calculate_fx_no_rate_found(self):
        ctx = make_fx_ctx()
        result = treasury_service.calculate_fx(ctx, "USD", "EUR", "1000.00", "2024-01-15")
        assert "error" in result

    def test_calculate_fx_sets_queried_flag(self):
        ctx = make_fx_ctx()
        assert not ctx.fx_queried
        treasury_service.calculate_fx(ctx, "USD", "GBP", "100.00", "2024-01-15")
        assert ctx.fx_queried

    def test_calculate_fx_invalid_date(self):
        ctx = make_fx_ctx()
        result = treasury_service.calculate_fx(ctx, "USD", "GBP", "100.00", "not-a-date")
        assert "error" in result


# ── LegalService tests ────────────────────────────────────────────────────────

def make_legal_ctx() -> EpisodeContext:
    """Context with a contract document and legal truth."""
    scenario = ScenarioBundle(
        scenario_id="test_legal",
        task_id="hard_liability_dispute",
        difficulty="hard",
        description="Legal test",
        ledger_lines=[
            {
                "txn_id": "TXN-H-D", "entity_id": "DE_SUB",
                "counterparty_entity_id": "UK_SUB", "account_code": "1300",
                "side": "debit", "money": {"amount": "50000.00", "currency": "EUR"},
                "booking_date": "2024-03-01", "description": "IC Receivable",
            },
        ],
        documents=[
            {
                "document_id": "CON-001", "document_type": "contract",
                "title": "Transit Contract", "body": "CIF terms apply.",
                "related_entity_ids": ["DE_SUB", "UK_SUB"],
            },
            {
                "document_id": "INV-001", "document_type": "invoice",
                "title": "Invoice", "body": "Standard invoice.",
            },
        ],
        fx_rates=[],
        legal_truth={
            "contract_document_id": "CON-001",
            "incoterm": "CIF",
            "liable_entity_id": "DE_SUB",
            "liable_event": "goods_damaged_in_transit",
            "rationale": "Under CIF, seller bears risk until destination port.",
        },
        objectives=["test"],
        step_limit=20,
        ground_truth={
            "required_matches": [], "required_adjustments": [],
            "required_eliminations": [], "total_expected_matches": 0,
            "total_expected_adjustments": 0, "total_expected_eliminations": 0,
        },
    )
    gt = GroundTruthChecklist.model_validate(scenario.ground_truth)
    legal_truth = LegalTruth.model_validate(scenario.legal_truth)

    ledger_lines = {}
    for line_data in scenario.ledger_lines:
        line = LedgerLine.model_validate(line_data)
        ledger_lines[line.txn_id] = line

    documents = {}
    for doc_data in scenario.documents:
        doc_type = doc_data["document_type"]
        if doc_type == "contract":
            doc = Contract.model_validate(doc_data)
        else:
            doc = Invoice.model_validate(doc_data)
        documents[doc.document_id] = doc

    return EpisodeContext(
        scenario=scenario,
        ground_truth=gt,
        ledger_lines=ledger_lines,
        documents=documents,
        legal_truth=legal_truth,
    )


class TestLegalService:
    def test_correct_contract_returns_liability(self):
        ctx = make_legal_ctx()
        result = legal_service.ask_legal_analyst(ctx, "CON-001", "Who is liable?")
        assert "error" not in result
        assert result["incoterm"] == "CIF"
        assert result["liable_entity_id"] == "DE_SUB"

    def test_correct_contract_sets_consulted_flag(self):
        ctx = make_legal_ctx()
        assert not ctx.legal_consulted
        legal_service.ask_legal_analyst(ctx, "CON-001", "Who is liable?")
        assert ctx.legal_consulted
        assert "CON-001" in ctx.evidence_cache

    def test_wrong_contract_gives_informational_response(self):
        ctx = make_legal_ctx()
        result = legal_service.ask_legal_analyst(ctx, "INV-001", "Who is liable?")
        # INV-001 is an invoice, not a contract
        assert "error" in result

    def test_nonexistent_document(self):
        ctx = make_legal_ctx()
        result = legal_service.ask_legal_analyst(ctx, "DOES-NOT-EXIST", "Q")
        assert "error" in result

    def test_no_legal_context_returns_error(self):
        ctx = load_smoke_ctx()  # Easy task — no legal truth
        result = legal_service.ask_legal_analyst(ctx, "DOC-E-001", "Who is liable?")
        assert "error" in result


# ── MatchingService tests ─────────────────────────────────────────────────────

class TestMatchingService:
    def test_execute_match_success(self):
        ctx = load_smoke_ctx()
        result = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        assert result["status"] == "ok"
        assert result["match_id"].startswith("MATCH-")
        # Statuses updated
        assert ctx.ledger_lines["TXN-E-001-D"].status == "matched"
        assert ctx.ledger_lines["TXN-E-001-C"].status == "matched"

    def test_execute_match_wrong_side_order(self):
        ctx = load_smoke_ctx()
        # Reversed: credit first, debit second
        result = matching_service.execute_match(ctx, "TXN-E-001-C", "TXN-E-001-D")
        assert result["status"] == "rejected"

    def test_execute_match_nonexistent_txn(self):
        ctx = load_smoke_ctx()
        result = matching_service.execute_match(ctx, "FAKE-TXN", "TXN-E-001-C")
        assert result["status"] == "rejected"
        assert "not found" in result["error"]

    def test_execute_match_already_matched(self):
        ctx = load_smoke_ctx()
        matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        result = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        assert result["status"] == "rejected"

    def test_execute_elimination_success(self):
        ctx = load_smoke_ctx()
        match_res = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        elim_res = matching_service.execute_elimination(ctx, "US_PARENT", match_res["match_id"])
        assert elim_res["status"] == "ok"
        assert elim_res["elimination_id"].startswith("ELIM-")
        assert ctx.ledger_lines["TXN-E-001-D"].status == "eliminated"

    def test_execute_elimination_wrong_entity(self):
        ctx = load_smoke_ctx()
        match_res = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        result = matching_service.execute_elimination(ctx, "DE_SUB", match_res["match_id"])
        assert result["status"] == "rejected"

    def test_execute_elimination_double_elimination(self):
        ctx = load_smoke_ctx()
        match_res = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
        matching_service.execute_elimination(ctx, "US_PARENT", match_res["match_id"])
        result = matching_service.execute_elimination(ctx, "US_PARENT", match_res["match_id"])
        assert result["status"] == "rejected"

    def test_post_adjustment_success(self):
        ctx = load_smoke_ctx()
        result = matching_service.post_adjustment(
            ctx, "US_PARENT", "6100", "1300", "243.90", "USD", "fx_variance",
            evidence_refs=["DOC-E-001"],
        )
        assert result["status"] == "ok"
        assert result["entry_id"].startswith("ADJ-")
        assert len(ctx.adjustments) == 1
        # New ledger lines created
        assert f"{result['entry_id']}-D" in ctx.ledger_lines

    def test_post_adjustment_invalid_account(self):
        ctx = load_smoke_ctx()
        result = matching_service.post_adjustment(
            ctx, "US_PARENT", "9999", "1300", "100.00", "USD", "fx_variance",
        )
        assert result["status"] == "rejected"

    def test_post_adjustment_negative_amount(self):
        ctx = load_smoke_ctx()
        result = matching_service.post_adjustment(
            ctx, "US_PARENT", "6100", "1300", "-100.00", "USD", "fx_variance",
        )
        assert result["status"] == "rejected"

    def test_post_adjustment_unknown_entity(self):
        ctx = load_smoke_ctx()
        result = matching_service.post_adjustment(
            ctx, "FAKE_ENTITY", "6100", "1300", "100.00", "USD", "fx_variance",
        )
        assert result["status"] == "rejected"


# ── AuditService tests ────────────────────────────────────────────────────────

class TestAuditService:
    def test_record_event(self):
        ctx = load_smoke_ctx()
        assert len(ctx.audit_log) == 0
        audit_service.record_event(ctx, "orchestrator", "query_open_items", "ok", "test")
        assert len(ctx.audit_log) == 1
        assert ctx.audit_log[0].action_type == "query_open_items"

    def test_detect_loops_no_loop(self):
        ctx = load_smoke_ctx()
        for action in ["execute_match", "fetch_document", "calculate_fx", "execute_match", "query_open_items"]:
            audit_service.record_event(ctx, "orchestrator", action, "ok", f"detail-{action}")
        assert not audit_service.detect_loops(ctx)

    def test_detect_loops_identical_actions(self):
        ctx = load_smoke_ctx()
        for _ in range(6):
            audit_service.record_event(ctx, "orchestrator", "query_open_items", "ok", "entity=US_PARENT")
        assert audit_service.detect_loops(ctx)

    def test_count_action_type(self):
        ctx = load_smoke_ctx()
        audit_service.record_event(ctx, "orchestrator", "execute_match", "ok")
        audit_service.record_event(ctx, "orchestrator", "execute_match", "ok")
        audit_service.record_event(ctx, "orchestrator", "fetch_document", "ok")
        assert audit_service.count_action_type(ctx, "execute_match") == 2
        assert audit_service.count_action_type(ctx, "fetch_document") == 1


# ── Determinism test ──────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_scenario_same_result(self):
        """Running the same actions on the same scenario produces identical results."""
        def run_scenario():
            ctx = load_smoke_ctx()
            r1 = ledger_service.query_open_items(ctx, entity_id="US_PARENT")
            r2 = matching_service.execute_match(ctx, "TXN-E-001-D", "TXN-E-001-C")
            r3 = ledger_service.query_ledger_balance(ctx, "US_PARENT", "1300")
            return r1["total_count"], r2["status"], r3["net_balance"]

        result_a = run_scenario()
        result_b = run_scenario()
        assert result_a == result_b
