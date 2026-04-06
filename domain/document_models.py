"""Document domain models: invoices, emails, contracts, shipment reports."""

from datetime import date

from pydantic import BaseModel, Field

from .enums import DocumentType
from .money import Money


class Document(BaseModel):
    """Base document model stored in the environment."""

    document_id: str
    document_type: DocumentType
    title: str
    body: str  # Full text content
    related_entity_ids: list[str] = Field(default_factory=list)
    related_txn_ids: list[str] = Field(default_factory=list)
    issue_date: date | None = None


class DocumentSummary(BaseModel):
    """Public summary of a document exposed to the agent.

    The full body is returned by the fetch_document tool.
    This summary appears in observation metadata.
    """

    document_id: str
    document_type: DocumentType
    title: str
    snippet: str  # First ~200 chars of body


class Invoice(Document):
    """Invoice document with structured financial data."""

    document_type: DocumentType = "invoice"
    vendor_name: str = ""
    buyer_name: str = ""
    amount: Money | None = None
    due_date: date | None = None


class Contract(Document):
    """Contract document with shipping/legal terms."""

    document_type: DocumentType = "contract"
    incoterm: str | None = None  # e.g. "CIF", "FOB"
    origin_entity_id: str | None = None
    destination_entity_id: str | None = None


class ShipmentReport(Document):
    """Shipment report with damage/loss details."""

    document_type: DocumentType = "shipment_report"
    shipment_id: str = ""
    damage_description: str = ""
    loss_amount: Money | None = None
