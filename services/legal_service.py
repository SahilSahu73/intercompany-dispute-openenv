"""Legal service: deterministic liability determination from seeded contract truth."""

from domain.scenario_models import EpisodeContext


def ask_legal_analyst(ctx: EpisodeContext, document_id: str, question: str) -> dict:
    """Interpret a contract document and return a liability determination.

    The legal truth is pre-seeded per scenario. This service does NOT
    perform LLM inference — it returns the deterministic answer from the
    scenario's ground truth legal_truth field.

    Side effect: sets ctx.legal_consulted = True and adds document_id
    to ctx.evidence_cache when the correct contract is queried.

    Returns on success:
        {
            "document_id": str,
            "question": str,
            "incoterm": str,
            "liable_entity_id": str,
            "liable_event": str,
            "rationale": str,
        }
    Returns on failure:
        {"error": str}
    OR informational (wrong document):
        {"document_id": str, "question": str, "answer": str}
    """
    if not ctx.legal_truth:
        return {
            "error": (
                "No legal context available for this scenario. "
                "Legal analyst is not needed for this task difficulty."
            )
        }

    doc = ctx.documents.get(document_id)
    if not doc:
        return {"error": f"Document not found: {document_id}"}

    if doc.document_type != "contract":
        return {
            "error": (
                f"Document {document_id} is a {doc.document_type}, not a contract. "
                "Legal analysis requires a contract document."
            )
        }

    # Check if it's the relevant contract
    if document_id != ctx.legal_truth.contract_document_id:
        return {
            "document_id": document_id,
            "question": question,
            "answer": (
                "This contract does not contain relevant liability terms for the current dispute. "
                "Try fetching the transit/shipping contract."
            ),
        }

    # Correct contract — mark evidence and consultation
    ctx.legal_consulted = True
    ctx.evidence_cache.add(document_id)

    return {
        "document_id": document_id,
        "question": question,
        "incoterm": ctx.legal_truth.incoterm,
        "liable_entity_id": ctx.legal_truth.liable_entity_id,
        "liable_event": ctx.legal_truth.liable_event,
        "rationale": ctx.legal_truth.rationale,
    }
