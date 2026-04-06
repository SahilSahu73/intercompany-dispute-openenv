"""Document service: retrieve document text and metadata."""

from domain.scenario_models import EpisodeContext


def fetch_document(ctx: EpisodeContext, document_id: str) -> dict:
    """Retrieve full document text by ID.

    Side effect: adds document_id to ctx.evidence_cache so the
    environment knows the agent has gathered this evidence.

    Returns on success:
        {
            "document_id": str,
            "document_type": str,
            "title": str,
            "body": str,
            "related_entity_ids": list[str],
        }
    Returns on failure:
        {"error": "Document not found: {document_id}"}
    """
    doc = ctx.documents.get(document_id)
    if not doc:
        return {"error": f"Document not found: {document_id}"}

    # Track that agent has fetched this document (used for evidence checking by grader)
    ctx.evidence_cache.add(document_id)

    return {
        "document_id": doc.document_id,
        "document_type": doc.document_type,
        "title": doc.title,
        "body": doc.body,
        "related_entity_ids": doc.related_entity_ids,
    }
