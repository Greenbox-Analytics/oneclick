"""Collaborator-scoped contract derivation. Parses each linked (or chosen) contract ONCE
(cached by content_hash, collaborator-independent), then filters the parsed parties to the
named collaborator and returns their split + a confidence signal. Terms are NOT extracted
by the underlying parser — they're entered manually in the frontend review screen."""

import os
import tempfile

from supabase import Client

from registry import contract_splits
from utils.contract_parsing.cache import get_or_parse


def _norm(s) -> str:
    return (s or "").strip().lower()


def _pdf_bytes_to_markdown(content: bytes) -> str:
    """Convert PDF bytes to markdown via a temp file (pymupdf4llm, local, no LLM).
    Page markers are stripped downstream by get_or_parse (canonical parse input)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        from utils.ingestion.pdf_markdown import pdf_to_markdown

        return pdf_to_markdown(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _parse_file_cached(db: Client, file_row: dict) -> dict | None:
    """Return the pivoted splits for a project_files row, using the shared parse cache.
    Prefers the stored `contract_markdown` as the canonical text (the same source OneClick
    uses) so both tools share cache entries and no PDF is downloaded when the markdown is
    already present; falls back to downloading + converting the PDF only when it isn't.
    Returns None if the file can't be parsed."""
    file_path = file_row.get("file_path")
    stored_md = file_row.get("contract_markdown")
    if not stored_md and not file_path:
        return None

    def _load_text() -> str:
        if stored_md:
            return stored_md
        content = db.storage.from_("project-files").download(file_path)
        return _pdf_bytes_to_markdown(content)

    try:
        contract_data = get_or_parse(db, _load_text)
    except Exception:
        # Download/convert/parse failed on a miss — behave like the old "can't download".
        return None

    return contract_splits.parse_royalty_splits(contract_data=contract_data, main_artist_name="")


async def derive_for_collaborator(db: Client, work_id: str, name: str, email=None, contract_file_ids=None) -> dict:
    """Resolve contract files (all linked, or the given subset), parse (cached), and filter
    to the collaborator named `name`. Returns split + confidence + matched_file_ids."""
    if contract_file_ids:
        file_ids = list(contract_file_ids)
    else:
        wf = (db.table("work_files").select("file_id").eq("work_id", work_id).execute()).data or []
        file_ids = [r["file_id"] for r in wf]

    parties = []  # each: parsed party dict + "_file_id"
    for fid in file_ids:
        row = (
            db.table("project_files")
            .select("file_path, content_hash, file_name, contract_markdown")
            .eq("id", fid)
            .maybe_single()
            .execute()
        )
        if not row or not row.data:
            continue
        parsed = _parse_file_cached(db, row.data)
        if not parsed:
            continue
        for p in parsed.get("parties", []):
            parties.append({**p, "_file_id": fid})

    # Match the collaborator by name (exact, else substring either direction).
    target = None
    for p in parties:
        pn = _norm(p.get("name"))
        if pn and (pn == _norm(name) or _norm(name) in pn or pn in _norm(name)):
            target = p
            break

    if not target:
        return {
            "found": False,
            "confidence": "low",
            "master_pct": None,
            "publishing_pct": None,
            "terms": [],
            "matched_file_ids": [],
        }

    master = target.get("master_pct")
    publishing = target.get("publishing_pct")
    # Scoped confidence: low if no percentage extracted for them (NOT based on cap-table totals).
    has_pct = bool(master) or bool(publishing)
    return {
        "found": True,
        "confidence": "high" if has_pct else "low",
        "master_pct": master,
        "publishing_pct": publishing,
        "terms": [],  # not auto-extracted; manual entry in the review UI
        "matched_file_ids": [target["_file_id"]],
    }
