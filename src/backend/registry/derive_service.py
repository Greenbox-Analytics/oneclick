"""Collaborator-scoped contract derivation. Parses each linked (or chosen) contract ONCE
(cached by content_hash, collaborator-independent), then filters the parsed parties to the
named collaborator and returns their split + a confidence signal. Terms are NOT extracted
by the underlying parser — they're entered manually in the frontend review screen."""

import os
import tempfile

from supabase import Client

from registry import contract_splits


def _parse_pdf_bytes(content: bytes) -> dict:
    """Run the existing parser on PDF bytes. main_artist_name="" so the cached result is
    collaborator-independent (name matching happens in derive_for_collaborator)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        return contract_splits.parse_royalty_splits(pdf_path=tmp_path, main_artist_name="")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _parse_file_cached(db: Client, file_row: dict) -> dict | None:
    """Return the parsed result for a project_files row, using contract_parse_cache by
    content_hash. Parses + caches on a miss. Returns None if the file can't be downloaded."""
    chash = file_row.get("content_hash")
    if chash:
        cached = db.table("contract_parse_cache").select("parsed").eq("content_hash", chash).maybe_single().execute()
        if cached and cached.data:
            return cached.data["parsed"]
    file_path = file_row.get("file_path")
    if not file_path:
        return None
    try:
        content = db.storage.from_("project-files").download(file_path)
    except Exception:
        return None
    parsed = _parse_pdf_bytes(content)
    if chash:
        db.table("contract_parse_cache").insert({"content_hash": chash, "parsed": parsed}).execute()
    return parsed


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
            .select("file_path, content_hash, file_name")
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

    # Match the collaborator by name or alias (exact, else substring either
    # direction). Cached payloads that predate aliases simply have none.
    target = None
    for p in parties:
        if contract_splits.matches_party_name(p.get("name") or "", p.get("aliases"), name):
            target = p
            break

    if not target:
        return {
            "found": False,
            "confidence": "low",
            "master_pct": None,
            "publishing_pct": None,
            "soundexchange_pct": None,
            "terms": [],
            "matched_file_ids": [],
        }

    master = target.get("master_pct")
    publishing = target.get("publishing_pct")
    # .get() tolerates cached payloads that predate the soundexchange bucket.
    soundexchange = target.get("soundexchange_pct")
    # Scoped confidence: low if no percentage extracted for them (NOT based on cap-table totals).
    has_pct = bool(master) or bool(publishing)
    return {
        "found": True,
        "confidence": "high" if has_pct else "low",
        "master_pct": master,
        "publishing_pct": publishing,
        "soundexchange_pct": soundexchange,
        "terms": [],  # not auto-extracted; manual entry in the review UI
        "matched_file_ids": [target["_file_id"]],
    }
