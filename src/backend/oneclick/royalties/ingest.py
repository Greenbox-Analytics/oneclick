import re


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def upsert_payee(db, user_id: str, display_name: str) -> str:
    """Return the id of an existing or newly-inserted royalty_payees row."""
    norm = normalize_name(display_name)
    res = db.table("royalty_payees").select("id").eq("user_id", user_id).eq("normalized_name", norm).execute()
    rows = res.data or []
    if rows:
        return rows[0]["id"]
    insert_res = (
        db.table("royalty_payees")
        .insert({"user_id": user_id, "normalized_name": norm, "display_name": display_name})
        .execute()
    )
    return (insert_res.data or [{}])[0]["id"]


def persist_statement_rows(db, user_id: str, calculation_id: str, rows, currency: str) -> None:
    """Delete existing rows for this calculation then re-insert from *rows*."""
    db.table("royalty_statement_rows").delete().eq("calculation_id", calculation_id).execute()
    if not rows:
        return
    db.table("royalty_statement_rows").insert(
        [
            {
                "calculation_id": calculation_id,
                "user_id": user_id,
                "song_title": r.song_title,
                "vendor": getattr(r, "vendor", None),
                "country": getattr(r, "country", None),
                "country_code": getattr(r, "country_code", None),
                "delivery_type": getattr(r, "delivery_type", None),
                "delivery_format": getattr(r, "delivery_format", None),
                "sale_date": getattr(r, "sale_date", None),
                "units_sold": getattr(r, "units_sold", None),
                "net_income": getattr(r, "net_income", None),
                "net_payable": getattr(r, "net_payable", None),
                "isrc": getattr(r, "isrc", None),
                "upc": getattr(r, "upc", None),
                "currency": currency,
            }
            for r in rows
        ]
    ).execute()


def compute_statement_meta(rows):
    dates = sorted(r.sale_date for r in rows if getattr(r, "sale_date", None))
    return {
        "period_start": dates[0] if dates else None,
        "period_end": dates[-1] if dates else None,
        "statement_total": sum((r.net_payable or 0) for r in rows),
    }


def statement_meta(db, calculation_id):
    res = (
        db.table("royalty_statement_rows")
        .select("sale_date, net_payable")
        .eq("calculation_id", calculation_id)
        .execute()
    )
    rows = res.data or []
    dates = sorted(r["sale_date"] for r in rows if r.get("sale_date"))
    return {
        "period_start": dates[0] if dates else None,
        "period_end": dates[-1] if dates else None,
        "statement_total": sum((r.get("net_payable") or 0) for r in rows),
    }


def _file_meta(db, file_id: str) -> dict:
    """{'name','content_hash'} for a project_files row, or {} if it's gone."""
    res = db.table("project_files").select("file_name, content_hash").eq("id", file_id).execute()
    rows = res.data or []
    if not rows:
        return {}
    return {"name": rows[0].get("file_name"), "content_hash": rows[0].get("content_hash")}


def sync_calc_royalties(
    db,
    user_id: str,
    calculation_id: str | None,
    royalty_statement_id: str,
    project_id: str,
    results: dict,
    statement_currency: str,
    contract_ids: list[str],
    statement_rows=None,
    persist_rows: bool = True,
    conflict_resolutions: list[dict] | None = None,
    revision_decision: dict | None = None,
    check_only: bool = False,
) -> str:
    """Persist statement rows (best-effort) then run the GATED ledger sync.

    SyncGateError propagates — callers turn it into a 409 (confirm) or a
    needs_confirmation SSE event (stream). Returns the effective statement id.
    """
    from datetime import UTC, datetime

    from oneclick.royalties.ledger_sync import gated_sync

    period_start = period_end = None
    try:
        if statement_rows is not None:
            # persist_rows=False: rows supplied purely to derive the REAL period
            # (the confirm flow reuses its gate-time parse for BOTH the check and
            # the full sync — one parse, and the two periods can't diverge —
            # while the endpoint's own rows-persist step stays the authoritative
            # writer of the rows table).
            if calculation_id is not None and persist_rows:
                persist_statement_rows(db, user_id, calculation_id, statement_rows, statement_currency)
            meta = compute_statement_meta(statement_rows)
        elif calculation_id is not None:
            meta = statement_meta(db, calculation_id)
        else:
            meta = {}
        period_start = meta.get("period_start")
        period_end = meta.get("period_end")
    except Exception as e:  # noqa: BLE001 - best-effort; MUST NOT block line sync
        print(f"[royalties] statement-rows step failed for calc {calculation_id}: {e}")

    ps = period_start or datetime.now(UTC).date().isoformat()
    pe = period_end or ps

    return gated_sync(
        db,
        user_id,
        calculation_id,
        royalty_statement_id,
        project_id,
        results,
        statement_currency,
        ps,
        pe,
        contract_ids,
        statement_file=_file_meta(db, royalty_statement_id),
        contract_files={cid: _file_meta(db, cid) for cid in contract_ids},
        conflict_resolutions=conflict_resolutions,
        revision_decision=revision_decision,
        check_only=check_only,
    )
