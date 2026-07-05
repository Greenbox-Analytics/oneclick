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


def sync_royalty_lines(
    db,
    user_id: str,
    calculation_id: str,
    royalty_statement_id: str,
    project_id: str,
    results: dict,
    statement_currency: str,
    period_start,
    period_end,
) -> None:
    """Refresh royalty_lines for a (statement, project) bucket from *results*.

    Delete-then-insert is idempotent: calling twice with the same results
    yields the same set of rows.  Never touches royalty_calculations.
    """
    (
        db.table("royalty_lines")
        .delete()
        .eq("royalty_statement_id", royalty_statement_id)
        .eq("project_id", project_id)
        .execute()
    )

    # Build a case-insensitive, trimmed work-title → id map scoped to the project.
    works_res = db.table("works_registry").select("id, title").eq("project_id", project_id).execute()
    work_map = {w["title"].strip().lower(): w["id"] for w in (works_res.data or []) if w.get("title")}

    payments = results.get("payments", [])
    if not payments:
        return

    db.table("royalty_lines").insert(
        [
            {
                "user_id": user_id,
                "calculation_id": calculation_id,
                "royalty_statement_id": royalty_statement_id,
                "payee_id": upsert_payee(db, user_id, payment["party_name"]),
                "project_id": project_id,
                "work_id": work_map.get(payment["song_title"].strip().lower()),
                "song_title": payment["song_title"],
                "role": payment.get("role"),
                "royalty_type": payment.get("royalty_type"),
                "percentage": payment.get("percentage"),
                "song_revenue": payment.get("total_royalty"),
                "amount_owed": payment.get("amount_to_pay"),
                "statement_currency": statement_currency,
                "period_start": period_start,
                "period_end": period_end,
            }
            for payment in payments
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


def sync_calc_royalties(
    db,
    user_id: str,
    calculation_id: str,
    royalty_statement_id: str,
    project_id: str,
    results: dict,
    statement_currency: str,
    statement_rows=None,
) -> None:
    """Persist statement rows + sync royalty lines after a calc is saved.

    The statement-rows step (persistence + period derivation) is **best-effort and
    isolated**: if it fails — e.g. the optional ``royalty_statement_rows`` table is
    absent, or a statement parses oddly — we log and continue. ``sync_royalty_lines``
    (which creates the payees/lines the UI shows) is on the **critical path and ALWAYS
    runs**, with a fallback period, so payee/owed data is never blocked by a
    statement-rows problem.

    ``statement_rows``: pre-parsed ``list[StatementRow]`` (the /confirm path) → persisted
    and the period derived in-memory. ``None`` (the stream cache-hit path) → the period is
    derived from already-persisted rows.
    """
    from datetime import UTC, datetime

    period_start = period_end = None
    try:
        if statement_rows is not None:
            persist_statement_rows(db, user_id, calculation_id, statement_rows, statement_currency)
            meta = compute_statement_meta(statement_rows)
        else:
            meta = statement_meta(db, calculation_id)
        period_start = meta.get("period_start")
        period_end = meta.get("period_end")
    except Exception as e:  # noqa: BLE001 - best-effort; MUST NOT block line sync
        print(f"[royalties] statement-rows step failed for calc {calculation_id}: {e}")

    ps = period_start or datetime.now(UTC).date().isoformat()
    pe = period_end or ps
    sync_royalty_lines(
        db, user_id, calculation_id, royalty_statement_id, project_id, results, statement_currency, ps, pe
    )
