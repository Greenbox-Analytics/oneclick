"""Gated ledger sync for OneClick royalties (spec 2026-07-23, rev 3).

This module is the ONLY writer of royalty_lines. Pure helpers here; the
DB engine (gates + upsert) lives below them (added by a later task).
"""

from oneclick.royalties import history
from oneclick.royalties.ingest import normalize_name, upsert_payee

EPS = 0.01

# Same normalization everywhere a name/title becomes a key — this is copy #1,
# not #4: reuse ingest.normalize_name rather than re-implementing the regex.
song_key = normalize_name
_party_key = normalize_name


def type_key(royalty_type: str | None) -> str:
    """Normalized entitlement class. Every ledgered share today is
    streaming-earnable, so streaming-equivalents (and legacy None) map to
    'streaming'; anything else normalizes to its lowercase form (future classes)."""
    from oneclick.royalty_calculator import is_streaming_equivalent_royalty_type

    if royalty_type is None or is_streaming_equivalent_royalty_type(royalty_type):
        return "streaming"
    return normalize_name(royalty_type)


def aggregate_payments(payments: list[dict], run_contract_ids: list[str], resolutions: list[dict] | None = None):
    """Pre-aggregate raw calculator payments into one entry per obligation
    identity (party, song_key, royalty_type_key).

    Within one source-set: amounts sum (multiple shares from one contract).
    Across source-sets: equal totals corroborate (union sources, not additive);
    unequal totals are a CONFLICT unless a resolution picks the governing
    contract. Returns (aggregated: {key: entry}, conflicts: [payload]).
    """
    res_map = {
        (r["party_key"], r["song_key"], r["royalty_type_key"]): r["governing_contract_id"] for r in (resolutions or [])
    }

    # key -> source-frozenset -> accumulated group
    grouped: dict[tuple, dict[frozenset, dict]] = {}
    for p in payments:
        key = (_party_key(p["party_name"]), song_key(p["song_title"]), type_key(p.get("royalty_type")))
        sources = frozenset(p.get("source_contract_ids") or run_contract_ids)
        g = grouped.setdefault(key, {}).setdefault(
            sources,
            {
                "amount": 0.0,
                "percentage": 0.0,
                "pct_known": True,
                "party_name": p["party_name"],
                "song_title": p["song_title"],
                "role": p.get("role"),
                "royalty_type": p.get("royalty_type"),
            },
        )
        g["amount"] += float(p.get("amount_to_pay") or 0)
        if p.get("percentage") is None:
            g["pct_known"] = False
        else:
            g["percentage"] += float(p["percentage"])

    aggregated: dict[tuple, dict] = {}
    conflicts: list[dict] = []
    for key, by_sources in grouped.items():
        groups = list(by_sources.items())
        governing = res_map.get(key)
        if governing:
            filtered = [(s, g) for s, g in groups if governing in s]
            if filtered:
                groups = filtered
            else:
                # The chosen governing contract isn't in ANY in-run group: the
                # user kept an ABSENT contract's number. Drop the identity here —
                # gate 4 preserves the stored line. Falling back to all groups
                # would re-raise the in-run conflict forever (three-way loop).
                continue

        base_sources, base = groups[0]
        merged_sources = set(base_sources)
        conflicting = []
        for s, g in groups[1:]:
            if abs(g["amount"] - base["amount"]) <= EPS:
                merged_sources |= s  # corroboration
            else:
                conflicting.append((s, g))

        if conflicting:
            conflicts.append(
                {
                    "party_key": key[0],
                    "song_key": key[1],
                    "royalty_type_key": key[2],
                    "party_name": base["party_name"],
                    "song_title": base["song_title"],
                    "claims": [
                        {
                            "contract_ids": sorted(s),
                            "amount": round(g["amount"], 2),
                            "percentage": g["percentage"] if g["pct_known"] else None,
                        }
                        for s, g in groups
                    ],
                }
            )
            continue

        aggregated[key] = {
            **base,
            "percentage": base["percentage"] if base["pct_known"] else None,
            "sources": merged_sources,
        }
    return aggregated, conflicts


# ---------------------------------------------------------------------------
# DB engine: gates + run-scoped upsert. The ONLY writer of royalty_lines.
# ---------------------------------------------------------------------------


class SyncGateError(Exception):
    """A blocking gate fired. gate ∈ {'superseded','conflict','revision'};
    payload is the structured question for the client. No writes happened."""

    def __init__(self, gate: str, payload: dict):
        super().__init__(gate)
        self.gate = gate
        self.payload = payload


def _periods_overlap(a_start, a_end, b_start, b_end) -> bool:
    return bool(a_start and a_end and b_start and b_end and a_start <= b_end and b_start <= a_end)


def gated_sync(
    db,
    user_id: str,
    calculation_id: str | None,
    royalty_statement_id: str,
    project_id: str,
    results: dict,
    statement_currency: str,
    period_start,
    period_end,
    contract_ids: list[str],
    statement_file: dict | None = None,  # {"name","content_hash"} of the run's statement
    contract_files: dict | None = None,  # {cid: {"name","content_hash"}}
    conflict_resolutions: list[dict] | None = None,
    revision_decision: dict | None = None,  # {"replace": old_stmt_id} | {"none": True}
    check_only: bool = False,
) -> str:
    """Run every gate, then upsert. Returns the EFFECTIVE statement id
    (adoption may substitute an existing one). Raises SyncGateError with no
    writes when a gate needs the user."""
    contract_files = contract_files or {}
    run_set = set(contract_ids)

    # ---- Gate 1: tombstone -------------------------------------------------
    tomb = (
        db.table("royalty_statement_supersessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("old_statement_id", royalty_statement_id)
        .eq("kind", "superseded")
        .execute()
    )
    if tomb.data:
        raise SyncGateError(
            "superseded",
            {"old_statement_id": royalty_statement_id, "new_statement_id": tomb.data[0]["new_statement_id"]},
        )

    # ---- Gate 2: statement adoption by content hash --------------------------
    effective_stmt = royalty_statement_id
    stmt_hash = (statement_file or {}).get("content_hash")
    if stmt_hash:  # null never matches null
        match = (
            db.table("royalty_lines")
            .select("royalty_statement_id")
            .eq("user_id", user_id)
            .eq("project_id", project_id)
            .eq("statement_content_hash", stmt_hash)
            .neq("royalty_statement_id", royalty_statement_id)
            .execute()
        )
        if match.data:
            effective_stmt = match.data[0]["royalty_statement_id"]

    # Load the project's lines once — gates 3-5 and the engine all use them.
    all_lines = (
        db.table("royalty_lines").select("*").eq("user_id", user_id).eq("project_id", project_id).execute()
    ).data or []

    # ---- Gate 3: contract adoption by content hash ---------------------------
    # COMPUTED here, APPLIED only after every gate passes: a SyncGateError must
    # leave the ledger byte-identical on both write paths.
    hash_to_cid = {m["content_hash"]: cid for cid, m in contract_files.items() if m.get("content_hash")}
    pending_adoptions: list[tuple[dict, list]] = []  # (line, new_sources)
    if hash_to_cid:
        for line in all_lines:
            sources = [dict(e) for e in (line.get("source_contracts") or [])]
            swapped = False
            for entry in sources:
                new_cid = hash_to_cid.get(entry.get("hash"))
                if new_cid and entry.get("id") != new_cid:
                    entry["id"] = new_cid
                    entry["name"] = contract_files[new_cid].get("name") or entry.get("name")
                    swapped = True
            if swapped:
                pending_adoptions.append((line, sources))

    # ---- Aggregate the run's payments ----------------------------------------
    aggregated, conflicts = aggregate_payments(
        results.get("payments", []) or [], contract_ids, resolutions=conflict_resolutions
    )
    if conflicts:
        raise SyncGateError("conflict", {"scope": "in_run", "conflicts": conflicts})

    payee_names = {
        p["id"]: p["normalized_name"]
        for p in (db.table("royalty_payees").select("id, normalized_name").eq("user_id", user_id).execute().data or [])
    }

    stmt_lines = [l for l in all_lines if l["royalty_statement_id"] == effective_stmt]
    by_identity = {(payee_names.get(l["payee_id"], ""), l["song_key"], l["royalty_type_key"]): l for l in stmt_lines}
    # payee_locked lines match on (ORIGINAL party, song, type) — the party
    # component is load-bearing: on a multi-party song, only the split party's
    # payment may land on the locked line.
    locked_by_party_song = {
        (l.get("locked_party_key"), l["song_key"], l["royalty_type_key"]): l
        for l in stmt_lines
        if l.get("payee_locked")
    }

    # ---- Gate 4: cross-run conflicts vs stored lines --------------------------
    res_map = {
        (r["party_key"], r["song_key"], r["royalty_type_key"]): r["governing_contract_id"]
        for r in (conflict_resolutions or [])
    }
    cross = []
    for key, entry in list(aggregated.items()):
        stored = by_identity.get(key) or locked_by_party_song.get(key)
        if not stored:
            continue
        stored_ids = {e.get("id") for e in (stored.get("source_contracts") or [])}
        if not stored_ids or stored_ids <= run_set:
            continue
        # Compare PERCENTAGES when both known — expense changes alter every net
        # amount while the splits are identical (a recalculation, not a
        # disagreement). Fall back to amounts only when a percentage is missing.
        if stored.get("percentage") is not None and entry.get("percentage") is not None:
            differs = abs(float(stored["percentage"]) - float(entry["percentage"])) > EPS
        else:
            differs = abs(float(stored["amount_owed"]) - entry["amount"]) > EPS
        if not differs:
            continue
        governing = res_map.get(key)
        if governing is None:
            absent = [e for e in stored["source_contracts"] if e.get("id") not in run_set]
            cross.append(
                {
                    "party_key": key[0],
                    "song_key": key[1],
                    "royalty_type_key": key[2],
                    "party_name": entry["party_name"],
                    "song_title": entry["song_title"],
                    "stored": {
                        "amount": stored["amount_owed"],
                        "percentage": stored.get("percentage"),
                        "asserted_by": absent,
                    },
                    "new": {
                        "amount": round(entry["amount"], 2),
                        "percentage": entry.get("percentage"),
                        "contract_ids": sorted(entry["sources"]),
                    },
                }
            )
        # else: a resolution exists. Governing-in-run → aggregate_payments
        # already filtered to that group (upsert writes the run's number).
        # Governing-ABSENT → aggregate_payments already dropped the identity
        # (its `continue` branch), so it never reaches this loop — the stored
        # line stands untouched.
    if cross:
        raise SyncGateError("conflict", {"scope": "cross_run", "conflicts": cross})

    # ---- Gate 5: revision overlap ---------------------------------------------
    # Candidates are ALWAYS computed first; a client-supplied "replace" id is
    # only honored when this gate would have offered it (fail closed — the
    # decision is client input, never trusted to name an arbitrary statement).
    # Pairs already dismissed as unrelated never re-prompt (either direction).
    dismissed_rows = (
        db.table("royalty_statement_supersessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("kind", "not_related")
        .execute()
    ).data or []
    dismissed = {frozenset((r["old_statement_id"], r["new_statement_id"])) for r in dismissed_rows}
    new_pairs = {(key[0], key[1]) for key in aggregated}
    candidates: dict[str, dict] = {}
    for l in all_lines:
        if l["royalty_statement_id"] == effective_stmt:
            continue
        if frozenset((l["royalty_statement_id"], effective_stmt)) in dismissed:
            continue
        pair = (payee_names.get(l["payee_id"], ""), l["song_key"])
        if pair in new_pairs and _periods_overlap(l["period_start"], l["period_end"], period_start, period_end):
            c = candidates.setdefault(
                l["royalty_statement_id"],
                {
                    "statement_id": l["royalty_statement_id"],
                    "name": l.get("statement_file_name"),
                    "period_start": l["period_start"],
                    "period_end": l["period_end"],
                    "total": 0.0,
                },
            )
            c["total"] += float(l["amount_owed"] or 0)
    replace_id = (revision_decision or {}).get("replace")
    if replace_id:
        if replace_id not in candidates:
            raise SyncGateError("revision", {"candidates": list(candidates.values())})
        if not check_only:
            execute_replace(db, user_id, replace_id, effective_stmt, project_id)
        all_lines = [l for l in all_lines if l["royalty_statement_id"] != replace_id]
    elif candidates:
        if revision_decision and revision_decision.get("none"):
            # Persist the dismissal so the prompt never re-fires for these
            # pairs. Last gate — nothing can raise after this point.
            if not check_only:
                for c in candidates.values():
                    db.table("royalty_statement_supersessions").insert(
                        {
                            "user_id": user_id,
                            "old_statement_id": c["statement_id"],
                            "new_statement_id": effective_stmt,
                            "kind": "not_related",
                        }
                    ).execute()
        else:
            raise SyncGateError("revision", {"candidates": list(candidates.values())})

    if check_only:
        return effective_stmt

    # ---- Apply deferred contract adoptions (all gates passed) ------------------
    for line, new_sources in pending_adoptions:
        history.record(db, user_id, "adopted", dict(line), "contract_adopted")
        db.table("royalty_lines").update({"source_contracts": new_sources}).eq("id", line["id"]).execute()
        line["source_contracts"] = new_sources  # keep in-memory view consistent

    # ---- Upsert engine ----------------------------------------------------------
    stmt_meta = {
        "statement_content_hash": stmt_hash,
        "statement_file_name": (statement_file or {}).get("name"),
    }
    # One works load for the whole run — not one query per inserted line.
    works_by_key = {
        song_key(w.get("title") or ""): w
        for w in (db.table("works_registry").select("id, title").eq("project_id", project_id).execute().data or [])
        if w.get("title")
    }
    matched_line_ids = set()
    for key, entry in aggregated.items():
        locked = locked_by_party_song.get(key)
        existing = locked or by_identity.get(key)
        sources_snapshot = [
            {
                k: v
                for k, v in {
                    "id": cid,
                    "name": (contract_files.get(cid) or {}).get("name"),
                    "hash": (contract_files.get(cid) or {}).get("content_hash"),
                }.items()
                if v is not None
            }
            for cid in sorted(entry["sources"])
        ]
        if existing:
            matched_line_ids.add(existing["id"])
            kept_outside = [e for e in (existing.get("source_contracts") or []) if e.get("id") not in run_set]
            new_sources = sources_snapshot + kept_outside
            changed = abs(float(existing["amount_owed"]) - entry["amount"]) > EPS or (
                (existing.get("percentage") is None) != (entry.get("percentage") is None)
                or (
                    existing.get("percentage") is not None
                    and entry.get("percentage") is not None
                    and abs(float(existing["percentage"]) - entry["percentage"]) > EPS
                )
            )
            if changed:
                history.record(db, user_id, "updated", dict(existing), str(calculation_id))
            if changed or [e.get("id") for e in (existing.get("source_contracts") or [])] != [
                e["id"] for e in new_sources
            ]:
                db.table("royalty_lines").update(
                    {
                        "amount_owed": entry["amount"],
                        "percentage": entry.get("percentage"),
                        "role": entry.get("role"),
                        "royalty_type": entry.get("royalty_type"),
                        "source_contracts": new_sources,
                        "calculation_id": calculation_id,
                        "period_start": period_start,
                        "period_end": period_end,
                        "statement_currency": statement_currency,
                        **stmt_meta,
                    }
                ).eq("id", existing["id"]).execute()
        else:
            payee_id = upsert_payee(db, user_id, entry["party_name"])
            work = works_by_key.get(key[1])
            db.table("royalty_lines").insert(
                {
                    "user_id": user_id,
                    "calculation_id": calculation_id,
                    "royalty_statement_id": effective_stmt,
                    "payee_id": payee_id,
                    "project_id": project_id,
                    "work_id": work["id"] if work else None,
                    "song_title": entry["song_title"],
                    "song_key": key[1],
                    "role": entry.get("role"),
                    "royalty_type": entry.get("royalty_type"),
                    "royalty_type_key": key[2],
                    "percentage": entry.get("percentage"),
                    "song_revenue": None,
                    "amount_owed": entry["amount"],
                    "statement_currency": statement_currency,
                    "period_start": period_start,
                    "period_end": period_end,
                    "source_contracts": sources_snapshot,
                    "payee_locked": False,
                    **stmt_meta,
                }
            ).execute()

    # ---- Stale-delete / partial retraction (run-scoped authority) -------------
    for line in stmt_lines:
        if line["id"] in matched_line_ids:
            continue
        source_ids = {e.get("id") for e in (line.get("source_contracts") or [])}
        if not source_ids:
            continue  # unknown owner: NEVER ours to delete (∅ ⊆ S must not fire)
        if source_ids <= run_set:
            history.record(db, user_id, "deleted", dict(line), str(calculation_id))
            db.table("royalty_lines").delete().eq("id", line["id"]).execute()
        elif source_ids & run_set:
            kept = [e for e in line["source_contracts"] if e.get("id") not in run_set]
            history.record(db, user_id, "source_removed", dict(line), str(calculation_id))
            db.table("royalty_lines").update({"source_contracts": kept}).eq("id", line["id"]).execute()

    return effective_stmt


def execute_replace(db, user_id: str, old_statement_id: str, new_statement_id: str, project_id: str) -> None:
    """User confirmed 'this replaces statement X': supersede X's lines, re-point
    its coverage to the new bucket, tombstone X, and drop X's calc cache."""
    old_lines = (
        db.table("royalty_lines")
        .select("*")
        .eq("user_id", user_id)
        .eq("royalty_statement_id", old_statement_id)
        .eq("project_id", project_id)
        .execute()
    ).data or []
    for line in old_lines:
        history.record(db, user_id, "superseded", dict(line), "revision_replace")
        db.table("royalty_lines").delete().eq("id", line["id"]).execute()

    # Coverage has no user_id column — ownership goes through royalty_payouts.
    # Scope to the caller's payouts so a shared statement id can never let one
    # user re-point another user's coverage.
    payout_ids = [
        p["id"] for p in (db.table("royalty_payouts").select("id").eq("user_id", user_id).execute().data or [])
    ]
    cov_rows = []
    if payout_ids:
        cov_rows = (
            db.table("royalty_payout_coverage")
            .select("*")
            .eq("royalty_statement_id", old_statement_id)
            .eq("project_id", project_id)
            .in_("payout_id", payout_ids)
            .execute()
        ).data or []
    for cov in cov_rows:
        history.record(db, user_id, "coverage_moved", dict(cov), "revision_replace")
        # The composite PK forbids two rows per (payout, statement, project):
        # if this payout already covers the NEW bucket, merge amounts instead.
        collision = (
            db.table("royalty_payout_coverage")
            .select("*")
            .eq("payout_id", cov["payout_id"])
            .eq("royalty_statement_id", new_statement_id)
            .eq("project_id", project_id)
            .execute()
        ).data or []
        if collision:
            db.table("royalty_payout_coverage").update(
                {
                    "covered_amount": float(collision[0]["covered_amount"]) + float(cov["covered_amount"]),
                    # moved_from marks the merged amount as revision-moved: without
                    # it, a later revert+cancel would destroy the merged paid amount.
                    "moved_from": {
                        "statement_id": old_statement_id,
                        "project_id": project_id,
                        "action": "revision_replace",
                    },
                }
            ).eq("id", collision[0]["id"]).execute()
            db.table("royalty_payout_coverage").delete().eq("id", cov["id"]).execute()
        else:
            db.table("royalty_payout_coverage").update(
                {
                    "royalty_statement_id": new_statement_id,
                    "moved_from": {
                        "statement_id": old_statement_id,
                        "project_id": project_id,
                        "action": "revision_replace",
                    },
                }
            ).eq("id", cov["id"]).execute()

    db.table("royalty_statement_supersessions").insert(
        {
            "user_id": user_id,
            "old_statement_id": old_statement_id,
            "new_statement_id": new_statement_id,
            "kind": "superseded",
        }
    ).execute()
    # Cache is rebuildable; the tombstone is the durable guard.
    db.table("royalty_calculations").delete().eq("royalty_statement_id", old_statement_id).eq(
        "user_id", user_id
    ).execute()


def remove_contract_from_ledger(db, user_id: str, contract_id: str) -> None:
    """Contract deletion guardrail: strip this contract's assertions.
    Sole-source lines are deleted (paid imbalance surfaces as credit via the
    derived math); shared lines just lose the entry. Payment records untouched."""
    lines = (
        db.table("royalty_lines")
        .select("*")
        .eq("user_id", user_id)
        .contains("source_contracts", [{"id": contract_id}])
        .execute()
    ).data or []
    for line in lines:
        kept = [e for e in (line.get("source_contracts") or []) if e.get("id") != contract_id]
        if kept:
            history.record(db, user_id, "source_removed", dict(line), "contract_deleted")
            db.table("royalty_lines").update({"source_contracts": kept}).eq("id", line["id"]).execute()
        else:
            history.record(db, user_id, "deleted", dict(line), "contract_deleted")
            db.table("royalty_lines").delete().eq("id", line["id"]).execute()
