"""Append-only recorder for royalty_ledger_history.

History is load-bearing: a failed history write must abort the mutation it
precedes, so exceptions are deliberately NOT caught here. Valid actions are
enforced by the table's CHECK constraint — the single source of truth (a
Python-side duplicate list already drifted once and shipped a runtime crash).
"""


def record(db, user_id: str, action: str, old_row: dict, cause: str) -> None:
    db.table("royalty_ledger_history").insert(
        {
            "user_id": user_id,
            "action": action,
            "old_row": old_row,
            "cause": cause,
        }
    ).execute()
