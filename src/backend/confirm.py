"""Shared helpers for typed destructive-action confirmations (delete-<name>)."""

import unicodedata


class ConfirmationError(Exception):
    """Raised when a destructive action's typed confirmation name doesn't match."""


def normalize_name(s: str | None) -> str:
    """Trim + Unicode-NFC-normalize a name for confirmation comparison."""
    return unicodedata.normalize("NFC", (s or "").strip())
