"""Tests for scripts/clear_oneclick_cache.py — truncates OneClick cache tables."""

from unittest.mock import MagicMock

import pytest


def _make_supabase_mock(calc_count: int = 3, junction_count: int = 5) -> MagicMock:
    """Supabase client mock that returns row counts for select, no-op for delete."""
    sb = MagicMock()

    def _table(name):
        builder = MagicMock()
        builder.select.return_value = builder
        builder.delete.return_value = builder
        builder.neq.return_value = builder

        if name == "royalty_calculations":
            count = calc_count
        elif name == "royalty_calculation_contracts":
            count = junction_count
        else:
            count = 0

        builder.execute.return_value = MagicMock(data=[], count=count)
        return builder

    sb.table.side_effect = _table
    return sb


class TestClearOneclickCache:
    def test_deletes_junction_then_main(self, monkeypatch, capsys):
        from scripts import clear_oneclick_cache

        sb = _make_supabase_mock(calc_count=3, junction_count=5)
        monkeypatch.setattr(clear_oneclick_cache, "_get_supabase", lambda: sb)

        rc = clear_oneclick_cache.main(["--yes"])
        assert rc == 0

        # FK order: junction must be deleted before parent.
        called_tables = [c.args[0] for c in sb.table.call_args_list]
        first_delete_idx = next(i for i, t in enumerate(called_tables) if t == "royalty_calculation_contracts")
        first_main_idx = next(i for i, t in enumerate(called_tables) if t == "royalty_calculations")
        assert first_delete_idx < first_main_idx, "junction must be cleared before parent"

        out = capsys.readouterr().out
        assert "Deleted" in out

    def test_dry_run_does_not_delete(self, monkeypatch, capsys):
        from scripts import clear_oneclick_cache

        sb = _make_supabase_mock()
        monkeypatch.setattr(clear_oneclick_cache, "_get_supabase", lambda: sb)

        rc = clear_oneclick_cache.main(["--dry-run"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_missing_env_exits_2(self, monkeypatch):
        from scripts import clear_oneclick_cache

        monkeypatch.delenv("VITE_SUPABASE_URL", raising=False)
        monkeypatch.delenv("VITE_SUPABASE_SECRET_KEY", raising=False)

        with pytest.raises(SystemExit) as exc:
            clear_oneclick_cache._get_supabase()
        assert exc.value.code == 2
