"""Minimal in-memory Supabase fake for royalties engine tests.

Mirrors the REAL post-migration schema — never add columns here the real
schema lacks."""

USER = "u1"


class FakeQuery:
    def __init__(self, table):
        self.t, self.filters, self._update, self._insert, self._delete = table, [], None, None, False

    def select(self, *_, **__):
        return self

    def eq(self, col, val):
        self.filters.append(lambda r: r.get(col) == val)
        return self

    def neq(self, col, val):
        self.filters.append(lambda r: r.get(col) != val)
        return self

    def in_(self, col, vals):
        self.filters.append(lambda r: r.get(col) in vals)
        return self

    def contains(self, col, frag):
        # jsonb @> for the one shape the code uses: a list-of-dicts fragment
        # where each fragment dict subset-matches some array element.
        self.filters.append(
            lambda r: all(any(all(e.get(k) == v for k, v in f.items()) for e in r.get(col, [])) for f in frag)
        )
        return self

    def update(self, patch):
        self._update = patch
        return self

    def insert(self, rows):
        self._insert = rows if isinstance(rows, list) else [rows]
        return self

    def delete(self):
        self._delete = True
        return self

    def execute(self):
        rows = [r for r in self.t.rows if all(f(r) for f in self.filters)]
        if self._insert is not None:
            for r in self._insert:
                r.setdefault("id", f"{self.t.name}-{len(self.t.rows) + 1}")
                self.t.rows.append(dict(r))
            return type("R", (), {"data": self._insert})
        if self._update is not None:
            for r in rows:
                r.update(self._update)
            return type("R", (), {"data": rows})
        if self._delete:
            self.t.rows[:] = [r for r in self.t.rows if r not in rows]
            return type("R", (), {"data": rows})
        return type("R", (), {"data": [dict(r) for r in rows]})


class FakeTable:
    def __init__(self, name):
        self.name, self.rows = name, []


class FakeDB:
    def __init__(self):
        self.tables = {
            n: FakeTable(n)
            for n in (
                "royalty_lines",
                "royalty_ledger_history",
                "royalty_statement_supersessions",
                "royalty_payout_coverage",
                "royalty_calculations",
                "royalty_payees",
                "project_files",
                "works_registry",
                "royalty_payouts",
            )
        }

    def table(self, name):
        return FakeQuery(self.tables[name])
