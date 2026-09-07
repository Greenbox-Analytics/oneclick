from unittest.mock import MagicMock

from pagination import fetch_all


def _factory(pages):
    """A zero-arg query factory: each call returns a FRESH builder (as the real
    caller's lambda would) whose execute() yields the next page. Returns the
    factory and the list of builders it handed out, so the test can prove one
    .range() per builder — the whole reason fetch_all takes a factory."""
    it = iter(pages)
    builders = []

    def make():
        b = MagicMock()
        b.order.return_value = b
        b.range.return_value = b
        b.execute.return_value = MagicMock(data=next(it))
        builders.append(b)
        return b

    return make, builders


def test_fetch_all_pages_until_a_short_page_with_a_fresh_builder_per_page():
    make, builders = _factory(
        [
            [{"i": n} for n in range(500)],
            [{"i": n} for n in range(500, 1000)],
            [{"i": 1000}] * 5,
        ]
    )
    rows = fetch_all(make)
    assert len(rows) == 1005
    assert len(builders) == 3
    # Exactly ONE range() per builder — a reused builder would show two calls
    # on one object (and, on real postgrest, appended offset/limit params).
    assert [b.range.call_args.args for b in builders] == [(0, 499), (500, 999), (1000, 1499)]
    assert all(b.range.call_count == 1 for b in builders)


def test_fetch_all_orders_every_page_so_offsets_line_up():
    # .range() is bare LIMIT/OFFSET: without an ORDER BY the two pages have no
    # guaranteed common row order and can repeat or skip rows.
    make, builders = _factory([[{"i": n} for n in range(500)], []])
    fetch_all(make)
    assert all(b.order.call_args.args == ("id",) for b in builders)
    assert all(b.order.call_count == 1 for b in builders)


def test_fetch_all_order_by_is_overridable():
    make, builders = _factory([[]])
    fetch_all(make, order_by="created_at")
    assert builders[0].order.call_args.args == ("created_at",)


def test_fetch_all_page_size_stays_under_the_server_max_rows_cap():
    # 1,000 is Supabase's db-max-rows default; a page sitting ON it comes back
    # short whenever the cap is lowered, ending the loop mid-table.
    make, builders = _factory([[]])
    fetch_all(make)
    assert builders[0].range.call_args.args[1] < 999


def test_fetch_all_exact_multiple_makes_one_extra_empty_call():
    make, builders = _factory([[{"i": n} for n in range(500)], []])
    assert len(fetch_all(make)) == 500
    assert len(builders) == 2


def test_fetch_all_empty():
    assert fetch_all(_factory([[]])[0]) == []
    assert fetch_all(_factory([None])[0]) == []
