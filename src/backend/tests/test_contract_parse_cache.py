import json

from utils.contract_parsing.cache import (
    deserialize_contract_data,
    serialize_contract_data,
)
from utils.contract_parsing.models import ContractData, Party, RoyaltyShare, Work


def _sample() -> ContractData:
    return ContractData(
        parties=[Party(name="Jane Doe", role="artist"), Party(name="Acme Records", role="label")],
        works=[Work(title="Midnight", work_type="master recording")],
        royalty_shares=[
            RoyaltyShare(party_name="Jane Doe", royalty_type="Streaming", percentage=50.0, terms=None, basis="net"),
            RoyaltyShare(
                party_name="Acme Records", royalty_type="Master", percentage=50.0, terms="recoupable", basis=None
            ),
        ],
        contract_summary="",
        default_basis="gross",
    )


def test_serialize_round_trip_equal():
    cd = _sample()
    assert deserialize_contract_data(serialize_contract_data(cd)) == cd


def test_serialize_round_trip_through_json_preserves_none():
    cd = _sample()
    restored = deserialize_contract_data(json.loads(json.dumps(serialize_contract_data(cd))))
    assert restored == cd
    assert restored.royalty_shares[0].basis == "net"
    assert restored.royalty_shares[0].terms is None
    assert restored.royalty_shares[1].basis is None
    assert restored.royalty_shares[1].terms == "recoupable"
    assert restored.default_basis == "gross"


import hashlib
from unittest.mock import MagicMock

from utils.contract_parsing.cache import get_or_parse, parser_version


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mk_parser(cd):
    p = MagicMock()
    p.parse_contract.return_value = cd
    return p


def _db_with_hit(cd):
    db = MagicMock()
    (
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value
    ) = MagicMock(data={"parsed": serialize_contract_data(cd)})
    return db


def _db_with_miss():
    db = MagicMock()
    (
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value
    ) = MagicMock(data=None)
    return db


def test_cache_hit_skips_parser():
    cd = _sample()
    parser = _mk_parser(cd)
    db = _db_with_hit(cd)

    # load_text IS invoked (its result is needed to compute the key), but the LLM parse is
    # skipped on a hit.
    out = get_or_parse(db, lambda: "text", parser=parser)
    assert out == cd
    parser.parse_contract.assert_not_called()


def test_cache_miss_parses_once_and_upserts():
    cd = _sample()
    parser = _mk_parser(cd)
    db = _db_with_miss()

    out = get_or_parse(db, lambda: "the markdown", parser=parser)
    assert out == cd
    parser.parse_contract.assert_called_once_with(full_text="the markdown")
    upsert = db.table.return_value.upsert
    upsert.assert_called_once()
    payload = upsert.call_args.args[0]
    assert payload["content_hash"] == _key("the markdown")
    assert payload["parser_version"] == parser_version()
    assert upsert.call_args.kwargs.get("on_conflict") == "content_hash,parser_version"


def test_key_is_hash_of_canonical_text_single_valued():
    # Two callers whose markdown differs only by [[PAGE n]] markers must land on the SAME
    # key (canonicalization), and the key is exactly sha256 of the stripped text.
    db1 = _db_with_miss()
    get_or_parse(db1, lambda: "\n\n[[PAGE 1]]\n\nSame body", parser=_mk_parser(_sample()))
    key_with_markers = db1.table.return_value.upsert.call_args.args[0]["content_hash"]

    db2 = _db_with_miss()
    get_or_parse(db2, lambda: "Same body", parser=_mk_parser(_sample()))
    key_plain = db2.table.return_value.upsert.call_args.args[0]["content_hash"]

    assert key_with_markers == key_plain == _key("Same body")


def test_marker_stripping_is_applied_before_parse():
    cd = _sample()
    parser = _mk_parser(cd)
    db = _db_with_miss()
    get_or_parse(db, lambda: "\n\n[[PAGE 1]]\n\nHello world", parser=parser)
    parser.parse_contract.assert_called_once_with(full_text="Hello world")


def test_reads_are_scoped_to_current_parser_version():
    parser = _mk_parser(_sample())
    db = _db_with_miss()
    get_or_parse(db, lambda: "t", parser=parser)
    version_eq = db.table.return_value.select.return_value.eq.return_value.eq.call_args
    assert version_eq.args == ("parser_version", parser_version())


def test_bypass_skips_read_but_writes():
    cd = _sample()
    parser = _mk_parser(cd)
    db = _db_with_hit(cd)
    out = get_or_parse(db, lambda: "t", parser=parser, bypass=True)
    assert out == cd
    parser.parse_contract.assert_called_once()
    db.table.return_value.upsert.assert_called_once()


def test_read_failure_falls_back_to_live_parse():
    cd = _sample()
    parser = _mk_parser(cd)
    db = MagicMock()
    db.table.return_value.select.side_effect = RuntimeError("db down")
    out = get_or_parse(db, lambda: "t", parser=parser)
    assert out == cd
    parser.parse_contract.assert_called_once()


def test_write_failure_is_swallowed():
    cd = _sample()
    parser = _mk_parser(cd)
    db = _db_with_miss()
    db.table.return_value.upsert.side_effect = RuntimeError("write blew up")
    out = get_or_parse(db, lambda: "t", parser=parser)
    assert out == cd


def test_no_db_parses_without_touching_cache():
    cd = _sample()
    parser = _mk_parser(cd)
    out = get_or_parse(None, lambda: "t", parser=parser)
    assert out == cd
    parser.parse_contract.assert_called_once()


def test_deserialize_rejects_unknown_fields():
    import pytest

    d = serialize_contract_data(_sample())
    d["surprise_field"] = 1
    with pytest.raises(ValueError, match="Unexpected ContractData fields"):
        deserialize_contract_data(d)
