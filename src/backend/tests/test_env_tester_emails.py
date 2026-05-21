"""Tests for env_tester_emails / is_env_tester — mirrors env_admin_emails pattern."""


def test_empty_env_returns_empty_set(monkeypatch):
    from subscriptions.admin_auth import env_tester_emails

    monkeypatch.delenv("TESTER_EMAILS", raising=False)
    assert env_tester_emails() == set()


def test_parses_comma_separated_lowercased(monkeypatch):
    from subscriptions.admin_auth import env_tester_emails

    monkeypatch.setenv("TESTER_EMAILS", " Alice@Foo.com , bob@bar.com ,charlie@baz.com,")
    assert env_tester_emails() == {"alice@foo.com", "bob@bar.com", "charlie@baz.com"}


def test_is_env_tester_true_for_listed_email(monkeypatch):
    from subscriptions.admin_auth import is_env_tester

    monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
    assert is_env_tester("Tester@Example.com") is True


def test_is_env_tester_false_for_unlisted(monkeypatch):
    from subscriptions.admin_auth import is_env_tester

    monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
    assert is_env_tester("other@example.com") is False


def test_is_env_tester_none_or_empty(monkeypatch):
    from subscriptions.admin_auth import is_env_tester

    monkeypatch.setenv("TESTER_EMAILS", "tester@example.com")
    assert is_env_tester(None) is False
    assert is_env_tester("") is False
