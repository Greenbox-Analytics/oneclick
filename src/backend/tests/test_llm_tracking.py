"""Tests for utils/llm/tracking.py — TrackedOpenAI proxy + ai_usage_log writes."""

from unittest.mock import MagicMock

from utils.llm.tracking import TrackedOpenAI, llm_call_context, set_llm_context

TEST_USER = "00000000-0000-0000-0000-000000000001"


def _fake_response(model="gpt-5-mini", prompt=1000, completion=500, cached=200):
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = prompt
    resp.usage.completion_tokens = completion
    resp.usage.prompt_tokens_details.cached_tokens = cached
    return resp


def _tracked(inner, supabase):
    return TrackedOpenAI(inner, get_supabase=lambda: supabase)


class TestTrackedOpenAI:
    def test_delegates_and_returns_inner_response(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = _fake_response()
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            out = client.chat.completions.create(model="gpt-5-mini", messages=[])
        assert out is inner.chat.completions.create.return_value
        inner.chat.completions.create.assert_called_once()

    def test_logs_usage_row_with_context(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = _fake_response()
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            client.chat.completions.create(model="gpt-5-mini", messages=[])
        sb.table.assert_called_with("ai_usage_log")
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["user_id"] == TEST_USER
        assert row["tool"] == "zoe"
        assert row["input_tokens"] == 1000
        assert row["cached_tokens"] == 200
        assert row["cost_usd"] is not None

    def test_no_context_no_row_no_crash(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = _fake_response()
        client = _tracked(inner, sb)
        assert llm_call_context.get() is None
        client.chat.completions.create(model="gpt-5-mini", messages=[])
        sb.table.assert_not_called()

    def test_logging_failure_never_propagates(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = _fake_response()
        sb.table.side_effect = RuntimeError("db down")
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "oneclick"):
            out = client.chat.completions.create(model="gpt-5.2", messages=[])
        assert out is inner.chat.completions.create.return_value  # no raise

    def test_failed_call_logged_with_success_false_and_reraised(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.side_effect = RuntimeError("openai 500")
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            try:
                client.chat.completions.create(model="gpt-5-mini", messages=[])
                raise AssertionError("should have raised")
            except RuntimeError:
                pass
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["success"] is False

    def test_unknown_attrs_delegate(self):
        inner, sb = MagicMock(), MagicMock()
        client = _tracked(inner, sb)
        assert client.files is inner.files

    def test_cost_uses_requested_model_not_echoed_snapshot(self):
        # The API echoes versioned ids that aren't MODEL_RATES keys; the rate
        # lookup must key on what we REQUESTED or cost_usd silently nulls out.
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = _fake_response(model="gpt-5-mini-2026-03-01")
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            client.chat.completions.create(model="gpt-5-mini", messages=[])
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["model"] == "gpt-5-mini"
        assert row["cost_usd"] is not None


class TestTrackedEmbeddings:
    def test_embeddings_logged_with_context(self):
        inner, sb = MagicMock(), MagicMock()
        resp = MagicMock()
        resp.model = "text-embedding-3-small-v2"
        resp.usage.prompt_tokens = 40
        resp.usage.completion_tokens = None
        resp.usage.prompt_tokens_details = None
        inner.embeddings.create.return_value = resp
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            out = client.embeddings.create(model="text-embedding-3-small", input=["q"])
        assert out is resp
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["model"] == "text-embedding-3-small"
        assert row["input_tokens"] == 40

    def test_embeddings_failure_logged_and_reraised(self):
        inner, sb = MagicMock(), MagicMock()
        inner.embeddings.create.side_effect = RuntimeError("boom")
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "oneclick"):
            try:
                client.embeddings.create(model="text-embedding-3-small", input=["q"])
                raise AssertionError("should have raised")
            except RuntimeError:
                pass
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["success"] is False


def _fake_stream_chunks(model="gpt-5-mini"):
    """Simulate OpenAI streaming: N content chunks, then the terminal usage chunk."""
    c1 = MagicMock(choices=[MagicMock()], usage=None)
    c2 = MagicMock(choices=[MagicMock()], usage=None)
    final = MagicMock(choices=[], model=model)
    final.usage.prompt_tokens = 1200
    final.usage.completion_tokens = 800
    final.usage.prompt_tokens_details.cached_tokens = 300
    return [c1, c2, final]


class TestStreamedCalls:
    def test_stream_options_injected(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = iter(_fake_stream_chunks())
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            list(client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True))
        kwargs = inner.chat.completions.create.call_args.kwargs
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_usage_recorded_from_terminal_chunk(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = iter(_fake_stream_chunks())
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            chunks = list(client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True))
        assert len(chunks) == 3  # all chunks pass through, incl. terminal
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["input_tokens"] == 1200 and row["output_tokens"] == 800
        assert row["cached_tokens"] == 300 and row["cost_usd"] is not None

    def test_context_snapshot_survives_exit(self):
        # Identity captured at create() time — recording happens after the
        # with-block exits (mirrors threadpool-iterated sync generators).
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = iter(_fake_stream_chunks())
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            stream = client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True)
        list(stream)  # consumed OUTSIDE the context
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["user_id"] == TEST_USER

    def test_mid_stream_error_logged_failed(self):
        def broken():
            yield MagicMock(choices=[MagicMock()], usage=None)
            raise RuntimeError("connection reset")

        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = broken()
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            stream = client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True)
            try:
                list(stream)
                raise AssertionError("should have raised")
            except RuntimeError:
                pass
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["success"] is False

    def test_client_disconnect_still_records(self):
        # GeneratorExit (client disconnect) bypasses `except Exception` — the
        # guarded finally must still record a success=False row.
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = iter(_fake_stream_chunks())
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            stream = client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True)
        it = iter(stream)
        next(it)  # consume one chunk...
        it.close()  # ...then the server abandons the generator (disconnect)
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["success"] is False

    def test_caller_stream_options_not_clobbered(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.return_value = iter(_fake_stream_chunks())
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            list(
                client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[],
                    stream=True,
                    stream_options={"include_usage": False, "other_key": 1},
                )
            )
        kwargs = inner.chat.completions.create.call_args.kwargs
        assert kwargs["stream_options"] == {"include_usage": False, "other_key": 1}

    def test_stream_create_failure_logged(self):
        inner, sb = MagicMock(), MagicMock()
        inner.chat.completions.create.side_effect = RuntimeError("auth error")
        client = _tracked(inner, sb)
        with set_llm_context(TEST_USER, "zoe"):
            try:
                client.chat.completions.create(model="gpt-5-mini", messages=[], stream=True)
                raise AssertionError("should have raised")
            except RuntimeError:
                pass
        row = sb.table.return_value.insert.call_args[0][0]
        assert row["success"] is False


class TestIterWithLlmContext:
    def test_context_live_during_each_resumption(self):
        from utils.llm.tracking import iter_with_llm_context, llm_call_context

        seen = []

        def gen():
            seen.append(llm_call_context.get())
            yield "a"
            seen.append(llm_call_context.get())
            yield "b"

        assert list(iter_with_llm_context(TEST_USER, "zoe", gen())) == ["a", "b"]
        assert seen == [(TEST_USER, "zoe"), (TEST_USER, "zoe")]
        assert llm_call_context.get() is None  # reset after each step


class TestSubmitWithContext:
    def test_context_propagates_into_worker_thread(self):
        from concurrent.futures import ThreadPoolExecutor

        from utils.llm.tracking import submit_with_context

        def read_ctx():
            return llm_call_context.get()

        with ThreadPoolExecutor(max_workers=1) as pool, set_llm_context(TEST_USER, "oneclick"):
            via_helper = submit_with_context(pool, read_ctx).result()
            via_plain = pool.submit(read_ctx).result()
        assert via_helper == (TEST_USER, "oneclick")  # helper propagates
        assert via_plain is None  # documents WHY the helper exists


class TestClientWrapping:
    def test_get_openai_client_returns_tracked(self, monkeypatch):
        import utils.llm.client as client_mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(client_mod, "openai_client", None)
        c = client_mod.get_openai_client()
        assert isinstance(c, TrackedOpenAI)

    def test_zoe_module_client_is_tracked(self, monkeypatch):
        # Module-level import raises if OPENAI_API_KEY is unset; pin it so the
        # first import works even in a bare env (load_dotenv won't override it).
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import zoe_chatbot.contract_chatbot as zoe_mod

        assert isinstance(zoe_mod.openai_client, TrackedOpenAI)
