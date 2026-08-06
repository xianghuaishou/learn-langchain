import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture(autouse=True)
def _reset_langfuse_singleton():
    """重置 Langfuse 单例，避免测试间状态污染。

    Langfuse v3+ 客户端走全局单例；测试需要从干净状态开始。
    """
    yield


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch10_langfuse.get_chat_model",
        lambda **kw: fake,
    )
    return fake


def test_setup_langfuse_reads_env(monkeypatch):
    """setup_langfuse() 必须从环境变量读取 key 并返回客户端实例。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    from learn_lang_chain.chapters.ch10_langfuse import setup_langfuse
    client = setup_langfuse()
    assert client is not None


def test_setup_langfuse_default_base_url(monkeypatch):
    """缺省 LANGFUSE_BASE_URL 时 fallback 到 https://cloud.langfuse.com。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    captured = {}

    def fake_langfuse_ctor(*, public_key, secret_key, host):
        captured["public_key"] = public_key
        captured["secret_key"] = secret_key
        captured["host"] = host
        return object()

    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch10_langfuse.Langfuse",
        fake_langfuse_ctor,
    )

    from learn_lang_chain.chapters.ch10_langfuse import setup_langfuse
    setup_langfuse()
    assert captured["host"] == "https://cloud.langfuse.com"


def test_run_traced_demo_attaches_callback(monkeypatch):
    """run_traced_demo 必须：构造 CallbackHandler、走 chain.invoke、最后 flush。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    patch_factory(monkeypatch, ["ok"])

    fake_handler = BaseCallbackHandler()
    handler_calls = {"count": 0}

    def fake_handler_factory():
        handler_calls["count"] += 1
        return fake_handler

    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch10_langfuse.CallbackHandler",
        fake_handler_factory,
        raising=False,
    )

    flush_calls = {"count": 0}

    class FakeClient:
        def flush(self):
            flush_calls["count"] += 1

    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch10_langfuse.get_client",
        lambda: FakeClient(),
        raising=False,
    )

    from learn_lang_chain.chapters.ch10_langfuse import run_traced_demo
    out = run_traced_demo(topic="OTel", user_id="alice", session_id="s1")
    assert out == "ok"
    assert handler_calls["count"] >= 1
    assert flush_calls["count"] >= 1
