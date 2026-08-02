from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch09_tracing.get_chat_model",
        lambda **kw: fake,
    )


import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    """在每个测试之间重置 ch09 的模块级状态与 OTel 全局状态。

    之所以需要这个 fixture：``setup_tracing`` 会修改 OTel 的全局 TracerProvider
    以及 ``BaseCallbackManager.__init__``。如果不重置，测试之间会出现
    上一次 instrument 的 wrapper 仍指向旧 provider、新 exporter 收不到 span
    等问题。
    """
    import learn_lang_chain.chapters.ch09_tracing as ch09
    from langchain_core.callbacks import BaseCallbackManager

    ch09._ever_instrumented = False
    cur = BaseCallbackManager.__init__
    if hasattr(cur, "__wrapped__"):
        BaseCallbackManager.__init__ = cur.__wrapped__  # type: ignore[attr-defined]

    from opentelemetry import trace
    try:
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    except AttributeError:
        pass


from opentelemetry import trace
from learn_lang_chain.chapters.ch09_tracing import setup_tracing


def test_setup_creates_provider():
    """setup_tracing() 必须设置一个 TracerProvider 且不抛错。"""
    setup_tracing("test-svc")
    provider = trace.get_tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("service.name") == "test-svc"


def test_chain_invocation_emits_spans(monkeypatch):
    """调用链后，InMemorySpanExporter 必须收到至少一个 span。"""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from learn_lang_chain.chapters.ch09_tracing import run_traced_demo

    patch_factory(monkeypatch, ["ok"])
    exporter = InMemorySpanExporter()
    setup_tracing("test-svc", exporter=exporter)

    out = run_traced_demo()
    assert out == "ok"

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1, "expected at least one span from LangChain instrumentation"
    # 至少有一个 span 应属于 LangChain 的工作流节点（由 LangchainInstrumentor 产生）
    assert any(
        s.attributes.get("gen_ai.provider.name") == "langchain"
        or "langchain" in (s.name or "").lower()
        for s in spans
    )