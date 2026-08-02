from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch09_tracing.get_chat_model",
        lambda **kw: fake,
    )


def test_setup_creates_provider():
    """setup_tracing() 必须设置一个 TracerProvider 且不抛错。"""
    from opentelemetry import trace
    from learn_lang_chain.chapters.ch09_tracing import setup_tracing
    setup_tracing("test-svc")
    provider = trace.get_tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("service.name") == "test-svc"


def test_chain_invocation_emits_spans(monkeypatch):
    """调用链后，InMemorySpanExporter 必须收到至少一个 span。"""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from learn_lang_chain.chapters.ch09_tracing import (
        run_traced_demo,
        setup_tracing,
    )

    patch_factory(monkeypatch, ["ok"])
    exporter = InMemorySpanExporter()
    setup_tracing("test-svc", exporter=exporter)

    out = run_traced_demo()
    assert out == "ok"

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1, "expected at least one span from LangChain instrumentation"
