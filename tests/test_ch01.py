from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_invoke_returns_string(monkeypatch):
    fake = FakeListChatModel(responses=["hello"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch01_basic_chat.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch01_basic_chat import ask
    assert ask("hi") == "hello"


def test_stream_yields_chunks(monkeypatch):
    fake = FakeListChatModel(responses=["abc"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch01_basic_chat.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch01_basic_chat import stream_chunks
    chunks = list(stream_chunks("hi"))
    assert "abc" in "".join(chunks)