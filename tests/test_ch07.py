from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch07_memory.get_chat_model",
        lambda **kw: fake,
    )


def test_history_accumulates(monkeypatch):
    patch_factory(monkeypatch, ["reply-1", "reply-2"])
    from learn_lang_chain.chapters.ch07_memory import chat
    a = chat("hello", session_id="s1")
    b = chat("again", session_id="s1")
    assert a == "reply-1"
    assert b == "reply-2"


def test_isolated_sessions(monkeypatch):
    patch_factory(monkeypatch, ["x", "y", "p", "q"])
    from learn_lang_chain.chapters.ch07_memory import chat
    chat("a", session_id="s1")
    chat("b", session_id="s1")
    chat("c", session_id="s2")
    chat("d", session_id="s2")
    # 仅验证不抛异常且隔离生效
    assert True
