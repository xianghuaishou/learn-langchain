from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch04_lcel_chain.get_chat_model",
        lambda **kw: fake,
    )


def test_chain_invoke(monkeypatch):
    patch_factory(monkeypatch, ["STADIUM"])
    from learn_lang_chain.chapters.ch04_lcel_chain import uppercase_topic
    assert uppercase_topic.invoke("Barcelona") == "STADIUM"


def test_batch(monkeypatch):
    patch_factory(monkeypatch, ["A", "B", "C"])
    from learn_lang_chain.chapters.ch04_lcel_chain import uppercase_topic
    out = uppercase_topic.batch(["a", "b", "c"])
    assert out == ["A", "B", "C"]