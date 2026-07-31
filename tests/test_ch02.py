from langchain_core.language_models.fake_chat_models import FakeListChatModel


def fake_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch02_prompt_template.get_chat_model",
        lambda **kw: fake,
    )


def test_prompt_template_translate(monkeypatch):
    fake_factory(monkeypatch, ["Bonjour"])
    from learn_lang_chain.chapters.ch02_prompt_template import translate
    assert translate("hello", "法语") == "Bonjour"


def test_chat_prompt_template_persona(monkeypatch):
    fake_factory(monkeypatch, ["I am a poet."])
    from learn_lang_chain.chapters.ch02_prompt_template import persona_reply
    assert "poet" in persona_reply().lower()