from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch05_tools_agent.get_chat_model",
        lambda **kw: fake,
    )


def test_tools_listed():
    from learn_lang_chain.chapters.ch05_tools_agent import TOOLS
    names = {t.name for t in TOOLS}
    assert {"get_weather", "calculator"} <= names


def test_get_weather():
    from learn_lang_chain.chapters.ch05_tools_agent import get_weather
    assert "晴" in get_weather.invoke({"city": "上海"})


def test_calculator():
    from learn_lang_chain.chapters.ch05_tools_agent import calculator
    assert calculator.invoke({"expr": "2 + 3 * 4"}) == 14
