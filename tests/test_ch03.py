import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch03_output_parser.get_chat_model",
        lambda **kw: fake,
    )


def test_str_parser(monkeypatch):
    patch_factory(monkeypatch, ["pong"])
    from learn_lang_chain.chapters.ch03_output_parser import echo
    assert echo("ping") == "pong"


def test_pydantic_parser(monkeypatch):
    patch_factory(
        monkeypatch,
        ['{"sentiment": "positive", "score": 0.9}'],
    )
    from learn_lang_chain.chapters.ch03_output_parser import analyze
    result = analyze("这家店服务超棒！")
    assert result.sentiment == "positive"
    assert result.score == pytest.approx(0.9)
