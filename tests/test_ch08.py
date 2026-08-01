from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_describe_image_url(monkeypatch):
    fake = FakeListChatModel(responses=["一只猫"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch08_multimodal.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch08_multimodal import describe_image_url
    out = describe_image_url("https://example.com/cat.jpg")
    assert "猫" in out


def test_message_payload_shape():
    from learn_lang_chain.chapters.ch08_multimodal import build_human_message
    msg = build_human_message("这是什么？", "https://example.com/x.png")
    content = msg.content
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].endswith(".png")
