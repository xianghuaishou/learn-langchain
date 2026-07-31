import os
import pytest

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("MINIMAX_API_HOST", "MINIMAX_API_KEY", "MINIMAX_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    from learn_lang_chain.config import get_chat_model
    with pytest.raises(EnvironmentError):
        get_chat_model()


def test_missing_host_raises(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    from learn_lang_chain.config import get_chat_model
    with pytest.raises(EnvironmentError):
        get_chat_model()


def test_default_model_is_MiniMax_m3(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    sentinel = object()

    def fake_chat_openai(*, model, base_url, api_key, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "learn_lang_chain.config.ChatOpenAI", fake_chat_openai
    )
    from learn_lang_chain.config import get_chat_model
    assert get_chat_model() is sentinel


def test_env_model_override(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-X")
    captured = {}

    def fake_chat_openai(*, model, base_url, api_key, **kwargs):
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        "learn_lang_chain.config.ChatOpenAI", fake_chat_openai
    )
    from learn_lang_chain.config import get_chat_model
    get_chat_model()
    assert captured["model"] == "MiniMax-X"
