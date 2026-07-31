import pytest

@pytest.fixture
def fake_chat():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    return FakeListChatModel(responses=["fake-response"])
