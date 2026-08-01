from langchain_core.embeddings import Embeddings


class StubEmbeddings(Embeddings):
    def __init__(self, dim=8):
        self.dim = dim

    def embed_documents(self, texts):
        import hashlib
        return [
            [int(hashlib.md5(t.encode()).hexdigest()[i : i + 2], 16) / 255
             for i in range(self.dim)]
            for t in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_retriever_returns_documents(monkeypatch):
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch06_rag.get_embeddings",
        lambda: StubEmbeddings(),
    )
    from learn_lang_chain.chapters.ch06_rag import build_retriever, answer
    retriever = build_retriever()
    docs = retriever.invoke("什么是 LCEL？")
    assert any("LCEL" in d.page_content for d in docs)
    # answer() 走 LLM：用 fake chat
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch06_rag.get_chat_model",
        lambda **kw: FakeListChatModel(responses=["基于文档：LCEL 是 ... "]),
    )
    out = answer("什么是 LCEL？")
    assert "LCEL" in out
