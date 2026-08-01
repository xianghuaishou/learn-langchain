from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from learn_lang_chain.config import get_chat_model

CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus.txt"


def get_embeddings():
    """MiniMax 兼容 OpenAI Embeddings；本地测试可通过 monkeypatch 替换。"""
    from langchain_openai import OpenAIEmbeddings
    import os
    return OpenAIEmbeddings(
        model=os.environ.get("MINIMAX_EMBED_MODEL", "MiniMax-Embedding"),
        base_url=os.environ["MINIMAX_API_HOST"],
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def build_retriever():
    docs = TextLoader(str(CORPUS_PATH), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": 2})


def _format(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def answer(question: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = build_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "仅依据以下上下文回答问题：\n{context}"),
            ("human", "{question}"),
        ]
    )
    chain = (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | get_chat_model()
        | StrOutputParser()
    )
    return chain.invoke(question)


if __name__ == "__main__":
    print(answer("什么是 LCEL？"))
