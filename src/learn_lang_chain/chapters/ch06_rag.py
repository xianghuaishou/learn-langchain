"""第六章：RAG（Retrieval‑Augmented Generation，检索增强生成）。

本章介绍 RAG —— 让聊天模型基于"你自己的数据"回答问题的标准范式。
流水线（pipeline）：

    加载文档  →  切分成小块（chunk）  →  向量化（embed）
        →  存入向量索引（vector index）
        →  查询时：把问题也向量化，检索最相似的 top‑k 个小块
        →  把这些小块作为上下文（context）注入 prompt，再交给模型回答。

之所以需要 RAG：模型本身的知识是训练时冻结的，无法回答私有/最新/垂直领域的问题。
RAG 用"检索"把外部知识临时塞进 prompt，相当于给模型一本可以随时翻阅的参考书。
"""

from pathlib import Path

# TextLoader：把纯文本文件读成 LangChain 的 Document 对象（每个 Document 带 page_content 和 metadata）。
from langchain_community.document_loaders import TextLoader
# FAISS：Meta 开源的内存向量数据库，做相似度搜索极快，无需额外启动任何外部服务。
from langchain_community.vectorstores import FAISS
# RecursiveCharacterTextSplitter：按字符递归切分长文档，是通用场景下推荐的默认切分器。
from langchain_text_splitters import RecursiveCharacterTextSplitter

from learn_lang_chain.config import get_chat_model

# 路径算术：__file__ 是 chapters/ch06_rag.py；parents[0]=chapters，parents[1]=learn_lang_chain，
# parents[2]=src，parents[3]=项目根目录 learn-lang-chain；再拼上 data/corpus.txt 得到语料文件。
CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus.txt"


def get_embeddings():
    """构造 Embeddings 客户端。

    这里复用 ``OpenAIEmbeddings``，但把 ``base_url`` 指向 MiniMax 的 API 主机：
    MiniMax 暴露了与 OpenAI 兼容的 ``/v1/embeddings`` 接口，所以无需单独的 SDK
    即可调用向量模型（请求体/响应体格式与 OpenAI 一致）。

    测试技巧：测试套件会用 ``monkeypatch`` 把本函数替换成确定性的 ``StubEmbeddings``，
    这样离线跑测试时不需要真实网络，也能得到稳定的向量结果。
    """
    from langchain_openai import OpenAIEmbeddings
    import os
    return OpenAIEmbeddings(
        model=os.environ.get("MINIMAX_EMBED_MODEL", "MiniMax-Embedding"),
        base_url=os.environ["MINIMAX_API_HOST"],
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def build_retriever():
    """构建检索器（retriever），分四步：

    1. 加载：用 ``TextLoader`` 把语料文件读成 ``Document`` 列表。
    2. 切分：用 ``RecursiveCharacterTextSplitter`` 切成小块，
       ``chunk_size=200`` 表示每块最多 200 个字符，
       ``chunk_overlap=20`` 表示相邻块重叠 20 个字符——
       重叠是为了让"切在句子中间"的情况尽量少发生，避免一句话的语义被截断。
    3. 向量化并建索引：用 ``get_embeddings()`` 把每个 chunk 转成向量，
       再用 ``FAISS.from_documents`` 在内存里建一个向量索引。
    4. 暴露检索器：``as_retriever(search_kwargs={"k": 2})`` 表示每次查询返回最相似的 2 个 chunk。
    """
    docs = TextLoader(str(CORPUS_PATH), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": 2})


def _format(docs) -> str:
    """把若干 ``Document`` 拼成一个字符串。

    检索器返回的是 ``List[Document]``，而 prompt 模板里 ``{context}`` 期望的是字符串；
    这个辅助函数把每个 ``Document`` 的 ``page_content`` 用空行连起来，
    形成适合塞进 prompt 的一段连续文本。
    """
    return "\n\n".join(d.page_content for d in docs)


def answer(question: str) -> str:
    """对用户问题生成基于语料库的答案。

    构造一个 LCEL（LangChain Expression Language）链：

        dict 构建器 | prompt | 模型 | 字符串解析器

    其中 dict 构建器：
        ``{"context": retriever | _format, "question": RunnablePassthrough()}``

    - ``retriever | _format``：用 ``|`` 把检索器和格式化函数串成一个 Runnable，
      给它传入问题字符串，它会先 ``retriever.invoke(question)`` 拿到 ``List[Document]``，
      再交给 ``_format`` 拼成单字符串，最终作为 ``context`` 的值。
    - ``RunnablePassthrough()``：原样把入参透传到 ``question`` 槽位，不做任何处理。

    整条链路的输入是问题字符串，输出是模型最终回答的字符串。
    """
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
    # 用 ``|`` 把 dict 构建器的输出（{context, question} 字典）
    # 接到 prompt：模板里的 {context} 和 {question} 占位符会自动按 key 取值。
    chain = (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | get_chat_model()
        | StrOutputParser()
    )
    # 这里传的是纯字符串，而不是 dict：
    # RunnablePassthrough() 会原样把入参传到 ``question`` 槽位，
    # 而 ``retriever | _format`` 会通过 ``.invoke(question)`` 消费这个字符串并产出 context。
    return chain.invoke(question)


if __name__ == "__main__":
    print(answer("什么是 LCEL？"))
