"""Chapter 07 — 为聊天模型添加按会话隔离的对话记忆。

本章引入两个新组件：``ChatMessageHistory`` 是按 ``session_id`` 保存消息的
内存列表；``RunnableWithMessageHistory`` 则包装原有链，在每次调用前把对应
会话的历史消息注入提示模板，并在调用后保存本轮对话。
"""

# ``BaseChatMessageHistory`` 是定义消息历史行为契约的抽象基类，
# ``ChatMessageHistory`` 是最简单的内存实现；生产环境通常会换成由
# Redis、SQL 或 PostgreSQL 支持的子类，让历史记录能够持久化和跨进程共享。
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# ``MessagesPlaceholder`` 是提示模板中的特殊插槽：它渲染的是消息列表，
# 而不是像普通的 ``{variable}`` 插槽那样渲染一个字符串。
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# ``RunnableWithMessageHistory`` 是一个 Runnable 包装器：每次 ``.invoke()``
# 都会先取得当前会话的历史并注入提示，运行内部链，再把新一轮对话追加到历史。
from langchain_core.runnables.history import RunnableWithMessageHistory

from learn_lang_chain.config import get_chat_model


# 这是保存各会话历史的 session store，属于模块级状态。真实应用通常使用
# Redis 或数据库；这里为了保持示例简单，只用 Python 字典按 session_id 保存。
_store: dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> BaseChatMessageHistory:
    """延迟创建并返回指定会话的 ``ChatMessageHistory``。

    ``RunnableWithMessageHistory`` 每次调用时都会从 runnable 配置中取得
    ``session_id``，再调用本函数。因此，将来要改用 Redis、SQL 等持久化存储时，
    只需在这一个位置替换历史记录的获取方式。
    """
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def build_chain():
    """构建包含三类消息插槽的聊天链。

    system 消息设定助手的人设；``MessagesPlaceholder("history")`` 会由包装器
    填入过去的对话轮次；human 消息则接收本次用户输入。之后仍沿用第 04 章的
    ``prompt | model | parser`` 模式，将提示、模型和字符串解析器连接起来。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是友好的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    return prompt | get_chat_model() | StrOutputParser()


def _build_runnable():
    """构建负责读取、注入并更新会话历史的 Runnable 包装器。

    这里每次调用都重新构建包装器，使测试通过 ``monkeypatch`` 替换
    ``get_chat_model`` 后能够立即生效。``input_messages_key="input"`` 告诉
    包装器输入字典中的哪个键是要追加到历史的新用户消息；
    ``history_messages_key="history"`` 则告诉它应把历史消息注入提示模板中
    哪个 ``MessagesPlaceholder`` 对应的键。
    """
    return RunnableWithMessageHistory(
        build_chain(),
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat(text: str, session_id: str = "default") -> str:
    """在指定会话中发送消息并返回模型生成的文本。

    ``config={"configurable": {"session_id": session_id}}`` 是把会话 ID 传给
    ``RunnableWithMessageHistory`` 的约定方式。调用流程是：包装器接收输入，
    根据 session_id 查找历史，将历史注入提示里的 ``MessagesPlaceholder``，
    运行内部链，最后把本轮用户消息和模型回复追加到该会话的历史中。
    """
    return _build_runnable().invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    # 先在 demo 会话中介绍自己叫 Bob，紧接着询问“我叫什么？”。
    # 因为 demo 的历史中已经保存第一轮对话，模型应该能够记住名字。
    print(chat("我叫 Bob。", session_id="demo"))
    print(chat("我叫什么？", session_id="demo"))
