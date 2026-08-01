"""第一章：调用聊天模型的两种最简方式。

本示例演示了使用 LangChain 的 OpenAI 兼容客户端（``ChatOpenAI``）调用 MiniMax-M3 的两种最基础方式：

1. **同步调用** —— ``model.invoke(question)``：阻塞地一次性拿到完整回复。
2. **流式调用** —— ``model.stream(question)``：以生成器的方式逐块（chunk）接收 token，适合需要即时反馈的场景。

实际生产中往往还会配合提示词模板、输出解析器、工具调用等能力，但所有这些高级特性都建立在这两种调用方式之上。
"""

# 该工厂会读取 .env 中的 MINIMAX_API_HOST / MINIMAX_API_KEY / MINIMAX_MODEL，
# 校验 API Key 是否存在，并返回一个配置好的 ChatOpenAI 实例；
# 其 base_url 默认指向 https://api.minimaxi.com/v1，因此可以无缝对接 MiniMax-M3。
from learn_lang_chain.config import get_chat_model


def ask(question: str) -> str:
    """以同步方式向模型提问并返回完整的字符串回复。

    - ``get_chat_model()`` 返回一个 LangChain 的 ``BaseChatModel``（本例中是 ``ChatOpenAI``）。
    - ``.invoke(question)`` 是标准的阻塞式调用入口，会返回一条 ``AIMessage``（或其子类）。
    - ``result.content`` 即助手回复的纯文本内容。
    """
    model = get_chat_model()
    result = model.invoke(question)
    return result.content


def stream_chunks(question: str):
    """以流式方式逐块产出模型回复。

    - 这是一个 **生成器**（使用了 ``yield``），而不是一次性返回的列表，方便把每个 chunk 接入下一个组件（例如实时打印、写入流式接口）。
    - ``streaming=True`` 让底层 OpenAI 客户端使用 Server-Sent Events（SSE），chunk 以 ``AIMessageChunk`` 对象形式到达。
    - ``if chunk.content:`` 用于过滤掉“空 content”的中间 chunk，避免消费者打印出空行。
    """
    model = get_chat_model(streaming=True)
    for chunk in model.stream(question):
        # 真实的聊天补全接口在有意义 token 之间经常会产出 content 为空字符串的 chunk，过滤它们可以让输出更干净。
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    q = "用一句话介绍 LangChain。"
    print("[non-stream]", ask(q))
    # end="" 表示打印时不自动追加换行，从而让后续的流式输出紧跟在标签后面；
    # flush=True 则强制立即把缓冲区内容刷到终端，使回复真正“逐 token”地呈现给用户。
    print("[stream]    ", end="")
    for c in stream_chunks(q):
        print(c, end="", flush=True)
    print()