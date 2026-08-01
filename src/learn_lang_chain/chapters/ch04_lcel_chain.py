"""Chapter 04 — LCEL (LangChain Expression Language) 链式编程。

本章是 LCEL 的核心。LangChain 中的几乎所有组件（``ChatPromptTemplate``、
``ChatModel``、``StrOutputParser``、``RunnableLambda``、``Retriever``……）
都实现了统一的 ``Runnable`` 协议，这意味着它们都可以使用相同的方式调用：

* ``.invoke(input)`` —— 同步调用一次
* ``.stream(input)``  —— 流式返回
* ``.batch([inputs])`` —— 批量调用
* ``.ainvoke(input)`` —— 异步版本

更重要的是，``Runnable`` 之间可以通过 ``|`` 运算符组合成一个新的
``Runnable``，从而像搭积木一样把提示模板、模型、解析器、自定义函数等
串成一个完整的链。本章的所有示例都会围绕这个核心思想展开。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ``RunnableLambda`` 可以把任意 Python 可调用对象（普通函数或 lambda）
# 包装成一个 ``Runnable``，从而能够通过 ``|`` 接入到 LCEL 链中，
# 也可以像其他 Runnable 一样调用 ``.invoke()`` / ``.batch()`` / ``.stream()``。
from langchain_core.runnables import RunnableLambda

from learn_lang_chain.config import get_chat_model


def build_chain():
    """构建并返回最经典的 3 阶段 LCEL 链：``prompt | model | parser``。

    三个阶段的输入/输出类型依次为：

    * ``prompt`` (``ChatPromptTemplate``)：接收 ``dict``，产出一组 ``Message`` 列表
    * ``model`` (``BaseChatModel``)：消费消息列表，返回一条 ``AIMessage``
    * ``parser`` (``StrOutputParser``)：从 ``AIMessage`` 中提取纯文本，返回 ``str``

    由于每一步的输出恰好是下一步要求的输入，整条链可以用 ``|`` 无缝连接，
    最终对外暴露的仍然是一个 ``Runnable``，因此支持 ``invoke/stream/batch``。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只用一个大写英文单词回答。"),
            ("human", "{topic}"),
        ]
    )
    return prompt | get_chat_model() | StrOutputParser()


def _run(topic: str) -> str:
    """真正执行一次 ``build_chain`` 的内部函数。

    这里有两条设计上的考量：

    1. 为什么把整条链包成 ``uppercase_topic = RunnableLambda(_run)``，
       而不是一个普通的 ``def uppercase_topic(topic): ...``？
       因为 ``RunnableLambda`` 会把可调用对象升级为 ``Runnable``，使得它
       也支持 ``.invoke()``、``.batch()``、``.stream()``，并且可以继续用
       ``|`` 与其他 ``Runnable`` 组合，从而保持与 LangChain 其他组件一致
       的接口风格。

    2. 为什么把 ``build_chain()`` 的调用放在函数体内，而不是在模块顶层
       ``CHAIN = build_chain()`` 提前构建？
       因为测试会通过 ``monkeypatch`` 把 ``get_chat_model`` 替换成假模型，
       而模块顶层只会在 ``import`` 时求值一次，模型引用会被"冻结"。
       改为每次调用时再 ``build_chain()``，可以让 ``monkeypatch`` 生效。
    """
    return build_chain().invoke({"topic": topic})


# 把上面的 ``_run`` 包成 ``Runnable``，对外暴露一个名叫 ``uppercase_topic``
# 的可调用链式组件，便于复用与组合。
uppercase_topic = RunnableLambda(_run)


if __name__ == "__main__":
    print(uppercase_topic.invoke("Madrid"))
    # 把现有链再 ``|`` 一个 ``RunnableLambda``，组成一条新的链：
    # ``build_chain()`` 的输出是 ``str``，会作为输入传给后面的 lambda，
    # 最终效果就是返回把字符串反转的结果。
    # 这展示了 LCEL 的另一个核心特性：``|`` 可以把任意 ``Runnable``
    # （包括用户自定义的 ``RunnableLambda``）拼接到任何链的末尾。
    decorated = build_chain() | RunnableLambda(lambda x: x[::-1])
    # 注意：这里的入参是 ``dict``，但传给 ``RunnableLambda`` 的 ``x`` 却是
    # ``str``——这是因为 LCEL 的 ``|`` 会把前一个 Runnable 的输出原样传给
    # 后一个 Runnable，而不是再回传原始输入。
    print(decorated.invoke({"topic": "Madrid"}))