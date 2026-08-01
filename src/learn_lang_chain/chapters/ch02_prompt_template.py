"""第 2 章：Prompt 模板与 LCEL 管道。

本章介绍 LangChain 中两种最常用的 Prompt 抽象：
- ``PromptTemplate``：单字符串模板，主要用于非聊天（completion）模型。
- ``ChatPromptTemplate``：按角色组织的消息列表，喂给聊天模型。

同时演示 LCEL 的 ``|``（管道）语法：
``prompt | model | parser``，每一段都是一个 ``Runnable``，
上一段的输出会自动作为下一段的输入。
"""

# PromptTemplate 是纯文本模板，渲染成一段字符串，
# 传统上配合 LLMChain / 老版本 LLM 使用；
# ChatPromptTemplate 则产出一组带角色的消息对象
# （SystemMessage / HumanMessage / AIMessage），专门给聊天模型用。
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 最简单的输出解析器：把模型返回的 AIMessage 拆开，只取出 .content 字段，
# 这样链的最终结果就是一个普通的 str。
from langchain_core.output_parsers import StrOutputParser

from learn_lang_chain.config import get_chat_model


def translate(text: str, target_lang: str) -> str:
    """用 PromptTemplate 构造翻译链并执行一次。

    ``PromptTemplate.from_template(...)`` 解析含 ``{var}`` 占位符的模板，
    在 ``invoke`` 时会把传入字典里的键值填回去。
    LCEL 中 ``prompt | model | parser`` 的含义是：每加一个 ``|``，
    就生成一个新的 ``Runnable``，其输入类型 = 上一阶段的输出类型，
    输出类型 = 本阶段的输出类型，从而把整条流水线串起来。
    最后 ``invoke({...})`` 把变量替换进模板，触发整条链运行。
    """
    prompt = PromptTemplate.from_template(
        "把下面的文本翻译成{target_lang}：\n{text}\n只输出译文。"
    )
    # 三段流水线：PromptTemplate(字符串) -> ChatModel(AIMessage) -> StrOutputParser(str)
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text, "target_lang": target_lang})


def persona_reply() -> str:
    """用 ChatPromptTemplate 构造带人设的角色回复。

    ``ChatPromptTemplate.from_messages([...])`` 接受一个 ``(role, content)`` 元组列表，
    支持的 role 包括：
    - ``system`` / ``human``(``user``) / ``ai``(``assistant``)。
    需要动态内容时，在 content 字符串里写 ``{var}`` 占位符即可。
    这里用 ``system`` 消息固定"角色设定"，因为 system 消息会被模型当作持久人设；
    而真正的提问放在 ``human`` 消息里，代表用户的实际请求。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位{role}，回答要符合角色设定。"),
            ("human", "{question}"),
        ]
    )
    # 三段流水线：ChatPromptTemplate(消息列表) -> ChatModel(AIMessage) -> StrOutputParser(str)
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"role": "诗人", "question": "请用一句话介绍自己。"})


if __name__ == "__main__":
    # 一次性把两种 Prompt 风格都跑一遍，便于直接对比输出差异。
    print(translate("Hello, world!", "日语"))
    print(persona_reply())