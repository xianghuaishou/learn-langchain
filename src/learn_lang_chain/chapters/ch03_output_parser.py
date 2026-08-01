"""第 03 章：OutputParser（输出解析器）。

本章介绍 LangChain 中的 ``OutputParser`` —— 它们的作用是把模型原始的
``AIMessage``（通常是一段文本）转成更便于程序使用的 Python 类型。

本章演示两种最常见的解析器：

* ``StrOutputParser`` —— 最朴素的解析器，只把消息体的字符串取出来。
* ``PydanticOutputParser`` —— 让模型按指定 JSON 结构输出，并自动校验、
  解析成一个 :class:`pydantic.BaseModel` 实例。
"""

# Literal 用来把字段限制为一组固定的取值，适合做"枚举"。
from typing import Literal

# PydanticOutputParser、StrOutputParser 都是 Runnable，可以放在 LCEL 管道的最后一环。
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

# LangChain 大量复用 Pydantic 来描述结构化字段；Field(ge=..., le=...) 可以约束数值范围。
from pydantic import BaseModel, Field

from learn_lang_chain.config import get_chat_model


def echo(text: str) -> str:
    """``StrOutputParser`` 的最小示例：让模型把输入原样回显，再把 ``AIMessage`` 解析成字符串。"""
    prompt = PromptTemplate.from_template("请原样回显：{text}")
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text})


class Sentiment(BaseModel):
    """情感分析结果的"目标结构"：当解析器成功时，最终返回的就是本模型的一个实例。

    字段说明：

    * ``sentiment: Literal["positive", "neutral", "negative"]``
      把取值限定在三个枚举值之中；模型一旦给出其它内容，Pydantic 会抛错。
    * ``score: float = Field(ge=0.0, le=1.0)``
      限定分数必须在 ``[0.0, 1.0]`` 区间内，越界会被 Pydantic 校验失败。
    * ``reason: str = ""``
      提供一个默认空字符串，模型省略该字段时也不会报错。
    """

    sentiment: Literal["positive", "neutral", "negative"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def analyze(text: str) -> Sentiment:
    """用 ``PydanticOutputParser`` 让模型按照 :class:`Sentiment` 的结构输出 JSON。"""
    parser = PydanticOutputParser(pydantic_object=Sentiment)  # 读取模型输出，并解析成 Sentiment 实例。
    prompt = PromptTemplate.from_template(
        "分析下面评论的情感，给出 JSON：\n{text}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    # parser.get_format_instructions() 返回一段说明，告诉模型"该输出怎样的 JSON（字段名、类型、约束）"。
    # PromptTemplate.partial(...) 预先把 format_instructions 填进模板变量，运行时只需再提供 text。
    chain = prompt | get_chat_model() | parser  # 管道最后换成了 PydanticOutputParser，因此 chain.invoke(...) 返回的是 Sentiment 对象而不是字符串。
    return chain.invoke({"text": text})


if __name__ == "__main__":
    print(echo("hello"))
    # analyze(...) 的返回类型是 Sentiment；这里打印时会调用 Pydantic 自动生成的 __repr__，例如 Sentiment(sentiment='positive', score=0.95, reason='...')。
    print(analyze("这家店服务超棒！"))
