from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from learn_lang_chain.config import get_chat_model


def echo(text: str) -> str:
    prompt = PromptTemplate.from_template("请原样回显：{text}")
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text})


class Sentiment(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def analyze(text: str) -> Sentiment:
    parser = PydanticOutputParser(pydantic_object=Sentiment)
    prompt = PromptTemplate.from_template(
        "分析下面评论的情感，给出 JSON：\n{text}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | get_chat_model() | parser
    return chain.invoke({"text": text})


if __name__ == "__main__":
    print(echo("hello"))
    print(analyze("这家店服务超棒！"))
