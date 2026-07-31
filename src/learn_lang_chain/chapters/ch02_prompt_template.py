from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from learn_lang_chain.config import get_chat_model


def translate(text: str, target_lang: str) -> str:
    prompt = PromptTemplate.from_template(
        "把下面的文本翻译成{target_lang}：\n{text}\n只输出译文。"
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text, "target_lang": target_lang})


def persona_reply() -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位{role}，回答要符合角色设定。"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"role": "诗人", "question": "请用一句话介绍自己。"})


if __name__ == "__main__":
    print(translate("Hello, world!", "日语"))
    print(persona_reply())