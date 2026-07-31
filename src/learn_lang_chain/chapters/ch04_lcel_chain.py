from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from learn_lang_chain.config import get_chat_model


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只用一个大写英文单词回答。"),
            ("human", "{topic}"),
        ]
    )
    return prompt | get_chat_model() | StrOutputParser()


def _run(topic: str) -> str:
    return build_chain().invoke({"topic": topic})


uppercase_topic = RunnableLambda(_run)


if __name__ == "__main__":
    print(uppercase_topic.invoke("Madrid"))
    decorated = build_chain() | RunnableLambda(lambda x: x[::-1])
    print(decorated.invoke({"topic": "Madrid"}))