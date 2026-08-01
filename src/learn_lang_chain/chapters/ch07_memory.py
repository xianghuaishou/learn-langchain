from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from learn_lang_chain.config import get_chat_model


_store: dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是友好的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    return prompt | get_chat_model() | StrOutputParser()


def _build_runnable():
    return RunnableWithMessageHistory(
        build_chain(),
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat(text: str, session_id: str = "default") -> str:
    return _build_runnable().invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    print(chat("我叫 Bob。", session_id="demo"))
    print(chat("我叫什么？", session_id="demo"))
