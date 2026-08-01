import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

DEFAULT_MODEL = "MiniMax-M3"


def get_chat_model(streaming: bool = False, temperature: float = 0.7) -> ChatOpenAI:
    host = os.environ.get("MINIMAX_API_HOST")
    key = os.environ.get("MINIMAX_API_KEY")
    model = os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL)
    if not key:
        raise EnvironmentError(
            "MINIMAX_API_KEY 未配置，请复制 .env.example 为 .env 并填写"
        )
    if not host:
        raise EnvironmentError("MINIMAX_API_HOST 未配置")
    return ChatOpenAI(
        model=model,
        base_url=host,
        api_key=key,
        temperature=temperature,
        streaming=streaming,
    )
