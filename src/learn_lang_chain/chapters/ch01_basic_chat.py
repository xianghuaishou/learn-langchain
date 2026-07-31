from learn_lang_chain.config import get_chat_model


def ask(question: str) -> str:
    model = get_chat_model()
    result = model.invoke(question)
    return result.content


def stream_chunks(question: str):
    model = get_chat_model(streaming=True)
    for chunk in model.stream(question):
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    q = "用一句话介绍 LangChain。"
    print("[non-stream]", ask(q))
    print("[stream]    ", end="")
    for c in stream_chunks(q):
        print(c, end="", flush=True)
    print()