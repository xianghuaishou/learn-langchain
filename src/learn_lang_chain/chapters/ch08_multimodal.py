from langchain_core.messages import HumanMessage

from learn_lang_chain.config import get_chat_model


def build_human_message(prompt: str, image_url: str) -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )


def describe_image_url(image_url: str, prompt: str = "请用中文简要描述这张图片。") -> str:
    model = get_chat_model()
    msg = build_human_message(prompt, image_url)
    return model.invoke([msg]).content


if __name__ == "__main__":
    print(describe_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/120px-Cat03.jpg"))
