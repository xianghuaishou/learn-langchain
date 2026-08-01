"""第 8 章:多模态输入(文本 + 图片)。

本章演示如何向聊天模型发送**多模态输入**(文本 + 图片 URL)。
关键点:`HumanMessage.content` 不仅可以接收字符串,还可以接收一个**带类型的
内容块列表**,从而把图片和文本一起交给模型。
"""

# `HumanMessage.content` 的类型是 `Union[str, list[ContentBlock]]`。
# 对于支持视觉的模型,我们传入一个列表,里面放两个内容块:
#   - {"type": "text", "text": ...}      — 文本问题
#   - {"type": "image_url", "image_url": {"url": ...}}  — 图片 URL
from langchain_core.messages import HumanMessage

from learn_lang_chain.config import get_chat_model


def build_human_message(prompt: str, image_url: str) -> HumanMessage:
    """构造一条携带图片的用户消息。

    `content` 是一个内容块列表,包含两部分:

    - ``{"type": "text", "text": prompt}`` — 文字问题/指令。
    - ``{"type": "image_url", "image_url": {"url": image_url}}`` — 图片,
      以 URL 形式提供,由模型自行去拉取。
      (另一种写法是传入 ``data:`` URI,把图片的 base64 字节内联进来。)

    这个 ``image_url`` 内容块的格式与 OpenAI Chat Completions 的视觉 API
    保持一致(MiniMax-M3 也沿用同一套规范)。
    """
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )


def describe_image_url(image_url: str, prompt: str = "请用中文简要描述这张图片。") -> str:
    """让模型根据 ``image_url`` 和 ``prompt`` 输出一段描述。

    - ``prompt`` 带有默认值,方便 demo 直接调用;实际使用时可换成自定义问题。
    - 使用 ``model.invoke([msg])``:聊天 API 期望接收一个**消息列表**(此处只有一条用户消息)。
    - 取 ``.content`` 即可得到回复文本;无论是字符串还是列表形式的回复,
      SDK 都会归一化为字符串内容(对简单回复而言)。
    """
    # 模型必须支持视觉输入;MiniMax-M3 支持图像理解。
    model = get_chat_model()
    msg = build_human_message(prompt, image_url)
    return model.invoke([msg]).content


if __name__ == "__main__":
    # 维基共享资源上的一张公开图片(稳定的测试图)。
    print(describe_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/120px-Cat03.jpg"))
