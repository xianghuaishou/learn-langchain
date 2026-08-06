"""第 10 章：用 Langfuse 跟踪 LangChain 调用。

Langfuse 是开源的 LLM 可观测性平台，提供 trace 记录、prompt 版本管理、
评分、人工标注等能力。本章演示 Langfuse v3+ 的 LangChain 集成方式：

1. 通过环境变量配置 Langfuse 客户端（单例）。
2. 创建 Langfuse CallbackHandler。
3. 通过 LangChain 的 callbacks config 机制把 handler 附加到链调用。
4. 通过 metadata 传递 user_id / session_id / tags 等 trace 属性。
5. 短脚本结束时调用 ``flush()`` 把事件推到 Langfuse 后端。

需要环境变量：
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_BASE_URL（默认 https://cloud.langfuse.com）

参考：https://langfuse.com/integrations/frameworks/langchain
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from learn_lang_chain.config import get_chat_model


_DEFAULT_BASE_URL = "https://cloud.langfuse.com"


def setup_langfuse(
    public_key: str | None = None,
    secret_key: str | None = None,
    base_url: str | None = None,
):
    """初始化 Langfuse 客户端（单例）。

    优先使用参数值；缺省时从环境变量读：
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_BASE_URL（默认 https://cloud.langfuse.com）

    返回 Langfuse 客户端实例（来自 ``get_client()``），便于调用方
    调 ``flush()`` 或 ``shutdown()``。
    """
    Langfuse(
        public_key=public_key or os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=secret_key or os.environ["LANGFUSE_SECRET_KEY"],
        host=base_url or os.environ.get("LANGFUSE_BASE_URL", _DEFAULT_BASE_URL),
    )
    return get_client()


def build_chain():
    """构造一条简单链：prompt | model | parser。"""
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}。")
    return prompt | get_chat_model() | StrOutputParser()


def run_traced_demo(
    topic: str = "Langfuse",
    user_id: str = "demo-user",
    session_id: str = "demo-session",
    tags: list[str] | None = None,
) -> str:
    """运行一条链，把 trace 推到 Langfuse。

    通过 ``config.metadata`` 传递 Langfuse 专属字段：
    - langfuse_user_id
    - langfuse_session_id
    - langfuse_tags

    这些字段会被 CallbackHandler 读出来，写到 Langfuse 后端，方便
    按用户/会话/标签过滤 trace。
    """
    chain = build_chain()
    handler = CallbackHandler()
    metadata = {
        "langfuse_user_id": user_id,
        "langfuse_session_id": session_id,
        "langfuse_tags": tags or ["ch10-demo"],
    }

    result = chain.invoke(
        {"topic": topic},
        config={"callbacks": [handler], "metadata": metadata},
    )

    # 短生命周期脚本必须 flush，否则进程退出时事件可能丢失
    setup_langfuse().flush()
    return result


if __name__ == "__main__":
    setup_langfuse()
    print(run_traced_demo())
