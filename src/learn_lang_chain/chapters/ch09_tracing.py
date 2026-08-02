"""第 09 章：用 OpenTelemetry 跟踪 LangChain 调用。

本章演示：
1. 初始化一个 OpenTelemetry TracerProvider 并挂上 span exporter。
2. 应用 LangchainInstrumentor，让所有 Runnable / tool / LLM 调用自动产出 span。
3. 运行一条链并通过 ConsoleSpanExporter 把 span 树打到 stdout。

OpenTelemetry 是云原生可观测性标准；span 可以导出到 Jaeger / Tempo /
Honeycomb / 自建 OTel Collector 等任意兼容后端。LangChain 通过
``opentelemetry-instrumentation-langchain`` 提供官方埋点，无需手写代码。
"""

from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from learn_lang_chain.config import get_chat_model


_instrumented = False


def setup_tracing(
    service_name: str = "learn-lang-chain",
    exporter=None,
) -> None:
    """初始化 OpenTelemetry TracerProvider 并自动埋点 LangChain。

    - ``service_name`` 写入每个 span 的 ``service.name`` resource 属性，便于
      在 Jaeger / Tempo 里按服务名过滤。
    - ``exporter`` 默认为 ``ConsoleSpanExporter``（把 span 打到 stdout），
      实战中可以替换成 ``OTLPSpanExporter`` 推到远端 collector。
    - 幂等：重复调用不会重复 patch LangChain（通过 ``_instrumented`` 标志）。
    - 每次调用都会新建一个 TracerProvider，便于测试用各自的 exporter。
    """
    global _instrumented

    # OpenTelemetry 默认禁止覆盖已经设置的 TracerProvider，重置其内部 OnceSet
    # 让 ``trace.set_tracer_provider`` 可以落地新的 provider。
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
    )
    span_exporter = exporter if exporter is not None else ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)

    if _instrumented:
        # ``LangchainInstrumentor._uninstrument`` 在新版本 wrapt + 路径格式
        # 下实际上没有拆掉 ``BaseCallbackManager.__init__`` 上的 wrapper：
        # ``opentelemetry.instrumentation.utils.unwrap`` 解析错误的目标字符串
        # 时会静默跳过，结果是同一份旧 wrapper 留在原地、它持有的 tracer 仍
        # 指向首次绑定的 provider，新的 callback handler 永远装不上。
        # 这里手动把 ``__init__`` 还原成原函数，再让 ``instrument`` 重新装新版。
        from langchain_core.callbacks import BaseCallbackManager
        cur = BaseCallbackManager.__init__
        if hasattr(cur, "__wrapped__"):
            BaseCallbackManager.__init__ = cur.__wrapped__  # type: ignore[attr-defined]

    instrumentor = LangchainInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = False
    instrumentor.instrument(tracer_provider=provider)
    _instrumented = True


def run_traced_demo() -> str:
    """构造一条简单链并执行；调用会被自动埋点为 span。

    返回模型回复的字符串。生产中可以把这条链换成 ch05 的 agent，
    这样整个 ReAct 循环都会被记录到 span 树里。
    """
    chain = (
        ChatPromptTemplate.from_template("用一句话介绍{topic}。")
        | get_chat_model()
        | StrOutputParser()
    )
    return chain.invoke({"topic": "OpenTelemetry"})


if __name__ == "__main__":
    setup_tracing("ch09-demo")
    print(run_traced_demo())
