"""第 09 章：用 OpenTelemetry 跟踪 LangChain 调用。

本章演示：
1. 初始化一个 OpenTelemetry TracerProvider 并挂上 span exporter。
2. 应用 LangchainInstrumentor，让所有 Runnable / tool / LLM 调用自动产出 span。
3. 运行一条链并通过 ConsoleSpanExporter 把 span 树打到 stdout。

OpenTelemetry 是云原生可观测性标准；span 可以导出到 Jaeger / Tempo /
Honeycomb / 自建 OTel Collector 等任意兼容后端。LangChain 通过
``opentelemetry-instrumentation-langchain`` 提供官方埋点，无需手写代码。
"""

# 标准 OpenTelemetry SDK 与导出器
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


# 模块级标志：记录是否曾经 instrument 过，便于后续判断是否需要拆旧 wrapper
_ever_instrumented = False


def setup_tracing(
    service_name: str = "learn-lang-chain",
    exporter=None,
) -> None:
    """初始化 OpenTelemetry TracerProvider 并自动埋点 LangChain。

    - ``service_name`` 写入每个 span 的 ``service.name`` resource 属性，便于
      在 Jaeger / Tempo 里按服务名过滤。
    - ``exporter`` 默认为 ``ConsoleSpanExporter``（把 span 打到 stdout），
      实战中可以替换成 ``OTLPSpanExporter`` 推到远端 collector。
    - 可重复调用：每次都会新建 TracerProvider 并重新 instrument；旧 wrapper
      会被手动拆除，避免其持有的 tracer 仍指向旧 provider。这对 Jupyter /
      交互式场景很有用（重复执行同一个 setup cell 不需要重启 kernel）。
    - 每次调用都新建一个 TracerProvider，便于不同测试拥有各自独立的 exporter。

    生产环境提示：本章为了可重复调用，覆盖了 OpenTelemetry 默认的
    "provider 只能 set 一次" 约束（``trace._TRACER_PROVIDER_SET_ONCE``）。
    在长跑服务里应在进程启动时调用一次 ``setup_tracing`` 即可，重复调用
    会丢已缓冲的 span 并触发 callback 重装。生产部署建议把 ``exporter``
    改为 ``OTLPSpanExporter`` 并配置 ``OTEL_EXPORTER_OTLP_ENDPOINT`` 环境变量。
    """
    global _ever_instrumented

    # OpenTelemetry 默认禁止覆盖已经设置的 TracerProvider，重置其内部 OnceSet
    # 让 ``trace.set_tracer_provider`` 可以落地新的 provider。
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
    )
    span_exporter = exporter if exporter is not None else ConsoleSpanExporter()
    # SimpleSpanProcessor：每条 span 结束时立即导出，便于学习时实时观察
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)

    if _ever_instrumented:
        _force_uninstrument_callback_manager()

    instrumentor = LangchainInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = False
    instrumentor.instrument(tracer_provider=provider)
    _ever_instrumented = True


def _force_uninstrument_callback_manager() -> None:
    """手动拆除 ``BaseCallbackManager.__init__`` 上残留的 OTel wrapper。

    背景：``LangchainInstrumentor._uninstrument()`` 在当前版本（0.62.x）下
    并未真正还原原函数——它依赖 ``opentelemetry.instrumentation.utils.unwrap``
    按 "module.attr" 路径解析，但实现里有个 rsplit 错误，导致对
    ``"langchain_core.callbacks.BaseCallbackManager.__init__"`` 这类路径
    静默 no-op。结果是同一份旧 wrapper 留在原地、它持有的 tracer 仍指向
    首次绑定的 provider，新 callback handler 永远装不上。

    这里直接读 ``__wrapped__``（wrapt 暴露的原函数引用）把它装回去。
    等上游修好后这段代码可以删除。
    """
    from langchain_core.callbacks import BaseCallbackManager
    cur = BaseCallbackManager.__init__
    if hasattr(cur, "__wrapped__"):
        BaseCallbackManager.__init__ = cur.__wrapped__  # type: ignore[attr-defined]


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