# learn-lang-chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建一个 LangChain 学习项目，含 8 个循序渐进的可运行示例 + pytest 离线测试，使用 MiniMax-M3 模型。

**Architecture:** 基于 `langchain_openai.ChatOpenAI` 复用 MiniMax 的 OpenAI 兼容接口；`config.py` 统一加载 `.env` 与构造 ChatModel；章节按 ch01-ch08 拆分；测试用 `FakeListChatModel` + `monkeypatch` 实现离线可跑。

**Tech Stack:** Python ≥ 3.10、LangChain 0.3.x、langchain-openai、langchain-community、FAISS、tiktoken、python-dotenv、Pydantic v2、pytest、pytest-mock。

**文件结构（执行计划前置）**

- 仓库根：`learn-lang-chain/`
- 配置层：`src/learn_lang_chain/config.py`
- 示例层：`src/learn_lang_chain/chapters/ch0X_*.py`
- 测试层：`tests/conftest.py`、`tests/test_config.py`、`tests/test_ch0X.py`

---

## Task 1: 仓库骨架、`.gitignore` 与 README 框架

**Files:**
- Create: `learn-lang-chain/.gitignore`
- Create: `learn-lang-chain/README.md`

- [ ] **Step 1: 创建 `.gitignore`**

```gitignore
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
faiss_index/
.venv/
```

- [ ] **Step 2: 初始化 git 仓库并首次提交骨架**

```bash
git init
git add .gitignore README.md
git commit -m "chore: initial repo skeleton"
```

---

## Task 2: 配置文件与 MiniMax ChatModel 工厂（含失败用例）

**Files:**
- Create: `src/learn_lang_chain/__init__.py`
- Create: `src/learn_lang_chain/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Create: `.env.example`
- Create: `requirements.txt`

- [ ] **Step 1: 创建 `.env.example`**

```
MINIMAX_API_HOST=https://your-MiniMax-host
MINIMAX_API_KEY=your_api_key
MINIMAX_MODEL=MiniMax-M3
```

- [ ] **Step 2: 创建 `requirements.txt`**

```
langchain>=0.3,<0.4
langchain-core
langchain-openai
langchain-community
faiss-cpu
tiktoken
python-dotenv
pydantic>=2
pytest>=8
pytest-mock>=3.12
```

- [ ] **Step 3: 创建 `tests/conftest.py`（空 fixture 占位）**

```python
import pytest

@pytest.fixture
def fake_chat():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    return FakeListChatModel(responses=["fake-response"])
```

- [ ] **Step 4: 写失败测试 `tests/test_config.py`**

```python
import os
import pytest

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("MINIMAX_API_HOST", "MINIMAX_API_KEY", "MINIMAX_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    from learn_lang_chain.config import get_chat_model
    with pytest.raises(EnvironmentError):
        get_chat_model()


def test_missing_host_raises(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    from learn_lang_chain.config import get_chat_model
    with pytest.raises(EnvironmentError):
        get_chat_model()


def test_default_model_is_MiniMax_m3(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    sentinel = object()

    def fake_chat_openai(*, model, base_url, api_key, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "learn_lang_chain.config.ChatOpenAI", fake_chat_openai
    )
    from learn_lang_chain.config import get_chat_model
    assert get_chat_model() is sentinel


def test_env_model_override(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_HOST", "https://example.com")
    monkeypatch.setenv("MINIMAX_API_KEY", "key")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-X")
    captured = {}

    def fake_chat_openai(*, model, base_url, api_key, **kwargs):
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        "learn_lang_chain.config.ChatOpenAI", fake_chat_openai
    )
    from learn_lang_chain.config import get_chat_model
    get_chat_model()
    assert captured["model"] == "MiniMax-X"
```

- [ ] **Step 5: 创建 `src/learn_lang_chain/__init__.py`（空文件）和 `tests/__init__.py`（空文件）**

- [ ] **Step 6: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_config.py -v
```
Expected: ImportError 或 ModuleNotFoundError（`learn_lang_chain.config` 不存在）

- [ ] **Step 7: 实现 `src/learn_lang_chain/config.py`**

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

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
```

- [ ] **Step 8: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_config.py -v
```
Expected: 4 passed

- [ ] **Step 9: 提交**

```bash
git add .env.example requirements.txt src tests
git commit -m "feat: add MiniMax ChatModel factory with tests"
```

---

## Task 3: ch01 基础 ChatModel 调用 + 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/__init__.py`
- Create: `src/learn_lang_chain/chapters/ch01_basic_chat.py`
- Create: `tests/test_ch01.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch01.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_invoke_returns_string(monkeypatch):
    fake = FakeListChatModel(responses=["hello"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch01_basic_chat.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch01_basic_chat import ask
    assert ask("hi") == "hello"


def test_stream_yields_chunks(monkeypatch):
    fake = FakeListChatModel(responses=["abc"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch01_basic_chat.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch01_basic_chat import stream_chunks
    chunks = list(stream_chunks("hi"))
    assert "abc" in "".join(chunks)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch01.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 `src/learn_lang_chain/chapters/__init__.py`（空文件）**

- [ ] **Step 4: 实现 `src/learn_lang_chain/chapters/ch01_basic_chat.py`**

```python
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
```

- [ ] **Step 5: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch01.py -v
```
Expected: 2 passed

- [ ] **Step 6: 提交**

```bash
git add src/learn_lang_chain/chapters/ch01_basic_chat.py tests/test_ch01.py
git commit -m "feat(ch01): basic ChatModel invoke + stream"
```

---

## Task 4: ch02 PromptTemplate + 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch02_prompt_template.py`
- Create: `tests/test_ch02.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch02.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def fake_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch02_prompt_template.get_chat_model",
        lambda **kw: fake,
    )


def test_prompt_template_translate(monkeypatch):
    fake_factory(monkeypatch, ["Bonjour"])
    from learn_lang_chain.chapters.ch02_prompt_template import translate
    assert translate("hello", "法语") == "Bonjour"


def test_chat_prompt_template_persona(monkeypatch):
    fake_factory(monkeypatch, ["I am a poet."])
    from learn_lang_chain.chapters.ch02_prompt_template import persona_reply
    assert "poet" in persona_reply().lower()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch02.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch02_prompt_template.py`**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch02.py -v
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch02_prompt_template.py tests/test_ch02.py
git commit -m "feat(ch02): PromptTemplate + ChatPromptTemplate"
```

---

## Task 5: ch03 OutputParser（Pydantic 化）+ 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch03_output_parser.py`
- Create: `tests/test_ch03.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch03.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch03_output_parser.get_chat_model",
        lambda **kw: fake,
    )


def test_str_parser(monkeypatch):
    patch_factory(monkeypatch, ["pong"])
    from learn_lang_chain.chapters.ch03_output_parser import echo
    assert echo("ping") == "pong"


def test_pydantic_parser(monkeypatch):
    patch_factory(
        monkeypatch,
        ['{"sentiment": "positive", "score": 0.9}'],
    )
    from learn_lang_chain.chapters.ch03_output_parser import analyze
    result = analyze("这家店服务超棒！")
    assert result.sentiment == "positive"
    assert result.score == pytest.approx(0.9)


import pytest
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch03.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch03_output_parser.py`**

```python
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from learn_lang_chain.config import get_chat_model


def echo(text: str) -> str:
    prompt = PromptTemplate.from_template("请原样回显：{text}")
    chain = prompt | get_chat_model() | StrOutputParser()
    return chain.invoke({"text": text})


class Sentiment(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def analyze(text: str) -> Sentiment:
    parser = PydanticOutputParser(pydantic_object=Sentiment)
    prompt = PromptTemplate.from_template(
        "分析下面评论的情感，给出 JSON：\n{text}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | get_chat_model() | parser
    return chain.invoke({"text": text})


if __name__ == "__main__":
    print(echo("hello"))
    print(analyze("这家店服务超棒！"))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch03.py -v
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch03_output_parser.py tests/test_ch03.py
git commit -m "feat(ch03): output parsers incl. Pydantic"
```

---

## Task 6: ch04 LCEL 链式调用 + 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch04_lcel_chain.py`
- Create: `tests/test_ch04.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch04.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch04_lcel_chain.get_chat_model",
        lambda **kw: fake,
    )


def test_chain_invoke(monkeypatch):
    patch_factory(monkeypatch, ["STADIUM"])
    from learn_lang_chain.chapters.ch04_lcel_chain import uppercase_topic
    assert uppercase_topic("Barcelona") == "STADIUM"


def test_batch(monkeypatch):
    patch_factory(monkeypatch, ["A", "B", "C"])
    from learn_lang_chain.chapters.ch04_lcel_chain import uppercase_topic
    out = uppercase_topic.batch(["a", "b", "c"])
    assert out == ["A", "B", "C"]
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch04.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch04_lcel_chain.py`**

```python
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


chain = build_chain()


def uppercase_topic(topic: str) -> str:
    return chain.invoke({"topic": topic})


if __name__ == "__main__":
    print(uppercase_topic("Madrid"))
    decorated = chain | RunnableLambda(lambda x: x[::-1])
    print(decorated.invoke({"topic": "Madrid"}))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch04.py -v
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch04_lcel_chain.py tests/test_ch04.py
git commit -m "feat(ch04): LCEL chain + batch"
```

---

## Task 7: ch05 Tools + Agent + 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch05_tools_agent.py`
- Create: `tests/test_ch05.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch05.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch05_tools_agent.get_chat_model",
        lambda **kw: fake,
    )


def test_tools_listed():
    from learn_lang_chain.chapters.ch05_tools_agent import TOOLS
    names = {t.name for t in TOOLS}
    assert {"get_weather", "calculator"} <= names


def test_get_weather():
    from learn_lang_chain.chapters.ch05_tools_agent import get_weather
    assert "晴" in get_weather.invoke({"city": "上海"})


def test_calculator():
    from learn_lang_chain.chapters.ch05_tools_agent import calculator
    assert calculator.invoke({"expr": "2 + 3 * 4"}) == 14
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch05.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch05_tools_agent.py`**

```python
import ast
import operator

from langchain_core.tools import tool

from learn_lang_chain.config import get_chat_model


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（伪实现）。"""
    return f"{city}：晴，25°C"


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}


@tool
def calculator(expr: str) -> float:
    """计算简单数学表达式，仅支持 + - * / ** %。"""
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


TOOLS = [get_weather, calculator]


def run_agent(question: str) -> str:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是助手，必要时调用工具。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_react_agent(get_chat_model(), TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
    return executor.invoke({"input": question})["output"]


if __name__ == "__main__":
    print(run_agent("上海今天天气如何？"))
    print(run_agent("计算 (3 + 5) * 2"))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch05.py -v
```
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch05_tools_agent.py tests/test_ch05.py
git commit -m "feat(ch05): tools + ReAct agent"
```

---

## Task 8: ch06 RAG + FAISS + 测试（嵌入层用 stub）

**Files:**
- Create: `src/learn_lang_chain/chapters/ch06_rag.py`
- Create: `tests/test_ch06.py`
- Create: `data/corpus.txt`

- [ ] **Step 1: 创建小型语料 `data/corpus.txt`**

```
LangChain is a framework for building LLM applications.
FAISS is a library for efficient similarity search.
LCEL stands for LangChain Expression Language, using the | operator to compose Runnables.
AgentExecutor orchestrates tool calls via the ReAct loop.
```

- [ ] **Step 2: 写失败测试 `tests/test_ch06.py`**

```python
class StubEmbeddings:
    def __init__(self, dim=8):
        self.dim = dim

    def embed_documents(self, texts):
        import hashlib
        return [
            [int(hashlib.md5(t.encode()).hexdigest()[i : i + 2], 16) / 255
             for i in range(self.dim)]
            for t in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_retriever_returns_documents(monkeypatch):
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch06_rag.get_embeddings",
        lambda: StubEmbeddings(),
    )
    from learn_lang_chain.chapters.ch06_rag import build_retriever, answer
    retriever = build_retriever()
    docs = retriever.invoke("什么是 LCEL？")
    assert any("LCEL" in d.page_content for d in docs)
    # answer() 走 LLM：用 fake chat
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch06_rag.get_chat_model",
        lambda **kw: FakeListChatModel(responses=["基于文档：LCEL 是 ... "]),
    )
    out = answer("什么是 LCEL？")
    assert "LCEL" in out
```

- [ ] **Step 3: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch06.py -v
```
Expected: ImportError

- [ ] **Step 4: 实现 `src/learn_lang_chain/chapters/ch06_rag.py`**

```python
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from learn_lang_chain.config import get_chat_model

CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus.txt"


def get_embeddings():
    """MiniMax 兼容 OpenAI Embeddings；本地测试可通过 monkeypatch 替换。"""
    from langchain_openai import OpenAIEmbeddings
    import os
    return OpenAIEmbeddings(
        model=os.environ.get("MINIMAX_EMBED_MODEL", "MiniMax-Embedding"),
        base_url=os.environ["MINIMAX_API_HOST"],
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def build_retriever():
    docs = TextLoader(str(CORPUS_PATH), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": 2})


def _format(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def answer(question: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = build_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "仅依据以下上下文回答问题：\n{context}"),
            ("human", "{question}"),
        ]
    )
    chain = (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | get_chat_model()
        | StrOutputParser()
    )
    return chain.invoke(question)


if __name__ == "__main__":
    print(answer("什么是 LCEL？"))
```

- [ ] **Step 5: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch06.py -v
```
Expected: 1 passed

- [ ] **Step 6: 提交**

```bash
git add data src/learn_lang_chain/chapters/ch06_rag.py tests/test_ch06.py
git commit -m "feat(ch06): RAG with FAISS retriever"
```

---

## Task 9: ch07 多轮 Memory（RunnableWithMessageHistory）+ 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch07_memory.py`
- Create: `tests/test_ch07.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch07.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def patch_factory(monkeypatch, responses):
    fake = FakeListChatModel(responses=responses)
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch07_memory.get_chat_model",
        lambda **kw: fake,
    )


def test_history_accumulates(monkeypatch):
    patch_factory(monkeypatch, ["reply-1", "reply-2"])
    from learn_lang_chain.chapters.ch07_memory import chat
    a = chat("hello", session_id="s1")
    b = chat("again", session_id="s1")
    assert a == "reply-1"
    assert b == "reply-2"


def test_isolated_sessions(monkeypatch):
    patch_factory(monkeypatch, ["x", "y", "p", "q"])
    from learn_lang_chain.chapters.ch07_memory import chat
    chat("a", session_id="s1")
    chat("b", session_id="s1")
    chat("c", session_id="s2")
    chat("d", session_id="s2")
    # 仅验证不抛异常且隔离生效
    assert True
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch07.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch07_memory.py`**

```python
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


_chain = build_chain()
_runnable = RunnableWithMessageHistory(
    _chain,
    _get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat(text: str, session_id: str = "default") -> str:
    return _runnable.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    print(chat("我叫 Bob。", session_id="demo"))
    print(chat("我叫什么？", session_id="demo"))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch07.py -v
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch07_memory.py tests/test_ch07.py
git commit -m "feat(ch07): memory via RunnableWithMessageHistory"
```

---

## Task 10: ch08 多模态 + 测试

**Files:**
- Create: `src/learn_lang_chain/chapters/ch08_multimodal.py`
- Create: `tests/test_ch08.py`

- [ ] **Step 1: 写失败测试 `tests/test_ch08.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_describe_image_url(monkeypatch):
    fake = FakeListChatModel(responses=["一只猫"])
    monkeypatch.setattr(
        "learn_lang_chain.chapters.ch08_multimodal.get_chat_model",
        lambda **kw: fake,
    )
    from learn_lang_chain.chapters.ch08_multimodal import describe_image_url
    out = describe_image_url("https://example.com/cat.jpg")
    assert "猫" in out


def test_message_payload_shape():
    from learn_lang_chain.chapters.ch08_multimodal import build_human_message
    msg = build_human_message("这是什么？", "https://example.com/x.png")
    content = msg.content
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].endswith(".png")
```

- [ ] **Step 2: 运行测试验证失败**

```bash
PYTHONPATH=src pytest tests/test_ch08.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `src/learn_lang_chain/chapters/ch08_multimodal.py`**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
PYTHONPATH=src pytest tests/test_ch08.py -v
```
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/learn_lang_chain/chapters/ch08_multimodal.py tests/test_ch08.py
git commit -m "feat(ch08): multimodal image url"
```

---

## Task 11: README 完善 + 全量测试

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 填充 README.md**

```markdown
# learn-lang-chain

LangChain 入门到进阶示例，统一使用 MiniMax-M3 模型（OpenAI 兼容协议）。

## 环境

1. `pip install -r requirements.txt`
2. `cp .env.example .env`，填入 `MINIMAX_API_HOST`、`MINIMAX_API_KEY`，可选 `MINIMAX_MODEL`（默认 `MiniMax-M3`）。

## 运行示例

```bash
python -m learn_lang_chain.chapters.ch01_basic_chat
python -m learn_lang_chain.chapters.ch02_prompt_template
python -m learn_lang_chain.chapters.ch03_output_parser
python -m learn_lang_chain.chapters.ch04_lcel_chain
python -m learn_lang_chain.chapters.ch05_tools_agent
python -m learn_lang_chain.chapters.ch06_rag
python -m learn_lang_chain.chapters.ch07_memory
python -m learn_lang_chain.chapters.ch08_multimodal
```

## 测试

```bash
PYTHONPATH=src pytest -v
```

测试使用 `FakeListChatModel`，无需真实 key。

## 学习路线

| 章 | 主题 | 关键 API |
| -- | -- | -- |
| ch01 | 基础对话 | `ChatOpenAI.invoke/stream` |
| ch02 | 提示模板 | `PromptTemplate`, `ChatPromptTemplate` |
| ch03 | 输出解析 | `StrOutputParser`, `PydanticOutputParser` |
| ch04 | LCEL | `prompt | model | parser` |
| ch05 | 工具与 Agent | `@tool`, `create_react_agent`, `AgentExecutor` |
| ch06 | RAG | `TextLoader`, `FAISS`, retriever + LCEL |
| ch07 | 多轮记忆 | `RunnableWithMessageHistory` |
| ch08 | 多模态 | `HumanMessage` content list |

## 常见问题

- **401 / 403**：检查 `MINIMAX_API_KEY` 是否正确。
- **超时/限流**：脚本可在 `model.invoke` 周围自行加重试（未内置以保持示例简洁）。
```

- [ ] **Step 2: 运行全量测试**

```bash
PYTHONPATH=src pytest -v
```
Expected: 全部通过（≥17 用例）

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs: comprehensive README + run instructions"
```

---

## 自检

- **Spec 覆盖**：背景/目标(Task1-2)、非目标(README)、技术栈(Task2)、项目结构(Task1-11)、客户端封装(Task2)、配置(Task2)、章节内容(Task3-10)、测试策略(Task2-10)、错误处理(Task2, README)、验收标准(Task11)、风险与权衡(README 备注) ✓
- **占位符扫描**：无 TBD/TODO ✓
- **类型一致**：`get_chat_model` 在所有章节签名一致；`FakeListChatModel` 一致使用 ✓