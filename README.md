# learn-lang-chain

LangChain 入门到进阶示例，统一使用 MiniMax-M3（OpenAI 兼容协议）。

## 1. 项目结构

```
learn-lang-chain/
├── .env.example                # 环境变量模板
├── requirements.txt
├── data/corpus.txt             # ch06 RAG 示例语料
├── src/learn_lang_chain/
│   ├── config.py               # 环境加载 + ChatModel 工厂
│   └── chapters/
│       ├── ch01_basic_chat.py
│       ├── ch02_prompt_template.py
│       ├── ch03_output_parser.py
│       ├── ch04_lcel_chain.py
│       ├── ch05_tools_agent.py
│       ├── ch06_rag.py
│       ├── ch07_memory.py
│       └── ch08_multimodal.py
└── tests/
    ├── conftest.py
    ├── test_config.py
    └── test_ch01.py … test_ch08.py
```

## 2. 环境准备

> 项目根目录已带 PyCharm 创建的 `.venv/`。本项目使用该虚拟环境。
>
> **Python 版本要求**：3.10 ≤ Python < 3.14。Python 3.14 上 `langchain==0.3.x` 与 `pydantic>=2` 存在已知 `_eval_type` 兼容问题（ch05 的 `run_agent` 触发），但所有测试仍可正常通过。

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖 + 把 learn_lang_chain 安装为 editable 包
# （editable 安装后，python -m 无需再设 PYTHONPATH）
pip install -r requirements.txt
pip install -e .

# 复制环境变量模板并填入真实 key
cp .env.example .env
# 编辑 .env，至少设置：
#   MINIMAX_API_HOST=https://api.minimaxi.com/v1   # 必须带 /v1，由 SDK 自动补 /chat/completions
#   MINIMAX_API_KEY=你的-key
#   MINIMAX_MODEL=MiniMax-M3                       # 可选，默认即为 MiniMax-M3
```

`.env` 中的配置会**覆盖**同名的 shell 环境变量（`config.py` 用 `load_dotenv(override=True)`）。如要临时使用 shell 值而非 `.env`，可在命令前加 `MINIMAX_API_HOST=...`。

如果系统中没有 `.venv/`，可用系统 Python：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3. 运行示例（chapters 目录）

每个章节文件都带有 `if __name__ == "__main__":` 入口，可直接通过 `python -m` 运行：

```bash
source .venv/bin/activate

python -m learn_lang_chain.chapters.ch01_basic_chat       # 基础对话 + 流式
python -m learn_lang_chain.chapters.ch02_prompt_template  # PromptTemplate + ChatPromptTemplate
python -m learn_lang_chain.chapters.ch03_output_parser    # StrOutputParser + PydanticOutputParser
python -m learn_lang_chain.chapters.ch04_lcel_chain       # LCEL pipe + RunnableLambda + batch
python -m learn_lang_chain.chapters.ch05_tools_agent      # @tool + ReAct Agent（AgentExecutor）
python -m learn_lang_chain.chapters.ch06_rag              # TextLoader + FAISS + Retriever
python -m learn_lang_chain.chapters.ch07_memory           # RunnableWithMessageHistory 多轮记忆
python -m learn_lang_chain.chapters.ch08_multimodal       # HumanMessage content list 多模态
```

PyCharm 用户也可以直接右键章节文件 → "Run"。

## 4. 运行测试

测试使用 `FakeListChatModel` + `monkeypatch`，**无需真实 API key**，所有用例离线可跑。

```bash
source .venv/bin/activate

# 跑全部测试（pyproject.toml 已配置 pythonpath = ["src"]）
.venv/bin/pytest tests/ -v

# 跑单个章节
.venv/bin/pytest tests/test_ch05.py -v

# 跑单个用例
.venv/bin/pytest tests/test_ch05.py::test_calculator -v
```

当前测试矩阵：

| 测试文件 | 用例数 |
| --- | --- |
| `tests/test_config.py` | 4 |
| `tests/test_ch01.py` | 2 |
| `tests/test_ch02.py` | 2 |
| `tests/test_ch03.py` | 2 |
| `tests/test_ch04.py` | 2 |
| `tests/test_ch05.py` | 3 |
| `tests/test_ch06.py` | 1 |
| `tests/test_ch07.py` | 2 |
| `tests/test_ch08.py` | 2 |
| **合计** | **20** |

## 5. 学习路线

| 章 | 主题 | 关键 API |
| -- | -- | -- |
| ch01 | 基础对话 | `ChatOpenAI.invoke/stream` |
| ch02 | 提示模板 | `PromptTemplate`, `ChatPromptTemplate` |
| ch03 | 输出解析 | `StrOutputParser`, `PydanticOutputParser` |
| ch04 | LCEL | `prompt \| model \| parser`, `RunnableLambda`, `.batch()` |
| ch05 | 工具与 Agent | `@tool`, `create_react_agent`, `AgentExecutor` |
| ch06 | RAG | `TextLoader`, `RecursiveCharacterTextSplitter`, `FAISS`, `Retriever` |
| ch07 | 多轮记忆 | `RunnableWithMessageHistory`, `ChatMessageHistory` |
| ch08 | 多模态 | `HumanMessage` content list（含 `image_url`） |

## 6. 常见问题

- **404 / 401 / 403**：检查 `.env` 中 `MINIMAX_API_HOST` 是否带 `/v1` 后缀（如 `https://api.minimaxi.com/v1`），`MINIMAX_API_KEY` 是否正确、是否带多余空格或换行。SDK 会自动在 host 后拼 `/chat/completions`，因此 host 不要写完整 endpoint。
- **超时 / 限流**：示例未内置重试，可在 `get_chat_model()` 返回的实例上自行设置 `max_retries`。
- **`ModuleNotFoundError: learn_lang_chain`**：未执行 `pip install -e .`，或没在项目根目录运行。
- **`EnvironmentError: MINIMAX_API_KEY 未配置`**：`.env` 不存在或 key 为空；先 `cp .env.example .env` 再填写。
- **测试导入 `langchain` 失败**：未激活 venv 或未安装依赖；先 `source .venv/bin/activate && pip install -r requirements.txt`。
- **shell 环境变量优先级问题**：本项目 `load_dotenv(override=True)`，因此 `.env` 优先于同名 shell env。临时想用 shell 值覆盖某项，可以 `MINIMAX_API_KEY=xxx python -m …`。