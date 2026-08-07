# learn-lang-chain 项目设计

日期：2026-07-31
目标用户：从零开始学习 LangChain 的开发者
模型：MiniMax-M3（可通过 `MINIMAX_MODEL` 覆盖）

## 1. 背景与目标

- 当前工作目录为空，需要从零搭建一个 LangChain 学习项目。
- 所有示例围绕 MiniMax 大模型展开，使用 OpenAI 兼容接口。
- 以可独立运行的脚本 + pytest 离线测试为载体，强调循序渐进。

## 2. 非目标

- 不构建生产级 LangChain 应用或服务端。
- 不覆盖 LangChain 所有模块（仅核心：ChatModel、Prompt、Parser、LCEL、Tools/Agent、RAG、Memory、多模态）。
- 不强制使用 uv、Poetry 等高级依赖管理工具（用户选择 pip + requirements.txt）。

## 3. 技术栈与依赖

- 语言：Python ≥ 3.10
- 核心依赖（固定在 requirements.txt）：
  - `langchain>=0.3,<0.4`
  - `langchain-core`
  - `langchain-openai`
  - `langchain-community`
  - `faiss-cpu`
  - `tiktoken`
  - `python-dotenv`
  - `pydantic>=2`
- 测试依赖：
  - `pytest`
  - `pytest-mock`
- 可选 extras（不在基线 requirements.txt 中提供，文档中提示按需安装）：
  - `unstructured`（高级文档加载）
  - `pillow`（多模态章节处理本地图片）

## 4. 项目结构

```
learn-lang-chain/
├── .env.example                # 环境变量模板
├── .gitignore
├── requirements.txt            # 基线依赖
├── README.md                   # 学习路线 + 运行说明
├── src/learn_lang_chain/
│   ├── __init__.py
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
    ├── test_ch01.py
    ├── test_ch02.py
    ├── test_ch03.py
    ├── test_ch04.py
    ├── test_ch05.py
    ├── test_ch06.py
    ├── test_ch07.py
    └── test_ch08.py
```

## 5. MiniMax 客户端封装

- MiniMax 提供与 OpenAI 兼容的 `/v1/chat/completions` 接口，因此复用 `langchain_openai.ChatOpenAI`。
- `src/learn_lang_chain/config.py` 暴露：
  - `get_chat_model(streaming: bool = False, temperature: float = 0.7) -> ChatOpenAI`
    - 从 `os.environ` 读取 `MINIMAX_API_HOST`、`MINIMAX_API_KEY`、`MINIMAX_MODEL`（默认 `MiniMax-M3`）。
    - 在导入时通过 `python-dotenv.load_dotenv()` 自动加载 `.env`。
    - 缺失 `MINIMAX_API_KEY` 时抛出 `EnvironmentError` 并提示复制 `.env.example`。
- 章节代码统一通过 `from learn_lang_chain.config import get_chat_model` 获取模型。

## 6. 配置与环境

- `.env.example` 内容：
  ```
  MINIMAX_API_HOST=https://你的-MiniMax-域名
  MINIMAX_API_KEY=your_api_key
  MINIMAX_MODEL=MiniMax-M3
  ```
- `.gitignore` 忽略 `.env`、`__pycache__/`、`.pytest_cache/`、`*.egg-info/`、`faiss_index/`。
- README 显式说明：示例需 `pip install -r requirements.txt` 后 `cp .env.example .env` 并填写 key。

## 7. 章节内容

每章以 `if __name__ == "__main__":` 包裹，可通过 `python -m learn_lang_chain.chapters.chXX` 运行。

1. **ch01_basic_chat**：直接调用 `ChatOpenAI`，演示普通调用与 `.stream()` 流式输出。
2. **ch02_prompt_template**：`PromptTemplate` 与 `ChatPromptTemplate`，演示变量替换与系统/用户角色。
3. **ch03_output_parser**：`StrOutputParser`、`JsonOutputParser`、基于 Pydantic 的结构化输出。
4. **ch04_lcel_chain**：使用 `|` 运算符组合 prompt + model + parser，演示 `RunnablePassthrough`、`RunnableLambda`、批处理 `.batch()` 与并发 `.stream()`。
5. **ch05_tools_agent**：用 `@tool` 定义本地工具（伪天气查询、计算器），结合 `create_react_agent` 与 `AgentExecutor`，展示工具调用循环。
6. **ch06_rag**：用 `TextLoader` + `RecursiveCharacterTextSplitter` + `OpenAIEmbeddings`（MiniMax 兼容）+ `FAISS` 构造向量库，使用 `Retriever` 与 LCEL 组合回答。
7. **ch07_memory**：在 LCEL 链中接入 `ConversationBufferMemory`（或新版 `RunnableWithMessageHistory`）实现多轮对话。
8. **ch08_multimodal**：构造包含 `image_url`（远程 URL 或本地 data URI）的 HumanMessage，演示图片理解。

## 8. 测试策略

- 所有测试均可在无真实 API key 下运行：
  - 使用 `langchain_core.language_models.fake_chat_models.FakeListChatModel` 注入固定回复。
  - 通过 `monkeypatch` 替换 `config.get_chat_model` 返回 fake 模型。
- `tests/conftest.py` 提供 `fake_chat` fixture。
- `tests/test_config.py` 验证：
  - 缺失 `MINIMAX_API_KEY` 时抛出 `EnvironmentError`。
  - 提供环境变量时 `get_chat_model` 返回的 `ChatOpenAI` 实例属性正确（用 `pytest-mock` 模拟 `ChatOpenAI` 构造）。
- 每个章节至少一个测试断言调用链能产出预期文本或结构。

## 9. 错误处理与日志

- `config.get_chat_model`：
  - 缺少 `MINIMAX_API_KEY` → 抛 `EnvironmentError("MINIMAX_API_KEY 未配置，请复制 .env.example 为 .env 并填写")`。
  - 缺少 `MINIMAX_API_HOST` → 抛 `EnvironmentError("MINIMAX_API_HOST 未配置")`。
- 章节代码不捕获 LLM 异常，让调用方看到真实堆栈，便于学习。
- README “常见问题” 段落覆盖：401（key 错）、超时（重试）、限流（指数退避提示）。

## 10. 验收标准

- `pip install -r requirements.txt && pytest` 在 CI 环境无网络时全部通过。
- 复制 `.env.example` 到 `.env` 并填入真实 key 后，`python -m learn_lang_chain.chapters.ch01` 可输出模型回复。
- README 列出每章学习目标、运行命令与对应测试。

## 11. 风险与权衡

- **MiniMax 接口兼容性**：依赖 OpenAI 兼容协议；若官方调整需在 `config.py` 调整参数。
- **Embeddings 兼容**：使用 `langchain_openai.OpenAIEmbeddings` 指向 MiniMax 的 embedding 接口；如官方未提供 embedding，ch06 退化为使用 BM25 检索（在本设计范围内可接受）。
- **新版本 LangChain API 变动**：固定 `<0.4` 版本区间，避免破坏性变更。