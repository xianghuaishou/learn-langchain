"""第 5 章：Tools & Agent（工具与 ReAct 代理）。

本章演示如何给一个聊天模型**挂载工具（tools）**——也就是让模型可以决定
调用的函数——并通过 LangChain 的 **ReAct** 模式自动编排"思考 → 调用工具 →
观察结果 → 继续推理"的循环。

两个核心概念：
- `@tool` 装饰器：把一个普通 Python 函数包装成模型可识别的工具对象。
- `AgentExecutor`：拿到 ReAct agent 后，负责真正执行工具调用并循环，
  直到模型给出最终答案。
"""

import ast
import operator

# 导入 `tool` 装饰器。`@tool` 会**反射**读取被装饰函数的：
#   1. 参数名 + 类型注解 → 生成 JSON Schema（告诉模型参数是什么类型）
#   2. 函数的 docstring  → 作为工具的"描述"（告诉模型何时该用这个工具）
# 装饰之后得到的对象是一个 `BaseTool`（同时可调用），模型就能在推理时
# 选择调用它。详见：https://python.langchain.com/docs/how_to/custom_tools/
from langchain_core.tools import tool

from learn_lang_chain.config import get_chat_model


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（伪实现）。

    ⚠️ 这只是一个**桩函数（stub）**，永远返回写死的"晴，25°C"。
    在生产环境中，你应该在这里调用真实的天气 API
    （例如 OpenWeather、和风天气、高德天气等），并把返回结果序列化成字符串。

    LangChain 通过两件事把这个函数暴露给模型：
    - 参数 `city: str` 的类型注解 → 生成 `{"city": {"type": "string"}}` 的 schema
    - 上面的 docstring            → 作为工具描述，告诉模型这个工具是干嘛的
    """
    return f"{city}：晴，25°C"


# AST 运算符节点 → Python 运算符函数的查找表（白名单）。
# `ast.parse(...)` 会把字符串解析成抽象语法树（AST），树上的每个运算符
# 节点（如 `ast.Add`、`ast.Mult`）都对应 Python 的一个内置运算符函数。
# 我们只把"安全的算术运算"放进这张表，这样 `_eval` 遇到任何不在表里的
# 节点都会抛错，从而把计算器变成一个**沙箱**：用户没法执行任意 Python 代码。
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}


@tool
def calculator(expr: str) -> float:
    """计算简单数学表达式，仅支持 + - * / ** %。

    安全要点：我们**故意不用** Python 自带的 `eval(expr)`，因为 `eval`
    会拿到当前的 globals/locals，等于让用户执行任意 Python 代码
    （例如 `__import__("os").system("rm -rf /")`）。

    这里改用 `ast.parse(expr, mode="eval")`：
    - `mode="eval"` 表示只能解析成单个表达式（不能是语句）
    - 我们手动遍历 AST 节点，**白名单**只支持以下运算符：
        `+  -  *  /  **  %` 以及一元负号 `-x`
    - 任何不在白名单里的节点（例如函数调用 `f(x)`、属性访问 `a.b`、
      名称 `__import__`）都会被 `_eval` 抛 `ValueError`，从而阻止执行。
    """
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def _eval(node):
    """递归地求值一个 AST 节点。

    三种合法情况：
    1. **常量** `ast.Constant`（数字字面量）→ 直接返回它的数值。
    2. **二元运算** `ast.BinOp`（例如 `a + b`）→ 从 `_BIN_OPS` 查表得到
       对应的运算符函数，递归求值左右子树后再做运算。
    3. **一元运算** `ast.UnaryOp`，且操作符是 `ast.USub`（即负号 `-x`）
       → 递归求值操作数后取相反数。

    小细节：`ast.Constant` 是 Python 3.8+ 引入的统一字面量节点，
    取代了旧的 `ast.Num` / `ast.Str` / `ast.Bytes`。`ast.Num` 在
    Python 3.12 中被正式移除，所以这里只能用 `ast.Constant`。

    安全兜底：遇到任何不在白名单里的节点类型，就抛出 `ValueError`。
    例如用户输入 `__import__("os")` 会被解析成 `ast.Call` 节点，
    它既不是常量也不是二元/一元运算，于是走到最后一行抛错，
    阻止了任意代码执行。
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


# 这个列表是 **agent 与外部世界之间的"契约"**：`create_react_agent`
# 会遍历它，把每个工具的（名字 + 描述 + 参数 schema）拼成一段提示词，
# 告诉模型"你有这些工具可用"。之后模型每一步推理都可能从这个列表里
# 挑一个工具来调用。增删工具时，只需改这一个列表即可。
TOOLS = [get_weather, calculator]


def run_agent(question: str) -> str:
    """构造并运行一个 ReAct agent 来回答 `question`。

    流程拆解：

    1. `ChatPromptTemplate` 构造提示词模板：
       - system：给模型设定角色（"必要时调用工具"）。
       - human：用户的实际问题，用 `{input}` 占位。
       - **关键**：`("placeholder", "{agent_scratchpad}")`。
         `MessagesPlaceholder` 会在运行时被替换成模型与工具之间
         全部的"草稿"对话历史——也就是 ReAct 里的 Thought / Action /
         Observation 这三件套。少了它，agent 就没有记忆，无法多步推理。

    2. `create_react_agent(model, tools, prompt)` 返回一个 `Runnable`。
       给它输入，它会输出一连串 `AIMessage` / `ToolMessage`，
       描述 agent 的推理轨迹（要调用哪个工具、参数是什么）。

    3. `AgentExecutor` 把上面的 Runnable **包装**起来，加上运行时
       才有的能力：
       (a) 真正去执行工具调用（拿到 `ToolMessage` 的 observation）
       (b) 循环：把 observation 塞回 scratchpad，让 agent 继续推理，
           直到它选择不再调用工具（输出最终答案）
       (c) `verbose=True` 把每一步 Thought / Action / Observation
           打到 stdout。**学习时很有用，生产环境太吵，建议关掉或改成日志**。
    """
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
    # 第一个问题专门触发 `get_weather` 工具（关键词：天气 + 城市）。
    print(run_agent("上海今天天气如何？"))
    # 第二个问题专门触发 `calculator` 工具（关键词：计算 + 数学表达式）。
    print(run_agent("计算 (3 + 5) * 2"))