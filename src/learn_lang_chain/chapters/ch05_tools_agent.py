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
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
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
