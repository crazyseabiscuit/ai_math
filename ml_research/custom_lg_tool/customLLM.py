# -*- coding: utf-8 -*-
"""
自定义LLM与LangGraph工具注册及ReAct实现示例
"""

# =========================
# 1. 导入依赖
# =========================
from langchain.llms.base import LLM
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
import json


# =========================
# 2. 自定义LLM类
# =========================
class MyCustomLLM(LLM):
    def __init__(self, tools=None, **kwargs):
        super().__init__(**kwargs)
        self._tools = tools or []

    def bind_tools(self, tools):
        # 返回一个新的 LLM 实例，绑定工具
        return MyCustomLLM(tools=tools)

    def my_model_api(self, prompt):
        # 这里可以对 prompt 做格式化处理
        tool_names = ', '.join([t.name for t in self._tools]) if self._tools else '无'
        return f"【模拟回复】{prompt}\n[已绑定工具: {tool_names}]"

    def _call(self, prompt, stop=None):
        # 这里实现你自己的LLM调用逻辑
        # 比如调用本地模型或自定义API
        response = self.my_model_api(prompt)
        return response

    @property
    def _llm_type(self):
        return "my-custom-llm"


# =========================
# 3. 工具定义与注册
# =========================
def search_tool(query: str) -> str:
    # 这里实现你的工具逻辑
    return f"搜索结果: {query}"


search = Tool(name="search", func=search_tool, description="用于搜索信息的工具")

tools = [search]
# tool_executor = ToolExecutor(tools)


# =========================
# 4. 定义Graph State
# =========================
class AgentState(TypedDict):
    # 消息历史，LangGraph会自动管理
    messages: Annotated[Sequence, add_messages]


# =========================
# 5. 定义节点
# =========================
# 工具名称到工具的映射
tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


# 这里需要将工具绑定到LLM（假设你的自定义LLM支持bind_tools方法，否则略过）
model = MyCustomLLM().bind_tools(tools)
# model = MyCustomLLM()


def call_model(state: AgentState, config=None):
    # 系统提示词
    system_prompt = SystemMessage(
        content="你是一个智能助手，请根据用户的需求调用工具并给出答案。"
    )
    # 这里假设你的LLM支持messages输入，否则需要适配
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果AI消息有tool_calls，则继续，否则结束
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"


# =========================
# 6. 构建Graph
# =========================
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# =========================
# 7. 主程序入口
# =========================
def main():
    graph = build_graph()
    # 构造输入消息
    messages = [HumanMessage(content="帮我搜索一下LangGraph是什么？")]
    result = graph.invoke({"messages": messages})
    # 打印结果
    for m in result["messages"]:
        print(m)


if __name__ == "__main__":
    main()
