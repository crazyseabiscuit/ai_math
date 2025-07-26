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
import ollama
from langchain_core.messages import AIMessage


# =========================
# 2. 自定义LLM类
# =========================
def ollama_deepseek_api(prompt, model="deepseek-r1:1.5b"):
    # 真实调用
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
    # mock 调试用
    # return f"【模拟 deepseek-r1:1.5b 回复】{prompt}"

class MyCustomLLM(LLM):
    def __init__(self, tools=None, **kwargs):
        super().__init__(**kwargs)
        self._tools = tools or []

    def bind_tools(self, tools):
        return MyCustomLLM(tools=tools)

    def my_model_api(self, prompt):
        # 只返回字符串
        if "搜索" in prompt:
            # 返回特殊标记，供 call_model 判断
            return "<<TOOL_CALL_SEARCH>>"
        return "【模拟 deepseek-r1:1.5b 回复】" + str(prompt)

    def _call(self, prompt, stop=None):
        return self.my_model_api(prompt)

    @property
    def _llm_type(self):
        return "my-custom-llm"


# =========================
# 3. 工具定义与注册
# =========================
def search_tool(query: str) -> str:
    # 这里实现你的工具逻辑
    print(f"*******************搜索结果: {query}")
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
    system_prompt = SystemMessage(
        content="你是一个智能助手，请根据用户的需求调用工具并给出答案。"
    )
    prompt = [system_prompt] + list(state["messages"])
    # 检查历史消息是否已经有 tool_call 或 tool message
    for msg in reversed(state["messages"]):
        # 如果已经有 tool message 或 tool_call，直接返回普通回复
        if isinstance(msg, ToolMessage):
            return {"messages": [AIMessage(content="【工具已调用，返回最终答案】")]}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # 已经有 tool_call，不再生成
            return {"messages": [AIMessage(content="【工具已调用，返回最终答案】")]}
    response = model.invoke(prompt)
    if response == "<<TOOL_CALL_SEARCH>>":
        # 构造 AIMessage，模拟 tool_call
        return {"messages": [
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call_search_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "LangGraph是什么？"}'
                            }
                        }
                    ]
                },
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"query": "LangGraph是什么？"},
                        "id": "call_search_1"
                    }
                ]
            )
        ]}
    else:
        return {"messages": [AIMessage(content=response)]}


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
