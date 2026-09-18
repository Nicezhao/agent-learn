from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime

from agent_demo.nodes.call_model import call_model
from agent_demo.state.agent_state import AgentState, ContextSchema
from agent_demo.tools import resolve_tools

_tool_nodes: dict[tuple[str, ...], ToolNode] = {}


def _get_tool_node(context: ContextSchema) -> ToolNode:
    "按工具名集合缓存 ToolNode，避免每轮迭代都重建"
    active_tools = resolve_tools(context)
    key = tuple(sorted(tool.name for tool in active_tools))
    node = _tool_nodes.get(key)
    if node is None:
        # State 里的消息字段是 message（单数），需显式告知 ToolNode，
        # 否则结果会写进不存在的 messages channel 而被静默丢弃
        node = ToolNode(active_tools)
        _tool_nodes[key] = node
    return node


def should_continue(state: AgentState) -> str:
    lastMsg = state["messages"][-1]
    if not isinstance(lastMsg, AIMessage) or not lastMsg.tool_calls:
        return END

    # ToolNode 自身会并发执行本轮全部 tool_calls，无需按调用逐个 Send
    return "tools"


async def call_tools(state: AgentState, config: RunnableConfig) -> dict:
    "按当前 context 动态选取工具，保证与 call_model 下发给模型的工具集合一致"
    node = _get_tool_node(get_runtime(ContextSchema).context)
    return await node.ainvoke(state, config)


def build_graph():
    workflow = StateGraph(state_schema=AgentState, context_schema=ContextSchema)

    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges("call_model", should_continue, ["tools", END])  # type: ignore
    workflow.add_edge("tools", "call_model")

    return workflow


customizeAgent = build_graph().compile()
