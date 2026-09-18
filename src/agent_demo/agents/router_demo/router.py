from langgraph.graph import START, StateGraph
from langgraph.types import Send

from agent_demo.agents.router_demo.router_node import router_node
from agent_demo.agents.router_demo.router_types import RouteState
from agent_demo.agents.router_demo.unknown import unknown

workflow = StateGraph(RouteState)


def after_route(state: RouteState) -> list[Send]:
    return [Send(route["agent_name"], route) for route in state["routes"]]


agent = (
    workflow.add_node("router_node", router_node)
    .add_node("unknown", unknown)
    .add_edge(START, "router_node")
    # LangGraph 的 path 参数签名没覆盖返回 Send 的情况，运行时支持，故忽略类型检查
    .add_conditional_edges("router_node", after_route)  # type: ignore[arg-type]
)
