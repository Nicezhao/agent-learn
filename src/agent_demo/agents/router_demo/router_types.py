from typing import Annotated, Literal, TypedDict

from langchain.messages import AnyMessage, HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

type AgentName = Literal["administration", "hr", "it", "unknown"]


class RouteConfig(TypedDict):
    agent_name: AgentName
    query: str


class RouteAgentResult(BaseModel):
    routes: list[RouteConfig] = Field(description="路由结果", min_length=1)


class Answer(RouteConfig):
    answer: str


def reducer_answer(prev, next):
    if next == "clear":
        return []
    if not prev:
        return next
    return prev + next


class RouteState(TypedDict):
    query: str  # 用户当前的请求
    messages: Annotated[list[AnyMessage], add_messages]
    routes: list[RouteConfig]
    answers: Annotated[list[Answer], reducer_answer]


async def agent_answer(state: RouteConfig, agent):
    query = state["query"]
    agent_name = state["agent_name"]
    result = await agent.ainvoke(input={"messages": [HumanMessage(query)]})
    answer = result["messages"][-1].content
    return {"answers": [{"query": query, "agent_name": agent_name, "answer": answer}]}
