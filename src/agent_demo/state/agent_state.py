from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class ContextSchema(BaseModel):
    model: str | None = Field(description="模型名称", default=None)
    temperature: float | None = Field(description="温度", default=None)
    top_p: float | None = Field(description="核采样概率", default=None)
    thinking: dict[str, Any] | None = Field(description="思考配置", default=None)

    # 提示词与工具配置
    system_prompt: str | None = Field(description="系统提示词", default=None)
    tools: list[str] | None = Field(description="工具列表", default=None)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 线程首次运行时生成，后续运行始终复用同一份快照
    system_prompt: str
