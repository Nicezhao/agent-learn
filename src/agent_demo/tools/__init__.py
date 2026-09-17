from typing import Any

from langchain_core.tools import BaseTool

from agent_demo.tools.gettime import get_current_time
from agent_demo.tools.websearch import web_search, fetch_url

_AVAILABLE_TOOLS: dict[str, BaseTool] = {
    "fetch_url": fetch_url,
    "web_search": web_search,
    "get_current_time": get_current_time,
}


tools: list[BaseTool] = list(_AVAILABLE_TOOLS.values())


def get_tools(names: list[str]) -> list[BaseTool]:
    """根据工具名称列表返回对应的工具函数列表，未知名称会被忽略"""
    return [_AVAILABLE_TOOLS[name] for name in names if name in _AVAILABLE_TOOLS]


def resolve_tools(context: Any) -> list[BaseTool]:
    """解析当前运行上下文应启用的工具，未指定时回退到全量工具

    模型侧与 ToolNode 侧都必须走这里，否则两边的工具集合会不一致
    """
    names = getattr(context, "tools", None)
    if names is None:
        return tools
    return get_tools(names)
