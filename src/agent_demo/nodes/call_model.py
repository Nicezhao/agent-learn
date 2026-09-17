from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from agent_demo.core.config import anthropic_settings
from agent_demo.state.agent_state import AgentState, ContextSchema
from agent_demo.tools import resolve_tools

_MODEL_CONFIG_FIELDS = ("model", "temperature", "top_p", "thinking")


def create_model():
    return init_chat_model(
        model=anthropic_settings.default_model,
        model_provider="anthropic",
        thinking={"type": "disabled"},
        configurable_fields=["model", "temperature", "top_p", "thinking"],
    )


def _get_system_prompts(state: AgentState) -> list[SystemMessage]:
    "根据 Thread 状态获取首次运行时生成的系统提示词快照"
    system_prompt = state.get("system_prompt")
    if not system_prompt:
        return []
    return [SystemMessage(system_prompt)]


def _get_context() -> ContextSchema:
    return get_runtime(ContextSchema).context


def _get_tools(context: ContextSchema):
    return resolve_tools(context)


def _get_model_config(context: ContextSchema) -> RunnableConfig:
    "模型 invoke 只读 config.configurable，不读 Context，需手动把模型配置转成调用配置"
    if not context:
        return {}
    configurable = {
        name: getattr(context, name)
        for name in _MODEL_CONFIG_FIELDS
        if getattr(context, name) is not None
    }
    return {"configurable": configurable}


async def call_model(state: AgentState) -> dict:
    model = create_model()
    messages = state["message"]
    context = _get_context()

    system_prompts = _get_system_prompts(state)
    active_tools = _get_tools(context)
    model_with_tools = model.bind_tools(active_tools)

    # 调用模型
    response = await model_with_tools.ainvoke(
        [*system_prompts, *messages],
        config=_get_model_config(context),
    )
    return {"message": [response]}
