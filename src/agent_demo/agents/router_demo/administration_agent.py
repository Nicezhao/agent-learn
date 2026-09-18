from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from agent_demo.agents.router_demo.router_types import RouteConfig
from agent_demo.core.config import anthropic_settings

system_prompt = """


"""

model = init_chat_model(
    model=anthropic_settings.default_model,
    model_provider="anthropic",
    thinking={"type": "disabled"},
)

agent = create_agent(model=model, system_prompt="")


async def administration(state: RouteConfig):
    pass
