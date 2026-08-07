from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.agents import create_agent
from langchain.tools import tool
from ..day1._00_common import get_model

@tool
def standard_booking(movie_title: str) -> str:
    """Book a standard seat."""
    return f"Standard seat booked for {movie_title}."

@tool
def vip_lounge_booking(movie_title: str) -> str:
    """Book a VIP lounge seat with premium service. VIP members only."""
    return f"VIP lounge seat booked for {movie_title}."

model = get_model()

# @wrap_model_call
# def gate_vip_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
#     """Only expose vip_lounge_booking to VIP members."""
#     is_vip = request.state.get("is_vip_member", False)
#     if not is_vip:
#         allowed = [t for t in request.tools if t.name != "vip_lounge_booking"]
#         request = request.override(tools=allowed)
#     return handler(request)

gated_agent = create_agent(
    # model="openai:gpt-5-mini",
    model=model,
    tools=[standard_booking, vip_lounge_booking],
    #middleware=[gate_vip_tools],
)

result_regular = gated_agent.invoke({"messages": [("user", "Book me a VIP lounge seat for Dune?")]})
print("Regular member result:", result_regular["messages"][-1].content)