from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
import os

api_key  = os.getenv("2026_OPENAI_KEY")

@tool
def standard_booking(movie_title: str) -> str:
    """Book a standard seat."""
    return f"Standard seat booked for {movie_title}."

@tool
def vip_lounge_booking(movie_title: str) -> str:
    """Book a VIP lounge seat with premium service. VIP members only."""
    return f"VIP lounge seat booked for {movie_title}."

model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key=api_key)

@wrap_model_call
def gate_vip_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Only expose vip_lounge_booking to VIP members."""
    is_vip = request.state.get("is_vip_member", False)
    print(f'Is Vip? {is_vip}')
    if not is_vip:
        allowed = [t for t in request.tools if t.name != "vip_lounge_booking"]
        request = request.override(tools=allowed)
    return handler(request)

gated_agent = create_agent(
    # model="openai:gpt-5-mini",
    model=model,
    tools=[standard_booking, vip_lounge_booking],
    middleware=[gate_vip_tools],
)

result_regular = gated_agent.invoke({"messages": [("user", "Book me a VIP lounge seat for Dune?")]})
print("Regular member result:", result_regular["messages"][-1].content)


result_vip = gated_agent.invoke(
    {"messages": [("user", "Book me a VIP lounge seat for Dune")], "is_vip_member": True}
)
print("VIP member result:", result_vip["messages"][-1].content)
print()
print("Same code, same query -- only the 'is_vip_member' flag differed. The model literally")
print("could not choose vip_lounge_booking in the first case -- it wasn't on its menu at all.")
