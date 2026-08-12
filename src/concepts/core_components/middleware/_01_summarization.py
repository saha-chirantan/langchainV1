from langchain_core.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from ....util._00_common import get_model
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv()

# import sqlite3

@tool
def save_trip_demo(user_id: str, destination: str) -> str:
    """Save a trip to the database. Irreversible without manual cleanup."""
    return f"Trip to {destination} saved for {user_id}."

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75F."


def show_agent_response(label, result):
    print(f"\n--- {label} ---  ({len(result['messages'])} messages in state)")
    for m in result["messages"]:
        print(f"[{m.type}] {str(m.content)[:150]}")


api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = api_key
)
agent = create_agent(
    model=openai_gpt_model,
    
    tools=[save_trip_demo, get_weather],
    middleware=[
        SummarizationMiddleware(
            model=openai_gpt_model,
            # trigger=('token', 3000),
            keep= ('messages', 3)

        )
    ]
)



show_agent_response('Level 1', agent.invoke({
    "messages":[{"role":"user", "content":"Hi, my name is Marcus and I live in Austin."}]
}))
show_agent_response('Level 2', agent.invoke({
    "messages":[{"role":"user", "content":"What's the weather like where I live?."}]
}))
show_agent_response('Level 3', agent.invoke({
    "messages":[{"role":"user", "content":"Great, thanks. Any tips for hot weather?"}]
}))

show_agent_response('Level 4', agent.invoke({
    "messages":[{"role":"user", "content":"What's my name and where do I live?"}]
}))
