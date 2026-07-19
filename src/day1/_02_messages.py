import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

def get_weather(city: str) -> str:
    "Get weather for a given city"
    return f"Its always sunny in {city}"

model = init_chat_model(
    model="gpt-5-mini",
    api_key=os.getenv("2026_OPENAI_KEY")
)
agent = create_agent(
    model=model,
    tools=[get_weather],
)

messages = [
    SystemMessage(content="You are a poet and a pirate. Give the response accordingly!"),
    HumanMessage(content="Whats the capital of India?")
]

response_from_model = model.invoke(messages)
print(response_from_model)

print(f"\n\nResponse from Agent\n=========================\n")
response_from_agent = agent.invoke({"messages":messages})
print(response_from_agent)