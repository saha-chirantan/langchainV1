import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

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
    
    system_prompt="You are a helpful asssitant"
)

result = agent.invoke({"messages":[
    {"role":"user", "content":"What's the weather in Tokyo"}
]})


print(result)
print("\n\n\n")
print(result["messages"][-1].content)
