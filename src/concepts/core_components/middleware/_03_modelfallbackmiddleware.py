from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
load_dotenv()
from rich import print as rich_print, print_json

api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini-typo",
    api_key = api_key
)
openai_gpt_fallback_model = init_chat_model(
    model="openai:gpt-3.5-turbo",
    api_key = api_key
)

# openai_fallback_model = 

agent = create_agent(
    model=openai_gpt_model,
    middleware=[
        ModelFallbackMiddleware(
            openai_gpt_fallback_model
        )
    ]
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

print(f"Model Name: {result["messages"][-1].response_metadata["model_name"]}"  )
print(result["messages"][-1].content)
# print_json(result)

