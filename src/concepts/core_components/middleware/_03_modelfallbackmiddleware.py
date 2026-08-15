from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = api_key
)

# openai_fallback_model = 

agent = create_agent(
    model=openai_gpt_model,
    middleware=[
        ModelFallbackMiddleware(
             "openai:gpt-3.5-turbo"
        )
    ]
)


