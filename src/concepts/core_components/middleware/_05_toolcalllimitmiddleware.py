from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from rich import print as rich_print

load_dotenv()

api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = api_key
)