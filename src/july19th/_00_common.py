import os
import time
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
load_dotenv()
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"]  = os.getenv("2026_OPENAI_KEY")


def get_model() -> BaseChatModel:
    return init_chat_model("openai:gpt-5-mini")