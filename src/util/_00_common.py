import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
load_dotenv()
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"]  = os.getenv("2026_OPENAI_KEY")


def get_model_raw() -> BaseChatModel:
    return init_chat_model("openai:gpt-5-mini")

def get_model(model: str) -> BaseChatModel:
    return init_chat_model(model)


def get_agent(model_name: str):
    model = get_model(model_name)
    print(f'\n[Inside get_agent()]: Printing model profile')
    print(f'Does the model support Structured Output?: {model.profile['structured_output']}\n')
    agent = create_agent(
        model=model

    )