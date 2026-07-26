import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
load_dotenv()

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"]  = os.getenv("2026_OPENAI_KEY")

model = init_chat_model(
    model="gpt-5-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,
)

# this is kind of few shot prompting!
messages = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="Whats the capital of India?"),
    AIMessage(content="The capital of India is New Delhi!"),
    HumanMessage(content="What is the capital of Srilanka?")
]

response = model.invoke(messages)

print(response)