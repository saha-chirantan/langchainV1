import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
load_dotenv()
from pprint import pprint


from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="gpt-5-mini",
    api_key=os.getenv("2026_OPENAI_KEY"),
    temperature=0.7,
    max_tokens=400,
    max_retries=6
)

messages = [
    SystemMessage(content="You are a poet and a pirate. Give the response accordingly!"),
    HumanMessage(content="Whats the capital of India?")
]

response_from_model = model.invoke(messages)
pprint(response_from_model)

print("\n\n=====================================")
print("text:             ", response_from_model.text)
print("content_blocks:   ", response_from_model.content_blocks)
print("id:               ", response_from_model.id)
print("tool_calls:       ", response_from_model.tool_calls)
