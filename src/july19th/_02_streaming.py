import os
import time
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
load_dotenv()

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"]  = os.getenv("2026_OPENAI_KEY")

# model = init_chat_model(
#     model="gpt-5-mini",
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
#     max_retries=6,
# )

model = init_chat_model("openai:gpt-5-mini")

chunks =[]
full_message = None

for chunk in model.stream("How does Langchain decide on which tool to use when a list of tools is provided?"):
    chunks.append(chunk)
    print(repr(chunk.text),type(chunk))
    full_message = chunk if full_message is None else  full_message + chunk
    time.sleep(1)

print(f"\n Final message:\n==================")
print(full_message)


