import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
load_dotenv()


from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="gpt-5-mini",
    api_key=os.getenv("2026_OPENAI_KEY")
)


openrouter_model = ChatOpenRouter(
    model="openrouter/free",
    api_key=os.getenv("2026_OPENROUTER_KEY")
)

response = openrouter_model.invoke("Hello Which model are you ?")
print(response)