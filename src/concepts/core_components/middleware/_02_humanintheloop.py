from langchain_core.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from ....util._00_common import get_model
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def your_read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"


def your_send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"



api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = api_key
)

agent = create_agent(
    model = openai_gpt_model,
    checkpointer= InMemorySaver(),
    tools=[your_read_email_tool, your_send_email_tool],
    # middleware=[
    #     HumanInTheLoopMiddleware(
    #         interrupt_on={
    #             "your_send_email_tool":{
    #                 "allowed_decisions":["approve","edit","reject"]
    #             },
    #             "your_read_email_tool": False
    #         }
    #     )
    # ]
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)

print(f"Agent with HITL configured!")
config = {'configurable':{"thread_id":"hitl"}}

result =agent.invoke({"messages": [("user", "Send an email to my manager Rahulxyz on xyz@gmail.com, subject being a leave application and the body giving details about personal vacation between 1st september 2026 to 30th september 2026.")]}, config=config)

print(f'\n Result: {result}')