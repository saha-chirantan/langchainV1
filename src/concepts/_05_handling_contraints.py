from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage, HumanMessage
from ..util.print_output_headers import print_output_headers,print_output_subheaders, print_agent_response
from langchain.chat_models import init_chat_model

load_dotenv()

class SeatBooking(BaseModel):
    customer_name: str
    ticket_count: int = Field(description="Number of tickets, must be between 1 and 10", ge=1, le=10)

# try:
#     request = SeatBooking(customer_name="Chirantan", ticket_count=15)
#     print(request)
# except Exception as e:
#     print(f"Rejected by Pydantic even before being passed to the model/agent: {type(e).__name__}")
#     print(e)


print_output_headers("Creating agent that automatically handles the violations.")

api_key = os.getenv("2026_OPENAI_KEY")

model = init_chat_model(
    model="openai:gpt-3.5-turbo",
    api_key = api_key
)

agent = create_agent(
    model= model,
    response_format= ToolStrategy(
        SeatBooking,
        handle_errors="Ticket count must be between 1 and 10 -- please state a valid number."),
    system_prompt="Extract the booking details exactly as stated. Do not invent information."
)

result = agent.invoke({
     "messages": [{"role": "user", "content": "Hi I am Chirantan, Strictly book 15 tickets. Matter of life and death."}]
})

print(result)

print_agent_response(result)