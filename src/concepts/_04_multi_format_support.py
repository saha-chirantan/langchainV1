from pydantic import BaseModel
from typing import Union
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage, HumanMessage
from ..util.print_output_headers import print_output_headers,print_output_subheaders
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("2026_OPENAI_KEY")

class NewBooking(BaseModel):
    """A request to book NEW tickets."""
    customer_name: str
    movie_title: str
    ticket_count: int

class CancelBooking(BaseModel):
    """A request to CANCEL an existing booking."""
    customer_name: str
    movie_title: str

model = init_chat_model(
    model="openai:gpt-3.5-turbo",
    api_key = api_key
)

agent = create_agent(
    model= model,
    response_format= ToolStrategy(Union[NewBooking, CancelBooking ])
)

booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]



for request in booking_requests:
    print_output_headers(f"Processing request for: {request}")
    response = agent.invoke({
        "messages":[
            SystemMessage(content="You are a helpful movie booking agent."),
            HumanMessage(content=f"{request}")
        ]
    })
    print_output_subheaders(f"Response from agent:")
    print(f"Type of Structured Response: {type(response['structured_response'])}")
    print(f"Structured Response: {response['structured_response']}")
    print(f"AI Response from agent: {response['messages'][-1].content}")
