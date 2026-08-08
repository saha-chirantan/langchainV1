from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from ....util.print_output_headers import print_agent_response

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

load_dotenv()


class TicketRequest(BaseModel):
     """Input for Movie ticket booking"""
     movie_name: str = Field(description="Name of the movie")
     ticket_count: int = Field(description="Number of tickets", ge=1, le=10)
     preferred_row: Literal["front", "middle", "back"] = Field(default="middle", description="Preferred seating row")

@tool(args_schema=TicketRequest, description="Find showtimes when customers look for movie tickets.")
def check_showtimes(movie_name: str, ticket_count: int,   preferred_row:str) -> str:
     """Check available showtimes for a movie at the cinema.

    Args:
        movie_title: The exact title of the movie to check
    """
     return f"Checked {ticket_count} seat(s) in the {preferred_row} row for movie: {movie_name}."

tools_available = [check_showtimes]

print("\n==============================================")
print(f"Tool Name: {check_showtimes.name}")
print(f"Tool description: {check_showtimes.description}")
print(f"Tool arguments: {check_showtimes.args}")


api_key = os.getenv("2026_OPENAI_KEY")

model = init_chat_model(
    model="openai:gpt-3.5-turbo",
    api_key = api_key
)

agent = create_agent(
    model= model,
    tools=tools_available,
    response_format= ToolStrategy(
        TicketRequest,
        handle_errors="Ticket count must be between 1 and 10 -- please state a valid number."),
    system_prompt="Extract the booking details exactly as stated. Do not invent information."
)

query =  "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya."

response = agent.invoke({
     "messages":[
          {"role":"user","content":f"{query}"}
     ]
})

print_agent_response(response)