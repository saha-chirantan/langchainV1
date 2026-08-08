from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
# from ..day1._00_common import get_agent, get_model
from ..day1._00_common import get_agent, get_model

load_dotenv()

model_name="openai:gpt-5-mini"
# model = get_model()

agent = get_agent(model_name)

class BookingRequest(BaseModel):
    customer_name: str = Field(description="The customer's name")
    movie_title: str = Field(description="The movie they want to see")
    action: Literal["book", "cancel"] = Field(description="Whether this is a new booking or a cancellation")
    ticket_count: int = Field(description="How many tickets, default 1 if not mentioned", default=1)

