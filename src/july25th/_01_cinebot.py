import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"]= os.getenv("2026_OPENAI_KEY")

#######################################################################################################
###########      SECTION O - Create the Model - brain for the Cinebot
#######################################################################################################

openai_model = init_chat_model("openai:gpt-5-mini")

# Testing the model is working
openai_model.invoke('Hi')
print('Cinebot brain is connected')


booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]
#######################################################################################################
###########      SECTION 1 - Extract the relevant information from user messages
###########      Extracted data might not be in a proper structure if Structured Output is not used
#######################################################################################################


for msg in booking_requests:
    response = openai_model.invoke(f"Extract the customer's name, movie, and what they want (book or cancel) from: {msg}")
    print(response.content)
    print("\n------------")

"""
Output
-------

Name: Priya
Movie: Interstellar
Action: Book (2 tickets for the 7pm show tonight)
---
{
  "name": "Rohan",
  "movie": "Dune Part Two",
  "action": "book"
}
---
{
  "customer_name": "Aisha",
  "movie": "Oppenheimer",
  "request": "cancel booking"
}
---
"""
#######################################################################################################
###########      SECTION 2 - Extract the relevant information from user messages
###########      With Structured Output - Extracted data might will be in a proper structure
#######################################################################################################

from pydantic import BaseModel, Field
from typing import Literal

class BookingRequest(BaseModel):
    customer_name: str = Field(description="The customer's name")
    movie_title: str = Field(description="The movie they want to see")
    action: Literal["book", "cancel"] = Field(description="Whether this is a new booking or a cancellation")
    ticket_count: int = Field(description="How many tickets, default 1 if not mentioned", default=1)


openai_model_structured = openai_model.with_structured_output(BookingRequest)

print('Schema is defined')

for message in booking_requests:
    structured_response  = openai_model_structured.invoke(f"Extract the booking request from: {message}")
    print(structured_response)

"""
Output
-------
customer_name='Priya' movie_title='Interstellar' action='book' ticket_count=2
customer_name='Rohan' movie_title='Dune: Part Two' action='book' ticket_count=1
customer_name='Aisha' movie_title='Oppenheimer' action='cancel' ticket_count=1

"""

##################################################################################################################
###########      SECTION 3 - Tool Strategy and Provider Strategy
###########
###########      Two different mechanisms achieve the same guarantee. ProviderStrategy uses the model provider's 
###########      own native structured-output feature (fast, but only works where supported).
###########  
###########      ToolStrategy fakes it via a synthetic tool call (works almost everywhere, slightly slower).
##################################################################################################################

from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

provider_strategy_model = openai_model.with_structured_output(BookingRequest, strategy=ProviderStrategy[BookingRequest])