import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
from langchain.agents import create_agent

from typing import Union
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy


class BookingRequest(BaseModel):
    name: str = Field(description="The name of the customer")
    movie: str = Field(description="The name of the movie")
    intent: Literal["book", "cancel"] = Field(description="Whether this is a request for booking or cancellation")
    ticket_count: int = Field(description="Number of tickets to be booked or cancelled", default=1)



@tool
def peek_showtimes(movie_title: str) -> str:
    """Check showtimes for a movie."""
    return "7:00 PM and 10:15 PM"

openai_api_key = os.getenv("2026_OPENAI_KEY")
openai_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = openai_api_key
    )



# incomplete_model = openai_model.bind_tools([peek_showtimes]).with_structured_output(BookingRequest)
# print('\nAbout to invoke ')
# result = incomplete_model.invoke("Is Interstellar showing tonight? Book 2 seats for Rohan")
# print("Result:", result)



complete_openai_model = create_agent(
    model=openai_model,    
    tools=[peek_showtimes],
    response_format=BookingRequest
    )


better_response = complete_openai_model.invoke(
    {"messages":[{"role":"user","content":"Can you book me a seat for the 9:30 showing of dune part two? I'm Rohan"}]}
)

print(better_response['structured_response'])


print('\n Multiple possibiities of Structured Response \n')

class NewBooking(BaseModel):
    """A request to book NEW tickets."""
    customer_name: str
    movie_title: str
    ticket_count: int

class CancelBooking(BaseModel):
    """A request to CANCEL an existing booking."""
    customer_name: str
    movie_title: str


union_agent = create_agent(
    model=openai_model,    
    tools=[peek_showtimes],
    response_format=ToolStrategy(Union[NewBooking, CancelBooking])
    )


query = "Book one ticket for Oppenheimer for Mayank"
query1 = "I want to cancel my movie Oppenheimer, I'm Aisha."
response_new = union_agent.invoke({
    "messages":[{"role": "user", "content": f'{query1}'}]
})

print(f"{'='*50}")
print('Using Tool Strategy to decide between NewBooking and Cancel Booking.')
print(f"{'='*50}")
print(f"Query: {query1}")
print(f"response_new['structured_response']: {response_new['structured_response']}")

print(f"\n{'='*50}")
print('Extracted Details from query:')
print(f"{'='*50}")

if isinstance(response_new['structured_response'], NewBooking):
    print(f"New Booking Received for {response_new['structured_response'].ticket_count} tickets")
elif isinstance(response_new['structured_response'], CancelBooking):
    print(f"Cancellation Received for movie {response_new['structured_response'].movie_title}.")




