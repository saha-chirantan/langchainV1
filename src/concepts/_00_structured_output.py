import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
from pydantic import BaseModel, Field
from typing import Literal

openai_api_key = os.getenv("2026_OPENAI_KEY")


openai_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = openai_api_key
    )

booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]

for msg in booking_requests:
    response = openai_model.invoke(f"Extract the customer's name, movie, and what they want (book or cancel) from: {msg}")
    print("\n ===========================================\n Response content if unstructured response is allowed\n ===========================================")
    print(response.content)

"""
Output:
---------

 ===========================================
 Response content if unstructured response is allowed
 ===========================================
Name: Priya
Movie: Interstellar
Action: Book (2 tickets for the 7:00 PM show tonight)

 ===========================================
 Response content if unstructured response is allowed
 ===========================================
{
  "customer_name": "Rohan",
  "movie": "Dune Part Two",
  "action": "book"
}

 ===========================================
 Response content if unstructured response is allowed
 ===========================================
- Customer name: Aisha
- Movie: Oppenheimer
- Action: Cancel booking


Conclusion:
There is no fixed structure on how the model response looks like

"""

print()

class BookingRequest(BaseModel):
    name: str = Field(description="The name of the customer")
    movie: str = Field(description="The name of the movie")
    intent: Literal["book", "cancel"] = Field(description="Whether this is a request for booking or cancellation")
    ticket_count: int = Field(description="Number of tickets to be booked or cancelled", default=1)


openai_model_with_structured_output = openai_model.with_structured_output(BookingRequest)

for message in booking_requests:
    structured_response = openai_model_with_structured_output.invoke(message)
    print(f"Type of structured_response: {type(structured_response)}")
    print(f"structured_response: {structured_response}")