from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
# from ..day1._00_common import get_agent, get_model
from ..util._00_common import get_agent, get_model
from ..util.print_output_headers import print_output_headers
from typing import Callable
from langchain.messages import HumanMessage

load_dotenv()

class BookingRequest(BaseModel):
    customer_name: str = Field(description="The customer's name")
    movie_title: str = Field(description="The movie they want to see")
    action: Literal["book", "cancel"] = Field(description="Whether this is a new booking or a cancellation")
    ticket_count: int = Field(description="How many tickets, default 1 if not mentioned", default=1)

primitive_model_name ="openai:gpt-3.5-turbo"
advanced_model_name="openai:gpt-5-mini"

models = [primitive_model_name, advanced_model_name]

for model_name in models:
    model_to_use = get_model(model_name)
    supports_structured_output = model_to_use.profile['structured_output']
    # agent = get_agent(advanced_model_name)
    if supports_structured_output:
        # final_model = model_to_use.with_structured_output(BookingRequest, strategy=ProviderStrategy(BookingRequest))
        final_agent = create_agent(
            model= model_to_use,
            response_format= ProviderStrategy(BookingRequest)
        )

        # response = final_model.invoke("Can u book me a seat for the 9:30 showing of dune part two? I'm Rohan")
        response = final_agent.invoke({
            "messages": [{
                "role":"user", "content": "Can u book me a seat for the 9:30 showing of dune part two? I'm Rohan"
            }] 
            })
        print_output_headers(f"Results for using: {model_name}")
        print(f"Model name: {model_name}")
        print(f"Supports Structured output? {supports_structured_output}")
        print("\nDetails of response returned\n---------------------------")
        print(f'Type of Structured Response: {type(response['structured_response'])}')
        print(f"Structured Response: {response['structured_response']}")
    else:
        final_agent = create_agent(
                    model= model_to_use,
                    response_format= ToolStrategy(BookingRequest)
                )
        response = final_agent.invoke({
                    "messages": [{
                        "role":"user", "content": "Can u book me a seat for the 9:30 showing of dune part two? I'm Rohan"
                    }] 
                    })
        print_output_headers(f"Results for using: {model_name}")
        print(f"Model name: {model_name}")
        print(f"Supports Structured output? {supports_structured_output}")
        print("\nDetails of response returned\n---------------------------")
        print(f'Type of Structured Response: {type(response['structured_response'])}')
        print(f"Structured Response: {response['structured_response']}")
