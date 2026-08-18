from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from rich import print as rich_print

load_dotenv()

api_key = os.getenv("2026_OPENAI_KEY")

openai_gpt_model = init_chat_model(
    model="openai:gpt-5-mini",
    api_key = api_key
)

@tool
def check_showtimes(movie_title: str) -> str:
    """Check available showtimes for a movie at the cinema."""
    fake_showtimes = {
        "interstellar": "7:00 PM and 10:15 PM",
        "dune part two": "9:30 PM only",
        "oppenheimer": "Sold out for tonight",
    }
    return fake_showtimes.get(movie_title.lower(), "No showtimes found for that title.")

@tool
def book_seats(movie_title: str, seat_count: int) -> str:
    """Book seats for a movie. Irreversible once confirmed."""
    return f"Booked {seat_count} seat(s) for {movie_title}."

@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking. Irreversible."""
    return f"Booking {booking_id} cancelled."

@tool
def check_order_status(booking_id: str) -> str:
    """Check the status of an existing booking."""
    return f"Booking {booking_id}: confirmed, 2 seats, Interstellar, 7:00 PM."

@tool
def get_refund_policy() -> str:
    """Get the cinema's refund policy -- exact wording, not to be paraphrased."""
    return "Refunds available up to 2 hours before showtime. No refunds after that."

@tool
def lookup_seat_map(movie_title: str, seat_number: str) -> str:
    """Look up a specific seat -- fails if the seat number format is wrong."""
    if not seat_number or not seat_number[0].isalpha():
        raise ValueError(f"Malformed seat number '{seat_number}' -- expected a letter+number like 'A12'.")
    return f"Seat {seat_number} for {movie_title}: available."

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]

agent = create_agent(
    model=openai_gpt_model,
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=cinebot_tools,
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=5,
            run_limit=2,
            exit_behavior="end",
        ),
    ],
)

"""
Thread limit - # of model calls across all invocations
run limit - # of model calls per invoke()
"""

print(f"\n Invocation 1 \n===================")

result_1 = agent.invoke(
    {"messages": [("user", "Can you tell me cinema's refund policy? ")]},
    config={"configurable": {"thread_id": "thread-1"}},
)

rich_print(result_1)

print(f"\n Invocation 2 \n===================")
result_2 = agent.invoke(
    {"messages": [("user", "Give me the show times for the movie oppenheimer? ")]},
    config={"configurable": {"thread_id": "thread-1"}},
)

rich_print(result_2)

print(f"\n Invocation 3 \n===================")
result_3 = agent.invoke(
    {"messages": [("user", "Give me the show times for the movie interstellar? ")]},
    config={"configurable": {"thread_id": "thread-1"}},
)

rich_print(result_3)


"""
Output:

 Invocation 1 
===================
{
    'messages': [
        HumanMessage(
            content="Can you tell me cinema's refund policy? ",
            additional_kwargs={},
            response_metadata={},
            id='bc7b82c9-4920-406a-b797-096df43d5343'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 149,
                    'prompt_tokens': 286,
                    'total_tokens': 435,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 128,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54R0M3R6vRKR4EwNsgon1zpr8Oy',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-5608-7050-84fb-73c5d57387be-0',
            tool_calls=[
                {
                    'name': 'get_refund_policy',
                    'args': {},
                    'id': 'call_OxEX0jf6V9VYc0Wu4sLUcEJz',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 286,
                'output_tokens': 149,
                'total_tokens': 435,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 128}
            }
        ),
        ToolMessage(
            content='Refunds available up to 2 hours before showtime. No 
refunds after that.',
            name='get_refund_policy',
            id='5b2ba12a-5a63-4cbc-9af8-d461471536c6',
            tool_call_id='call_OxEX0jf6V9VYc0Wu4sLUcEJz'
        ),
        AIMessage(
            content='Here’s the cinema’s refund policy (exact 
wording):\n\n"Refunds available up to 2 hours before showtime. No refunds after
that."',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 32,
                    'prompt_tokens': 331,
                    'total_tokens': 363,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54UoHwV3BVmc6LZXWP0QiJW2hgG',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--01a012fa-668d-7a03-a2a3-70918e1b88d3-0',
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 331,
                'output_tokens': 32,
                'total_tokens': 363,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        )
    ]
}

 Invocation 2 
===================
{
    'messages': [
        HumanMessage(
            content="Can you tell me cinema's refund policy? ",
            additional_kwargs={},
            response_metadata={},
            id='bc7b82c9-4920-406a-b797-096df43d5343'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 149,
                    'prompt_tokens': 286,
                    'total_tokens': 435,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 128,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54R0M3R6vRKR4EwNsgon1zpr8Oy',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-5608-7050-84fb-73c5d57387be-0',
            tool_calls=[
                {
                    'name': 'get_refund_policy',
                    'args': {},
                    'id': 'call_OxEX0jf6V9VYc0Wu4sLUcEJz',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 286,
                'output_tokens': 149,
                'total_tokens': 435,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 128}
            }
        ),
        ToolMessage(
            content='Refunds available up to 2 hours before showtime. No 
refunds after that.',
            name='get_refund_policy',
            id='5b2ba12a-5a63-4cbc-9af8-d461471536c6',
            tool_call_id='call_OxEX0jf6V9VYc0Wu4sLUcEJz'
        ),
        AIMessage(
            content='Here’s the cinema’s refund policy (exact 
wording):\n\n"Refunds available up to 2 hours before showtime. No refunds after
that."',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 32,
                    'prompt_tokens': 331,
                    'total_tokens': 363,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54UoHwV3BVmc6LZXWP0QiJW2hgG',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--01a012fa-668d-7a03-a2a3-70918e1b88d3-0',
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 331,
                'output_tokens': 32,
                'total_tokens': 363,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        HumanMessage(
            content='Give me the show times for the movie oppenheimer? ',
            additional_kwargs={},
            response_metadata={},
            id='97e890d1-8df0-49c0-9297-8dde46c6562d'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 27,
                    'prompt_tokens': 383,
                    'total_tokens': 410,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54Vy8bIrOeH9Kb10Z4b8PFThuke',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-6ada-7380-bffa-31a05a4e16ae-0',
            tool_calls=[
                {
                    'name': 'check_showtimes',
                    'args': {'movie_title': 'Oppenheimer'},
                    'id': 'call_VeDyusweg4q3Td8Y6pi3QuHO',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 383,
                'output_tokens': 27,
                'total_tokens': 410,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        ToolMessage(
            content='Sold out for tonight',
            name='check_showtimes',
            id='05ce8650-c37a-42e1-a41b-9c9ed821ccee',
            tool_call_id='call_VeDyusweg4q3Td8Y6pi3QuHO'
        ),
        AIMessage(
            content='I checked for Oppenheimer — the showings are sold out for 
tonight.\n\nWould you like me to:\n- Check showtimes for a different date 
(which date?), \n- Look at nearby cinemas, or \n- Try to check specific seat 
availability / attempt a booking if any seats open?\n\nTell me which option 
(and a date or cinema if relevant) and I’ll check.',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 470,
                    'prompt_tokens': 420,
                    'total_tokens': 890,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 384,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54XeAmF6ZheYqImKm8eq8NSxkkX',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--01a012fa-6fd5-7d73-9f9f-6e0dd8ae617f-0',
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 420,
                'output_tokens': 470,
                'total_tokens': 890,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 384}
            }
        )
    ]
}

 Invocation 3 
===================
{
    'messages': [
        HumanMessage(
            content="Can you tell me cinema's refund policy? ",
            additional_kwargs={},
            response_metadata={},
            id='bc7b82c9-4920-406a-b797-096df43d5343'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 149,
                    'prompt_tokens': 286,
                    'total_tokens': 435,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 128,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54R0M3R6vRKR4EwNsgon1zpr8Oy',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-5608-7050-84fb-73c5d57387be-0',
            tool_calls=[
                {
                    'name': 'get_refund_policy',
                    'args': {},
                    'id': 'call_OxEX0jf6V9VYc0Wu4sLUcEJz',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 286,
                'output_tokens': 149,
                'total_tokens': 435,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 128}
            }
        ),
        ToolMessage(
            content='Refunds available up to 2 hours before showtime. No 
refunds after that.',
            name='get_refund_policy',
            id='5b2ba12a-5a63-4cbc-9af8-d461471536c6',
            tool_call_id='call_OxEX0jf6V9VYc0Wu4sLUcEJz'
        ),
        AIMessage(
            content='Here’s the cinema’s refund policy (exact 
wording):\n\n"Refunds available up to 2 hours before showtime. No refunds after
that."',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 32,
                    'prompt_tokens': 331,
                    'total_tokens': 363,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54UoHwV3BVmc6LZXWP0QiJW2hgG',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--01a012fa-668d-7a03-a2a3-70918e1b88d3-0',
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 331,
                'output_tokens': 32,
                'total_tokens': 363,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        HumanMessage(
            content='Give me the show times for the movie oppenheimer? ',
            additional_kwargs={},
            response_metadata={},
            id='97e890d1-8df0-49c0-9297-8dde46c6562d'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 27,
                    'prompt_tokens': 383,
                    'total_tokens': 410,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54Vy8bIrOeH9Kb10Z4b8PFThuke',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-6ada-7380-bffa-31a05a4e16ae-0',
            tool_calls=[
                {
                    'name': 'check_showtimes',
                    'args': {'movie_title': 'Oppenheimer'},
                    'id': 'call_VeDyusweg4q3Td8Y6pi3QuHO',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 383,
                'output_tokens': 27,
                'total_tokens': 410,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        ToolMessage(
            content='Sold out for tonight',
            name='check_showtimes',
            id='05ce8650-c37a-42e1-a41b-9c9ed821ccee',
            tool_call_id='call_VeDyusweg4q3Td8Y6pi3QuHO'
        ),
        AIMessage(
            content='I checked for Oppenheimer — the showings are sold out for 
tonight.\n\nWould you like me to:\n- Check showtimes for a different date 
(which date?), \n- Look at nearby cinemas, or \n- Try to check specific seat 
availability / attempt a booking if any seats open?\n\nTell me which option 
(and a date or cinema if relevant) and I’ll check.',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 470,
                    'prompt_tokens': 420,
                    'total_tokens': 890,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 384,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54XeAmF6ZheYqImKm8eq8NSxkkX',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--01a012fa-6fd5-7d73-9f9f-6e0dd8ae617f-0',
            tool_calls=[],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 420,
                'output_tokens': 470,
                'total_tokens': 890,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 384}
            }
        ),
        HumanMessage(
            content='Give me the show times for the movie interstellar? ',
            additional_kwargs={},
            response_metadata={},
            id='d6d4cb53-f436-4bdf-9374-36e4c322bc16'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 26,
                    'prompt_tokens': 519,
                    'total_tokens': 545,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cache_write_tokens': None,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-mini-2025-08-07',
                'system_fingerprint': None,
                'id': 'chatcmpl-EE54dU9geMd2jLmuebZnViVr3D7WX',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--01a012fa-88b2-7381-ba72-5473a322cbdc-0',
            tool_calls=[
                {
                    'name': 'check_showtimes',
                    'args': {'movie_title': 'Interstellar'},
                    'id': 'call_gf4iAQLr3PG4ZQeUpmsqqmKo',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 519,
                'output_tokens': 26,
                'total_tokens': 545,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        ToolMessage(
            content='7:00 PM and 10:15 PM',
            name='check_showtimes',
            id='876dde06-2174-4e96-9f4f-dd13c63e9704',
            tool_call_id='call_gf4iAQLr3PG4ZQeUpmsqqmKo'
        ),
        AIMessage(
            content='Model call limits exceeded: thread limit (5/5)',
            additional_kwargs={},
            response_metadata={},
            id='131be0a1-77dd-49f6-805f-4c7f19a15c87',
            tool_calls=[],
            invalid_tool_calls=[]
        )
    ]
}
"""

