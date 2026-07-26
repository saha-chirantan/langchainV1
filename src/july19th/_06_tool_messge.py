from _00_common import get_model

model = get_model()

def get_weather(location: str) -> str:
    "Get the weather at a location"
    print("I was called from somewhere")
    
    # Hardcoded response
    return f"Its sunny in {location}"

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke('What is the weather in Delhi')

print("\n=========== Type of Response ===================\n")
print(type(response))
print("\n=========== Response ===================\n")
print(response)
print("\n==============Tool Calls================\n")
print(response.tool_calls)

print("\n============== Details of tool call================\n")
for tool_call in response.tool_calls:
    print(f'Tool name: {tool_call['name']}')
    print(f'Tool arguments: {tool_call['args']}')
    print(f'Tool Id: {tool_call['id']}')


"""
=========== Type of Response ===================

<class 'langchain_core.messages.ai.AIMessage'>

=========== Response ===================

content='' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 89, 'prompt_tokens': 131, 'total_tokens': 220, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 64, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-E5LoDgyyLWa63dfH6QdqiVmMk4h9S', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None} id='lc_run--019f96f6-b8c7-7b62-ae18-31f6057a250b-0' tool_calls=[{'name': 'get_weather', 'args': {'location': 'Delhi, India'}, 'id': 'call_OLGDMoUZvQ1GuW8pDY3n6C5A', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 131, 'output_tokens': 89, 'total_tokens': 220, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 64}}

==============Tool Calls================

[{'name': 'get_weather', 'args': {'location': 'Delhi, India'}, 'id': 'call_OLGDMoUZvQ1GuW8pDY3n6C5A', 'type': 'tool_call'}]
(langchain) 

"""

