
def print_output_headers(header: str):
    print(f'\n{"="*65}')
    print(f'{'='*10}{' '*5}{header}{' '*5}{'='*10}')
    print(f'{"="*65}\n')

def print_output_subheaders(header: str):
    print(f'\n{"-"*65}')
    print(f'{'-'*10}{' '*5}{header}{' '*5}{'-'*10}')
    print(f'{"-"*65}\n')


import json
from pprint import pprint


def print_agent_response(response):
    print("\n" + "=" * 70)
    print("🤖 AGENT RESPONSE")
    print("=" * 70)

    # Print each message
    for i, message in enumerate(response.get("messages", []), 1):
        print(f"\n--- Message {i} ---")

        # Message type
        print(f"Type: {message.__class__.__name__}")

        # Content
        content = getattr(message, "content", None)

        if content:
            print("\nContent:")
            if isinstance(content, (dict, list)):
                print(json.dumps(content, indent=2, default=str))
            else:
                print(content)

        # Tool calls
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            print("\n🔧 Tool Calls:")
            print(json.dumps(tool_calls, indent=2, default=str))

    # Structured output
    structured_response = response.get("structured_response")

    if structured_response:
        print("\n" + "-" * 70)
        print("📦 STRUCTURED RESPONSE")
        print("-" * 70)

        if hasattr(structured_response, "model_dump"):
            data = structured_response.model_dump()
        elif hasattr(structured_response, "dict"):
            data = structured_response.dict()
        else:
            data = structured_response

        print(json.dumps(data, indent=2, default=str))

    print("\n" + "=" * 70)