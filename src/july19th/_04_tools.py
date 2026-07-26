from _00_common import get_model


model = get_model()


def get_weather(location: str) -> str:
    "Get the weather at a location"
    print("I was called from somewhere")
    
    # Hardcoded response
    return f"Its sunny in {location}"

def set_password(new_pass):
    "It is tool to set password"
    return "Password set!"


model_with_tools = model.bind_tools([get_weather, set_password])

response = model_with_tools.invoke('Set password to admin123')

print(response.tool_calls)
