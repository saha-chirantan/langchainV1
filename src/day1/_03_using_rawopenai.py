import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("2026_OPENAI_KEY"))

response = client.responses.create(
    model="gpt-5-mini",
    input="Whats the capital of Karnataka, India?",
    
)

print(response)
print("\n\n==========================\n")
print(response.output_text)