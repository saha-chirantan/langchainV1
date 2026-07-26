from _00_common import get_model

from pydantic import BaseModel, Field



class Email(BaseModel):
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")

model = get_model()

structured_model = model.with_structured_output(Email)

response = structured_model.invoke("Write a message to my manager for leave")

print(type(response))
print(f"\n-----------------\n")
print(response)