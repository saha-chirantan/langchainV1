from _00_common import get_model

model = get_model()

question1 = "Hi, whats the capital of Canada?"
question2 = "Explain langchain in less tha 50 words."
question3 = "Explain agents in less than 30 words."



# responses = model.batch(
#     [question1,question2,question3]
#     )



responses = model.batch_as_completed(
    [question1,question2,question3]
)

for response in responses:
    print(response)
    print("\n=================\n")

