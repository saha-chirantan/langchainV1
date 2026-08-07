import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
load_dotenv()


api_key = os.getenv("2026_OPENAI_KEY")

"""
Tools:

Q: What they are: 
       => Tools are callable functions with well-defined inputs and outputs that get passed to a chat model.

Q: What does a tool do?
       => Tools extend what agents can do - letting the agent perform additional tasks which are beyond the basic capability of al LLM model to generate text 
           a) Fetch real-time data
           b) Execute code
           c) Query external databases
           etc.

Q: How/When does a model get choosen?
        => The model decides when to invoke a tool based on the conversation context, and what input arguments to provide.

"""

# Creating a basic tool
@tool("dummy_db_search")
def search_database(query: str, limit:int =10) -> str:
    """ Search the customer database for records matching teh query.
    
        Args:
             query: Search terms to look for
             limit: Maximum number of results to return
    
    """

    return f"Found {limit} results for '{query}'\n\n"

# In this file, we just define a tool and invoke the tool. We are not invoking model/agent.
# Tools can be invoked as well just like an agent since they both implement Runnable interface.

response = search_database.invoke("What is Langchain?")
print(f"\nType of response from tool invocation: {type(response)}")
print(f"\nActual from tool invocation: {response}\n---------------------------")