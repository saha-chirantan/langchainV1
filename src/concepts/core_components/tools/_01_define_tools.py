from langchain.tools import tool

@tool("overridden_name", description="Find showtimes when customers look for movie tickets.")
def check_showtimes(movie: str) -> str:
     """Check available showtimes for a movie at the cinema.

    Args:
        movie_title: The exact title of the movie to check
    """
     dummy_data = {
        "interstellar": "7:00 PM and 10:15 PM",
        "dune part two": "9:30 PM only",
        "oppenheimer": "Sold out for tonight"
     }

     return dummy_data.get(movie, "No showtimes found for that title.")

print("\n==============================================")
print(f"Tool Name: {check_showtimes.name}")
print(f"Tool description: {check_showtimes.description}")
print(f"Tool arguments: {check_showtimes.args}")