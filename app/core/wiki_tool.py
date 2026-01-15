import wikipedia
from langchain.tools import Tool

def wiki_search(query: str):
    try:
        return wikipedia.summary(query, sentences=3)
    except:
        return "No Wikipedia data found."

wiki_tool = Tool(
    name="Wikipedia Search",
    func=wiki_search,
    description="Search Wikipedia for general knowledge queries."
)
