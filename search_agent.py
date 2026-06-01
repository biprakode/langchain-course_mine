from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel , Field

load_dotenv()

# from tavily import TavilyClient
#
# tavily = TavilyClient()
#
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the internet
#     :param query : The query to search for:
#     :return returns: The search result:
#     """
#     return tavily.search(query = query , include_answer="basic",
#                          search_depth="advanced",
#                          max_results=1,
#                          country="united states")

from langchain_tavily import TavilySearch
tavily_search = TavilySearch(max_results=10)

@tool
def clean_search(query: str) -> str:
    """
    Search the internet for real-time information, news, and current events.
    :param query: The search query string.
    :return: Search result snippets.
    """
    # Simply pass the string query directly to the Tavily client run method
    return tavily_search.run(query)

llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [clean_search]


agent = create_agent(model = llm, tools = tools)

result = agent.invoke({
    "messages": [
        HumanMessage(content="Who is most likely to win F1 WDC 2026??")
    ]
})
print(result["messages"][-1].content)