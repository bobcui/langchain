from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=1)
search_results = search.invoke("what are the recent IT events in Vancouver?")
print(search_results)

tools = [search]

