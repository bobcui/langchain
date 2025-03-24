from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=5)
#search_results = search.invoke("what are the recent IT events in Vancouver?")
#print(search_results)
tools = [search]


from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai", store=True)
# model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke("hi there! My name is Bo. What is the weather in Vancouver?")

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke({
    "messages": [HumanMessage("hi there! What is the weather in Shanghai?")]
})

print(response["messages"])


