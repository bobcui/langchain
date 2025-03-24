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
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# response = agent_executor.invoke({
#     "messages": [HumanMessage("hi there! What is the weather in Shanghai?")]
# })
# print(response["messages"])


# for step in agent_executor.stream(
#     {"messages": [HumanMessage(content="hi there! What is the weather in Shanghai?")]},
#     stream_mode="values",
# ):
#     message = step["messages"][-1]
#     if isinstance(message, tuple):
#         print(message)
#     else:
#         message.pretty_print()    

for step, metadata in agent_executor.stream(
     {"messages": [HumanMessage(content="hi there! What is the weather in Shanghai?")]},
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")


# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="hi im bob!")]}, config
# ):
#     print(chunk)
#     print("----")

# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats my name?")]}, config
# ):
#     print(chunk)
#     print("----")