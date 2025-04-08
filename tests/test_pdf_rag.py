import os
import getpass

from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

vector_store = None
llm = None
config = None

def get_new_vector_store() -> VectorStore:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

@tool
def load_pdf_into_vector(file_path) -> str:
    """Use this tool when the user asks to load or upload a PDF file. Input should be the file url, like 'https://arxiv.org/pdf/2503.00085'."""

    print(f"Loading PDF into vector store: {file_path}")
    global vector_store
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pdf = loader.load()
        _ = vector_store.add_documents(documents=pdf)
        return "Pdf is loaded into vector."
    except Exception as e:
        return f"Error loading PDF: {e}"

@tool
def clear_pdf_vector_store() -> str:
    """Clear the current PDF data and reset the vector store."""

    print("Clearing vector store...")

    global vector_store
    vector_store = get_new_vector_store()
    return "Vector store cleared."

@tool(response_format="content_and_artifact")
def query_pdf_with_vector(query: str) -> tuple[str, list]:
    """Retrieve information related to a query from the loaded PDF."""
    
    print(f"Querying PDF with vector store: {query}")

    global vector_store
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    print(serialized[:300])
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState) -> dict:
    """Let GPT decide whether to respond or call a tool."""
    print("Deciding whether to respond or call a tool...")
    llm_with_tools = llm.bind_tools([load_pdf_into_vector, clear_pdf_vector_store, query_pdf_with_vector])
    response = llm_with_tools.invoke(state["messages"], config=config)
    return {"messages": [response]}

def generate(state: MessagesState) -> dict:
    """Generate GPT response using the output of tools."""
    print("Generating final response...")
    tool_outputs = [msg for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage)]
    docs_content = "\n\n".join(msg.content for msg in reversed(tool_outputs))
    
    system_message = SystemMessage(
        content=(
            "You are an assistant for answering questions based on retrieved context. "
            "If you don't know the answer, say so. Be concise (max 3 sentences).\n\n"
            f"{docs_content}"
        )
    )

    convo = [
        msg for msg in state["messages"]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", False)
    ]

    final_response = llm.invoke([system_message] + convo, config=config)
    return {"messages": final_response}

def route_after_tool(state: MessagesState):
    last_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    if last_ai_msg and last_ai_msg.tool_calls:
        name = last_ai_msg.tool_calls[0]["name"]
        print(f"last_ai_msg: {name}")
        return "generate" if name == "query_pdf_with_vector" else END
    return END


if __name__ == "__main__":

    llm = init_chat_model("gpt-4o-mini", model_provider="openai", store=True)
    vector_store = get_new_vector_store()
    config = {"configurable": {"thread_id": "aaa"}}
    memory = MemorySaver()

    tools = ToolNode([
        load_pdf_into_vector,
        clear_pdf_vector_store,
        query_pdf_with_vector
    ])

    graph = (
        StateGraph(MessagesState)
        .add_node("query_or_respond", query_or_respond)
        .add_node("tools", tools)
        .add_node("generate", generate)
        .set_entry_point("query_or_respond")
        .add_conditional_edges("query_or_respond", tools_condition, {
            "tools": "tools",
            END: END
        })
        .add_conditional_edges("tools", route_after_tool, {
            "generate": "generate",
            END: END
        })        
        .add_edge("generate", END)
        .compile(checkpointer=memory)
    )

    agent = create_react_agent(llm, 
        [load_pdf_into_vector, clear_pdf_vector_store, query_pdf_with_vector],
        checkpointer=memory)

    mode = input("👉 Choose mode ('agent' or 'graph'): ").strip().lower()

    print("🤖 LangGraph PDF Chatbot")
    print("Say something like:")
    print(" - 'Load this PDF: https://example.com/doc.pdf'")
    print(" - 'What does the PDF say about climate change?'")
    print(" - 'Clear everything'")
    print(" - Type 'exit' to quit.\n")

    def print_msg(msg):
        if isinstance(msg, tuple):
            print(msg)
        elif isinstance(msg, AIMessage):
            print(f"\n🤖 Bot: {msg.content}\n")
        else:
            msg.pretty_print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        if mode == "agent":
            for step in agent.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                stream_mode="values",
                config=config
            ):
                print_msg(step["messages"][-1])
        else:
            for step in graph.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                stream_mode="values",
                config=config
            ):
                print_msg(step["messages"][-1])

        # result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        # msg = result["messages"][-1]
        # if isinstance(msg, AIMessage):
        #     print(f"\n🤖 Bot: {msg.content}\n")
