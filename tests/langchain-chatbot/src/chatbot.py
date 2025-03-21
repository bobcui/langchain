import os
import getpass

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

class Chatbot:
    def __init__(self, thread_id:str=None):
        self.config = {"configurable": {"thread_id": thread_id}}
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.__call_model)

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def __call_model(self, state: MessagesState):
        response = self.model.invoke(state["messages"])
        return {"messages": response}

    def get_response(self, user_input):
        input_messages = [HumanMessage(user_input)]
        output = self.app.invoke({"messages": input_messages}, self.config)
        return output["messages"][-1].content


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key: ")

    chatbot = Chatbot(thread_id="123")
    print("Welcome to the Langchain Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()