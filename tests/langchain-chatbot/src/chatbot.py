import os
from langchain import ChatOpenAI
from langchain.chains import ConversationChain

class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.conversation = ConversationChain(llm=self.llm)

    def get_response(self, user_input):
        response = self.conversation.predict(input=user_input)
        return response

def main():
    chatbot = Chatbot()
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