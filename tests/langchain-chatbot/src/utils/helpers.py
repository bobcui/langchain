def preprocess_input(user_input):
    # Function to preprocess user input
    return user_input.strip().lower()

def format_response(response):
    # Function to format the chatbot's response
    return f"Chatbot: {response}"

def manage_conversation_state(state, user_input):
    # Function to manage conversation state
    state.append(user_input)
    return state