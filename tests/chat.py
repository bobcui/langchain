import getpass
import os
from langchain.globals import set_verbose

set_verbose(True)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

response = model.invoke("hi there! My name is Bo.")
print("response:", response.content)

response = model.invoke("What is my name?")
print("response:", response.content)

