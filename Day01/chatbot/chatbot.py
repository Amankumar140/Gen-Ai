from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_mistralai import ChatMistralAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

 

 

model = ChatMistralAI(model="mistral-small-2506")

messages=[
    SystemMessage(content="You are a funny ai assistant."),
]

print("------------- 0 to exit -------------")

while True:
    
    prompt=input("You : ")
    if prompt == "0":
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot : ", response.content)

print(messages)