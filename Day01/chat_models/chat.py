from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_mistralai import ChatMistralAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

#print("Loaded:", os.getenv("OPENAI_API_KEY"))

from langchain.chat_models import init_chat_model

model = ChatMistralAI(model="mistral-small-2506")
#print(model)

response = model.invoke("what is the capital of France?")

print(response.content)