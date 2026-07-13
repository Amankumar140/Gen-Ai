from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro"
)

model= ChatHuggingFace(llm=llm)

response=model.invoke("Aruranchal Pradesh is a state in which country?")
print(response.content)