from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

data = TextLoader("document_loaders/notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarizes the test"), ("human", "{data}")]
)


model = ChatMistralAI(model="mistral-small-2506")

final_prompt = template.format_messages(data=docs[0].page_content)


result = model.invoke(final_prompt)

print(result.content)
