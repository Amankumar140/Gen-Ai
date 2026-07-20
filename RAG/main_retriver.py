from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate(
    [
        (
            "system",
            "You are the ai assistance that summarize the pdf content in 500 words",
        ),
        ("human", ("{data}")),
    ]
)


model = ChatMistralAI(model="mistral-small-2506")
final_prompt = template.format_messages(data=docs)

response = model.invoke(final_prompt)
print(response.content)
