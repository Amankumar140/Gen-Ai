from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate

data = PyPDFLoader("document_loaders/A.pdf")
docs = data.load()

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

final_prompt = template.format_messages(data=docs[1].page_content)

result = model.invoke(final_prompt)

print(result.content)
