# In this the embeddings and splits and chunks used

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter



data=PyPDFLoader("document_loaders/A.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(docs)

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
final_prompt=template.format_messages(data=docs)

response= model.invoke(final_prompt)
print(response.content)

