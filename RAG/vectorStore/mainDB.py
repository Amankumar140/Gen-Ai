from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document_loaders/B.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)
