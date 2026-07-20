# second best

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter= RecursiveCharacterTextSplitter(
     
    chunk_size=100,
    chunk_overlap=10,
)

data=PyPDFLoader("document_loaders/A.pdf")

docs=data.load()

chunks =splitter.split_documents(docs)

print(len(chunks))

print(chunks[0].page_content)
