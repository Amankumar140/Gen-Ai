# second best

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter= TokenTextSplitter(
     
    chunk_size=1000,
    chunk_overlap=10,
)

data=PyPDFLoader("document_loaders/A.pdf")

docs=data.load()

chunks =splitter.split_documents(docs)

print(len(chunks))

print(chunks[0].page_content)
