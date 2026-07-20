from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/A.pdf")

docs = data.load()

print(docs[12].page_content)
print(len(docs[12].page_content))
