from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/in/mac/"

data=WebBaseLoader(url)
docs= data.load()

print(len(docs[0].page_content))
print(docs[0].page_content)