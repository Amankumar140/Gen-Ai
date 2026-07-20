from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document


from langchain_core.documents import Document

docs = [
    Document(
        page_content="Neural networks are inspired by the human brain.",
        metadata={"source": "DL Book", "chapter": 1, "page": 5},
    ),
    Document(
        page_content="Backpropagation computes gradients efficiently.",
        metadata={"source": "DL Book", "chapter": 3, "page": 42},
    ),
    Document(
        page_content="Convolutional Neural Networks are widely used in computer vision.",
        metadata={"source": "DL Book", "chapter": 9, "page": 225},
    ),
]


embedding_model = MistralAIEmbeddings(model= "mistral-embed")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result=vector_store.similarity_search_by_vector("What is used widely in computer vision", k=2)


for r in result:
    print(r.page_content)