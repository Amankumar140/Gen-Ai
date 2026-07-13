from dotenv import load_dotenv
from pathlib import Path    
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text=[
    "Arunachal Pradesh is a state in India.",   
]

embeddings=embedding.embed_documents(text)
print(embeddings)
print(len(embeddings[0]))