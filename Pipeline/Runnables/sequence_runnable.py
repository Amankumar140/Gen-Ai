from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "AI"})
print(result)
