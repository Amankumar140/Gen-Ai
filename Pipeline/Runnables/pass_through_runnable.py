from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

model = ChatMistralAI(model="mistral-small-2506")


parser = StrOutputParser()

code_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps to explain the code in simple words"),
    ("human", "Explain the following code in simple words:\n {code}")
])


seq= code_prompt | model | parser

seq2=RunnableParallel({
    "code":RunnablePassthrough(),
    "explain": explain_prompt | model | parser
})

chain= seq | seq2

result=chain.invoke({"topic":"Write a code for palindrome in python"})
print(result['code'])
print()
print()
print(result['explain'])