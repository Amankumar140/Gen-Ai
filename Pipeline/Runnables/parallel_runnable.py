from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

model = ChatMistralAI(model="mistral-small-2506")

short_prompt = ChatPromptTemplate.from_template("Explain {topic} in short words.")
detail_prompt = ChatPromptTemplate.from_template("Explain {topic} in detailed words.")

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"])
        | detail_prompt
        | model
        | parser,
    }
)

result = chain.invoke(
    {
        "short": {"topic": "AI"},
        "detailed": {"topic": "ML"},
    }
)
print(result["short"])
print()
print()
print()
print(result["detailed"])
