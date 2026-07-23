from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=5)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(""" 
        You are a helpful ai assistant.
        
        Summarize the following in in bullets.
        {news}
    """)


chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("Latest ai news of 2026")

result = chain.invoke({"news": news_result})

print(result)
