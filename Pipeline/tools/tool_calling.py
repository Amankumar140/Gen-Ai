from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print


@tool
def get_text_length(text: str) -> int:
    """Only docstring"""
    return len(text)


tools = {"get_text_length": get_text_length}

llm = ChatMistralAI(model="mistral-small-2506")

## tool binding

llm_with_tool = llm.bind_tools([get_text_length])

message = []

query = HumanMessage("Return the number of characters in text: 'hello how are you")

message.append(query)
# print(message)

result = llm_with_tool.invoke(message)
message.append(result)


### Extract the name of tool

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result = llm_with_tool.invoke(message)
print(result.content)
