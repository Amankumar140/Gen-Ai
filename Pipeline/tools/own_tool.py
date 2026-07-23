from langchain.tools import tool

@tool
def greeting_tool(name : str)->str:
    """ This is a greeting tool"""
    
    return f"Hello How are you {name}"

print(greeting_tool.invoke("aman"))