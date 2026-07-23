from dotenv import load_dotenv

load_dotenv()
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# weather tool
@tool
def get_weather(city: str) -> str:
    """_summary_
    get current weather of city

    """

    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()
    # print("DEBUG : ", data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# print(get_weather.invoke("Bangalore"))

# tool for latest news
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


@tool
def get_news(city: str) -> str:
    """Get the latest news"""

    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# print(get_news.invoke("Bangalore"))


llm = ChatMistralAI(model="mistral-small-2506")



@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call"""
    
    tool_name=request.tool_call["name"]
    confirm= input(f"Agent wants to call '{tool_name}'. Approve ?(YES?NO)")
    
    if confirm.lower() !="yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id= request.tool_call['id']
        )
    
    return handler(request)


#agent

agent = create_agent(
    llm,
    tools=[get_news, get_weather],
    system_prompt="You are a helpful city assistant.",
    middleware=[human_approval]
)

print("City Agent | Type exit to quit")

while True:
    user_input = input("You : ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

    print("Bot : ", result['messages'][-1].content)
