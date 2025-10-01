import os
from typing import TypedDict, Annotated, Optional, List, Dict
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage,SystemMessage
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END,START
from langchain_core.tools import tool
import wikipedia
import datetime
import httpx
from dotenv import load_dotenv


load_dotenv()


llm = AzureChatOpenAI(  # Run on Cloud Azure
            azure_deployment="gpt-4o",
            api_key="1WrA6nOsPNNtKcYjblU0bteFPWouxvxJRWH58dnhdVSjPN6Aj5fZJQQJ99BIACHrzpqXJ3w3AAAAACOGx8Ga",
            azure_endpoint="https://eagat-mfpc8eml-northcentralus.services.ai.azure.com/",
            api_version="2024-12-01-preview",
            temperature=0
        )
        

class Request(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]
    city: str
    days: int
    category: str

def request_node(state: Request):
    return state


@tool
def get_weather_forecast_tool(city: str, days: int) -> str:
    """
    OpenWeatherMap 5-day for every 3-hour forecast -> daily high/low with a label (sunny/rainy/cloudy).
    Returns a short human-readable string summary for up to <days>.
    For example: Weather: Rain, low 28.0°C / high 28.8°C
    """
    api_key = "1b32740b0bb45d686475d181cf4a0a1f"
    api_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        with httpx.Client(timeout=20) as client:
            response = client.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        return f"Failed to fetch weather data: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

    entries = data.get("list", [])
    if not entries:
        return f"No forecast data for {city}."

    # 3-hour blocks
    per_day: Dict[str, Dict[str, list]] = {}
    for it in entries:
        dt_txt = (it.get("dt_txt") or "")
        day = dt_txt.split(" ")[0] if dt_txt else None
        if not day:
            continue
        main = it.get("main", {})
        tmin = main.get("temp_min")
        tmax = main.get("temp_max")
        cond = ((it.get("weather") or [{}])[0].get("main") or "").lower()

        bucket = per_day.setdefault(day, {"mins": [], "maxs": [], "conds": []})
        if isinstance(tmin, (int, float)):
            bucket["mins"].append(float(tmin))
        if isinstance(tmax, (int, float)):
            bucket["maxs"].append(float(tmax))
        if cond:
            bucket["conds"].append(cond)

    if not per_day:
        return f"No temperature data for {city}."

    def label(conds: List[str]) -> str:  
        if any(c in {"rain", "drizzle", "thunderstorm"} for c in conds):
            return "rain"
        if any(c == "clear" for c in conds):
            return "sunny"
        return "cloudy"

    # Build up to days lines, sorted by date
    lines = [f"Forecast for {city}:"]
    for i, day in enumerate(sorted(per_day.keys())):
        if i >= int(days):
            break
        b = per_day[day]
        if not b["mins"] or not b["maxs"]:
            continue
        lo = min(b["mins"])
        hi = max(b["maxs"])
        lab = label(b["conds"])
        lines.append(f"  {day}: {lab}, low {lo:.2f}°C / high {hi:.2f}°C")

    if len(lines) == 1:
        return f"Could not aggregate forecast for {city}."
    return "\n".join(lines)



@tool
def find_points_of_interest_tool(city: str, category: str) -> str:
    """A simple tool that returns a small sentence from  Wikipedia summary for '<category> in <city>'."""
    try:
        topic = f"{category} in {city}"
        return wikipedia.summary(topic, sentences=2)
    except Exception as e:
        return f"error occurred: {e}"

    

def agent_node(state: Request):
    """
    Invokes the model to decide whether to respond or to call a tool.
    """
    hint = SystemMessage(content=(
    "Use tools as needed, but call each tool exactly one time with the same parameters. "
    "If you already have the weather and points of interest, produce the itinerary."
))
    messages = [hint] + state["messages"]
   
    tools = [get_weather_forecast_tool, find_points_of_interest_tool] #bind tools
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages) #the llm decides from the prompt that we gave in init
    return {"messages": [response]} #return the response to the state tool call or not.LLM says whether to use a tool or not. 


def tool_node(state: Request):
    """
    Checks the last message for tool calls and executes both tools at once.
    """
    tools_used= {
        "get_weather_forecast_tool": get_weather_forecast_tool,
        "find_points_of_interest_tool": find_points_of_interest_tool,
        }
    last_message = state["messages"][-1] #latest message from the agent, it is a tool call
    tool_calls = last_message.tool_calls #calls both tools
    tool_messages = []
    for tool_call in tool_calls:
        name = tool_call.get("name")
        tool_obj = tools_used.get(name)
        print(f"Invoking Tool: {tool_call['name']}")
        result = tool_obj.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages} #executes the chosen tool updates the state with the tool message
    
def should_continue(state: Request):
    """
    Checks if the agent wants to call a tool or if it has finished.
    """
    if state["messages"][-1].tool_calls: #If it contains any tool_calls continue. After agent_node
        return "continue"
    else:
        return "end"
    



def generate_plan(state: Request):

    # Single prompt: LLM handles indoor/outdoor activities
    prompt = ("""
        You are a precise travel planner. Use the weather text that you found from the weather tool
        as it is for each day,if a forecast line contains the word 'rain', plan INDOOR activities, 
        otherwise plan OUTDOOR activities based on what you found from the find_points_of_interest_tool .\n\n
        Destination: f"{city}\nDays: {days}\nFocus on Category: {category}\n\n"
        Under the forecast of each day write exactly below two or three indoor or outdoor activities depending on the weather
        but in general try focusing more on the category that is provided.
        Weather text:\n"
        Points of interest text\n\n"
        """
    )
    try:
    # Generate answer the programm using llm
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e) or 'No details'}"
        return {"messages": f"Error generating answer: {error_msg}"}
    

# Graph
def GraphState():
    graph = StateGraph(state_schema=Request)
    graph.add_node("entry", request_node)
    graph.add_node("agent", agent_node)
    graph.add_node("action", tool_node)
    graph.add_node("generate_plan", generate_plan)

    graph.add_edge(START, "entry")
    graph.add_edge("entry", "agent") 

    graph.add_conditional_edges("agent",should_continue, {"continue": "action", "end": END})
    graph.add_edge("action", "generate_plan")
    graph.add_edge("generate_plan", END)
    return graph.compile()



if __name__=="__main__": #run it only when i run the script   
    graph =GraphState()
    city=input("Which city are you planing to visit: ")
    days=input("For how many days: ")
    category=input("What activities are you planning to do there: ")


    # Provide an initial message for the agent to reason with and decide which tool to call.
    init = {"messages": [HumanMessage(content=f"Plan my trip to {city}. First, please bring a {days} days weather forecast. Then, below the forecast of each day find a few information about {category} and depending on the forecast propose indoor or outdoor activities.")],
            "city": city,
            "days": days,
            "category": category}
    final = graph.invoke(init, {"recursion_limit": 50})
    last = final["messages"][-1]
    print(getattr(last, "content", last))
